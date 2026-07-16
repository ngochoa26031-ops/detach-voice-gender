# -*- coding: utf-8 -*-
"""
detach-voice-gender - Xac dinh tung block SRT la giong nam hay nu, tu 1 file
video/audio + srt tach bang speech-to-text.
Dung pyannote (diarization) + wav2vec2 age/gender (audeering) de phan loai.

Ca 3 thu muc input/output/resume nam chung duoi 1 thu muc cha (mac dinh
"detach-voice-gender" trong /kaggle/working hoac cwd), giong cach to chuc cua
keepsfx, de de quan ly va de backup len Google Drive qua rclone.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
import json
from datetime import datetime
from pathlib import Path

import pysrt

from core import process_episode

HEADLESS = os.environ.get("GENDERSFX_HEADLESS", "0").strip().lower() in ("1", "true", "yes", "on")
if HEADLESS:
    gr = None
else:
    import gradio as gr

WORKER_SCRIPT = Path(__file__).resolve().parent / "process_worker.py"

MEDIA_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts",
              ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()


def _default_root():
    env_root = os.environ.get("GENDERSFX_ROOT", "").strip()
    if env_root:
        return env_root
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/detach-voice-gender"
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/detach-voice-gender"
    return os.path.join(os.getcwd(), "detach-voice-gender")


ROOT_DIR = _default_root()
INPUT_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
RESUME_DIR = os.path.join(ROOT_DIR, "resume")
for _d in (INPUT_DIR, OUTPUT_DIR, RESUME_DIR):
    os.makedirs(_d, exist_ok=True)

# ====== Backup len Google Drive qua rclone (tuy chon, bat bang bien moi truong) ======
RCLONE_REMOTE = os.environ.get("GENDERSFX_RCLONE_REMOTE", "").strip()
RCLONE_INPUT_REMOTE = os.environ.get("GENDERSFX_RCLONE_INPUT_REMOTE", "").strip()
RCLONE_RESUME_REMOTE = os.environ.get("GENDERSFX_RCLONE_RESUME_REMOTE", "").strip()
RCLONE_RATE_LIMIT_ARGS = ["--fast-list", "--tpslimit", "3", "--tpslimit-burst", "1"]
RCLONE_INPUT_PULL_ARGS = RCLONE_RATE_LIMIT_ARGS + ["--ignore-existing"]

EXIT_AFTER_DONE = os.environ.get("GENDERSFX_EXIT_AFTER_DONE", "0").strip().lower() in ("1", "true", "yes", "on")
EXIT_AFTER_DONE_DELAY = max(0, int(os.environ.get("GENDERSFX_EXIT_AFTER_DONE_DELAY", "15")))
AUTO_WATCH = os.environ.get("GENDERSFX_AUTO_WATCH", "1") != "0"
AUTO_WATCH_INTERVAL = int(os.environ.get("GENDERSFX_AUTO_WATCH_SEC", "20"))
STALE_LOCK_SEC = int(os.environ.get("GENDERSFX_STALE_LOCK_SEC", "120"))
MULTI_GPU_CHUNKS = os.environ.get("GENDERSFX_MULTI_GPU_CHUNKS", "1") != "0"
SPEAKER_MATCH_THRESHOLD = float(os.environ.get("GENDERSFX_SPEAKER_MATCH_THRESHOLD", "0.86"))


def detect_gpu_count():
    """So GPU vat ly (vd Kaggle T4 x2 -> 2). Dung de chay song song nhieu
    episode, moi episode 1 subprocess rieng gan cung 1 GPU."""
    counts = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if names:
            counts.append(("nvidia-smi", len(names), ", ".join(names)))
    except Exception as exc:
        print(f"[!] Khong doc duoc GPU bang nvidia-smi: {exc}", flush=True)

    try:
        import torch
        torch_count = torch.cuda.device_count()
        if torch_count:
            counts.append(("torch", torch_count, "torch.cuda.device_count()"))
    except Exception as exc:
        print(f"[!] Khong doc duoc GPU bang torch: {exc}", flush=True)

    if counts:
        source, count, detail = max(counts, key=lambda item: item[1])
        print(f"[*] Phat hien {count} GPU bang {source}: {detail}", flush=True)
        return max(1, count)

    print("[!] Khong phat hien GPU ro rang, tam dung 1 worker.", flush=True)
    return 1


if os.environ.get("GENDERSFX_GPU_WORKERS", "").strip():
    GPU_WORKERS = int(os.environ["GENDERSFX_GPU_WORKERS"])
    print(f"[*] Ep so GPU worker tu GENDERSFX_GPU_WORKERS={GPU_WORKERS}", flush=True)
else:
    GPU_WORKERS = detect_gpu_count()


def _rclone_available():
    return shutil.which("rclone") is not None


def _snapshot_files(folder):
    root = Path(folder)
    if not root.is_dir():
        return set()
    return {
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*")
        if p.is_file()
    }


def _rclone_push_dir(local_dir, remote, label="Drive"):
    if not remote or not _rclone_available() or not os.path.isdir(local_dir):
        return
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] [*] Dang day {label} len Drive: {local_dir} -> {remote}", flush=True)
        subprocess.run(
            ["rclone", "copy", "-q", local_dir, remote] + RCLONE_RATE_LIMIT_ARGS,
            check=True, timeout=1800,
        )
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [*] Da day {label} len Drive: {local_dir} -> {remote}", flush=True)
    except Exception as exc:
        print(f"[{ts}] [!] Day {label} len Drive LOI ({local_dir}): {exc}", flush=True)


def _rclone_pull_dir(remote, local_dir, skip_existing=False):
    if not remote or not _rclone_available():
        return
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        os.makedirs(local_dir, exist_ok=True)
        before_files = _snapshot_files(local_dir) if skip_existing else set()
        extra_args = RCLONE_INPUT_PULL_ARGS if skip_existing else RCLONE_RATE_LIMIT_ARGS
        if skip_existing:
            print(f"[{ts}] [*] Dang quet Drive input: {remote}", flush=True)
        result = subprocess.run(
            ["rclone", "copy", "-q", remote, local_dir] + extra_args,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0:
            if skip_existing:
                after_files = _snapshot_files(local_dir)
                new_files = sorted(after_files - before_files)
                ts = datetime.now().strftime("%H:%M:%S")
                if new_files:
                    preview = ", ".join(new_files[:6])
                    more = f", +{len(new_files) - 6} file nua" if len(new_files) > 6 else ""
                    print(
                        f"[{ts}] [*] Quet Drive input thay {len(new_files)} file moi: "
                        f"{preview}{more}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{ts}] [*] Quet Drive input: khong thay file moi "
                        f"(file da co tren Kaggle duoc bo qua).",
                        flush=True,
                    )
            else:
                print(f"[{ts}] [*] Da keo tu Drive: {remote} -> {local_dir}", flush=True)
        elif "directory not found" in result.stderr.lower():
            # Binh thuong: episode nay chua tung backup len Drive truoc do,
            # khong phai loi that.
            print(f"[{ts}] [*] Chua co du lieu cu tren Drive cho {remote} (binh thuong lan dau).", flush=True)
        else:
            print(f"[{ts}] [!] Keo tu Drive LOI ({remote}): {result.stderr.strip()[-500:]}", flush=True)
    except Exception as exc:
        print(f"[{ts}] [!] Keo tu Drive LOI ({remote}): {exc}", flush=True)


def _schedule_exit_after_done():
    if not EXIT_AFTER_DONE:
        return

    def _exit_later():
        print(f"[*] Xu ly xong, se tu thoat sau {EXIT_AFTER_DONE_DELAY}s...", flush=True)
        time.sleep(EXIT_AFTER_DONE_DELAY)
        os._exit(0)

    threading.Thread(target=_exit_later, daemon=True).start()


def _find_episode_pairs(folder):
    """Ghep cap (media, srt) trong 1 thu muc: cung ten file (khac duoi)."""
    media_files = {}
    srt_files = {}
    for p in Path(folder).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".srt":
            srt_files[p.stem] = p
        elif p.suffix.lower() in MEDIA_EXTS:
            media_files[p.stem] = p
    return {
        stem: {"media": media_files[stem], "srt": srt_files[stem]}
        for stem in media_files
        if stem in srt_files
    }


def list_input_episodes():
    pairs = _find_episode_pairs(INPUT_DIR)
    return sorted(pairs.keys())


def _episode_done(episode_name):
    out_dir = Path(OUTPUT_DIR) / episode_name
    return any(p.is_file() and p.stat().st_size > 0 for p in out_dir.glob("*_voiceblock.txt"))


def _episode_lock_path(episode_name):
    return Path(OUTPUT_DIR) / episode_name / ".lock"


def _pid_is_running(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _read_lock(lock_path):
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        if text.startswith("{"):
            return json.loads(text)
        return {"pid": int(text)}
    except Exception:
        return {}


def _cleanup_stale_lock(episode_name):
    lock_path = _episode_lock_path(episode_name)
    if not lock_path.exists():
        return False

    info = _read_lock(lock_path)
    pid = info.get("pid")
    age = max(0, time.time() - lock_path.stat().st_mtime)
    if pid and _pid_is_running(pid):
        return False
    if pid or age >= STALE_LOCK_SEC:
        lock_path.unlink(missing_ok=True)
        print(
            f"[*] Bo lock cu cua '{episode_name}' "
            f"(pid={pid or 'unknown'}, age={int(age)}s).",
            flush=True,
        )
        return True
    return False


def _try_acquire_lock(episode_name):
    """Lock file dung PHOI HOP giua auto-watch (chay subprocess rieng) va nut
    bam tren UI (chay trong process chinh), tranh 2 ben cung xu ly 1 episode
    cung luc (ghi de/hong ket qua nhau). open(mode='x') la atomic o he thong
    file that (kho Google Drive that/dia local), du 2 tien trinh check gan
    nhu cung luc van chi 1 ben tao file thanh cong."""
    lock_path = _episode_lock_path(episode_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(lock_path, "x", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "time": time.time()}, f)
        return lock_path
    except FileExistsError:
        if _cleanup_stale_lock(episode_name):
            return _try_acquire_lock(episode_name)
        return None


def _release_lock(lock_path):
    if lock_path:
        lock_path.unlink(missing_ok=True)


def _run_pipeline(media_path, srt_path, episode_name, progress=None):
    def report(msg):
        if progress:
            progress(0, desc=msg)
        print(f"[*] {msg}", flush=True)

    if RCLONE_RESUME_REMOTE:
        _rclone_pull_dir(f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}",
                         os.path.join(RESUME_DIR, episode_name))

    out_dir = Path(OUTPUT_DIR) / episode_name
    txt_path, srt_out_path = process_episode(
        media_path, srt_path, out_dir, HF_TOKEN,
        resume_dir=RESUME_DIR, episode_name=episode_name, progress_cb=report,
    )

    if RCLONE_REMOTE:
        _rclone_push_dir(str(out_dir), f"{RCLONE_REMOTE.rstrip('/')}/{episode_name}", label="output")
    if RCLONE_RESUME_REMOTE:
        _rclone_push_dir(os.path.join(RESUME_DIR, episode_name),
                         f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}",
                         label="resume")
    return txt_path, _speaker_txt_path(out_dir, srt_path), srt_out_path


PROCESS_LOCK = threading.Lock()


def _raise_ui_error(message):
    if gr is not None:
        raise gr.Error(message)
    raise RuntimeError(message)


def run_ui(episode_choice, media_upload, srt_upload, progress=None):
    import traceback
    try:
        if not HF_TOKEN:
            _raise_ui_error(
                "Thieu HF_TOKEN. Tren Kaggle: Add-ons -> Secrets -> them secret ten "
                "HF_TOKEN (token HuggingFace, da accept license "
                "pyannote/speaker-diarization-3.1 va pyannote/segmentation-3.0)."
            )
        with PROCESS_LOCK:
            if media_upload is not None and srt_upload is not None:
                media_path = Path(media_upload if isinstance(media_upload, str) else media_upload.name)
                srt_path = Path(srt_upload if isinstance(srt_upload, str) else srt_upload.name)
                episode_name = media_path.stem
            elif episode_choice:
                pairs = _find_episode_pairs(INPUT_DIR)
                if episode_choice not in pairs:
                    _raise_ui_error(f"Khong tim thay cap file cho '{episode_choice}' trong {INPUT_DIR}")
                media_path = pairs[episode_choice]["media"]
                srt_path = pairs[episode_choice]["srt"]
                episode_name = episode_choice
            else:
                _raise_ui_error(
                    "Chon 1 episode tu danh sach input, hoac upload truc tiep "
                    "ca file media va file srt."
                )

            # Phoi hop voi auto-watch (co the dang xu ly episode nay tren 1
            # subprocess GPU khac) bang lock file, tranh 2 ben cung ghi 1 out_dir.
            lock_path = _try_acquire_lock(episode_name)
            if lock_path is None:
                _raise_ui_error(
                    f"'{episode_name}' dang duoc auto-watch xu ly o mot worker khac, "
                    f"doi no xong roi thu lai (xem log console de biet tien do)."
                )
            try:
                txt_path, speaker_path, srt_out_path = _run_pipeline(media_path, srt_path, episode_name, progress)
            finally:
                _release_lock(lock_path)
            progress(1, desc="Xong!")
            if EXIT_AFTER_DONE:
                _schedule_exit_after_done()
            return str(txt_path), str(speaker_path), str(srt_out_path), f"OK: da xu ly xong '{episode_name}'."
    except Exception as exc:
        if gr is not None and isinstance(exc, gr.Error):
            raise
        if gr is None:
            raise
        tb = traceback.format_exc()
        print(tb, flush=True)
        return None, None, None, f"LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"


def refresh_input_list():
    if RCLONE_INPUT_REMOTE:
        _rclone_pull_dir(RCLONE_INPUT_REMOTE, INPUT_DIR, skip_existing=True)
    if gr is None:
        return list_input_episodes()
    return gr.update(choices=list_input_episodes())


def _is_persistent_drive_path(path):
    """True neu path nam trong Google Drive that (Colab mount), khong phai dia
    tam/ephemeral. Dung de CHAN xoa nham du lieu that su cua nguoi dung."""
    normalized = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    return "/drive/MyDrive/" in normalized or normalized.endswith("/drive/MyDrive")


def _sub_time_to_seconds(sub_time):
    return (sub_time.hours * 3600 + sub_time.minutes * 60
            + sub_time.seconds + sub_time.milliseconds / 1000)


def _seconds_to_sub_time(seconds):
    return pysrt.SubRipTime(milliseconds=max(0, int(round(seconds * 1000))))


def _open_srt_fallback(path):
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            return pysrt.open(str(path), encoding=enc)
        except Exception:
            pass
    return pysrt.open(str(path))


def _format_ranges(indices):
    if not indices:
        return ""
    ordered = sorted(set(int(i) for i in indices))
    ranges = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _parse_range_items(text):
    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def _read_gender_txt(path):
    label_to_gender = {
        "Nam": "male",
        "Nữ": "female",
        "Nu": "female",
        "Trẻ em": "child",
        "Tre em": "child",
        "Không rõ": "unknown",
        "Khong ro": "unknown",
    }
    result = {"male": [], "female": [], "child": [], "unknown": []}
    if not Path(path).is_file():
        return result
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if ":" not in line:
            continue
        label, ranges = line.split(":", 1)
        gender = label_to_gender.get(label.strip())
        if gender:
            result[gender].extend(_parse_range_items(ranges))
    return result


def _write_gender_txt(path, by_gender):
    labels = [("male", "Nam"), ("female", "Nữ"), ("child", "Trẻ em"), ("unknown", "Không rõ")]
    lines = []
    for gender, label in labels:
        values = by_gender.get(gender, [])
        if values:
            lines.append(f"{label}: {_format_ranges(values)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _read_speaker_txt(path):
    rows = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 4 or not parts[3].startswith("blocks "):
            continue
        rows.append({
            "speaker": parts[0],
            "gender": parts[1],
            "confidence": parts[2],
            "indices": _parse_range_items(parts[3][len("blocks "):]),
        })
    return rows


def _confidence_value(text):
    try:
        return float(str(text).replace("conf", "").strip())
    except Exception:
        return 0.0


def _merge_speaker_rows(rows):
    merged = {}
    for row in rows:
        key = row["speaker"]
        item = merged.setdefault(key, {
            "speaker": key,
            "gender": row["gender"],
            "confidence": row["confidence"],
            "indices": [],
        })
        item["indices"].extend(row.get("indices", []))
        if _confidence_value(row["confidence"]) > _confidence_value(item["confidence"]):
            item["confidence"] = row["confidence"]
        if item["gender"] == "unknown" and row["gender"] != "unknown":
            item["gender"] = row["gender"]
    return [merged[key] for key in sorted(merged)]


def _write_speaker_txt(path, rows):
    lines = []
    for row in rows:
        indices = row.get("indices", [])
        if not indices:
            continue
        lines.append(
            f"{row['speaker']} | {row['gender']} | {row['confidence']} | "
            f"blocks {_format_ranges(indices)}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _find_voiceblock_txt(out_dir):
    files = sorted(Path(out_dir).glob("*_voiceblock.txt"))
    return files[0] if files else None


def _find_speaker_txt(out_dir):
    files = sorted(Path(out_dir).glob("*_speaker.txt"))
    return files[0] if files else None


def _find_speaker_embed_json(out_dir):
    files = sorted(Path(out_dir).glob("*_speaker_embed.json"))
    return files[0] if files else None


def _read_speaker_embed_json(path):
    if not path or not Path(path).is_file():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _global_speaker_map(profiles):
    clusters = []
    mapping = {}
    for profile in profiles:
        local_name = profile.get("local_speaker") or profile.get("speaker")
        gender = profile.get("gender", "unknown")
        embedding = profile.get("embedding")
        samples = max(1, int(profile.get("samples") or 1))
        if not local_name or not embedding or gender == "unknown":
            continue

        best_idx, best_score = None, -1.0
        for idx, cluster in enumerate(clusters):
            if cluster["gender"] != gender:
                continue
            score = _cosine(embedding, cluster["embedding"])
            if score > best_score:
                best_idx, best_score = idx, score

        if best_idx is None or best_score < SPEAKER_MATCH_THRESHOLD:
            clusters.append({
                "name": f"GLOBAL_SPEAKER_{len(clusters):02d}",
                "gender": gender,
                "embedding": list(embedding),
                "samples": samples,
            })
            mapping[local_name] = clusters[-1]["name"]
            continue

        cluster = clusters[best_idx]
        total = cluster["samples"] + samples
        cluster["embedding"] = [
            (old * cluster["samples"] + new * samples) / total
            for old, new in zip(cluster["embedding"], embedding)
        ]
        cluster["samples"] = total
        mapping[local_name] = cluster["name"]
    return mapping


def _voiceblock_txt_path(out_dir, srt_path):
    return Path(out_dir) / f"{Path(srt_path).stem}_voiceblock.txt"


def _speaker_txt_path(out_dir, srt_path):
    return Path(out_dir) / f"{Path(srt_path).stem}_speaker.txt"


def _chunk_subtitles_by_count(srt_path, chunk_count):
    subs = _open_srt_fallback(srt_path)
    if chunk_count <= 1 or len(subs) <= 1:
        return []

    chunk_count = min(chunk_count, len(subs))
    chunks = []
    total = len(subs)
    for idx in range(chunk_count):
        start_i = round(idx * total / chunk_count)
        end_i = round((idx + 1) * total / chunk_count)
        part_subs = subs[start_i:end_i]
        if not part_subs:
            continue
        chunks.append({
            "start": _sub_time_to_seconds(part_subs[0].start),
            "end": _sub_time_to_seconds(part_subs[-1].end),
            "subs": part_subs,
        })
    return chunks


def _write_rebased_srt(subs, chunk_start, srt_path):
    rebased = pysrt.SubRipFile()
    for sub in subs:
        item = pysrt.SubRipItem(
            index=sub.index,
            start=_seconds_to_sub_time(_sub_time_to_seconds(sub.start) - chunk_start),
            end=_seconds_to_sub_time(_sub_time_to_seconds(sub.end) - chunk_start),
            text=sub.text,
        )
        rebased.append(item)
    rebased.save(str(srt_path), encoding="utf-8")


def _cut_media_chunk(media_path, chunk_start, chunk_end, chunk_media_path):
    duration = max(0.1, chunk_end - chunk_start)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{chunk_start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(media_path), "-ac", "1", "-ar", "16000", str(chunk_media_path),
        ],
        check=True, capture_output=True,
    )


def _prepare_episode_chunks(pair, episode_name):
    chunk_count = max(1, GPU_WORKERS)
    chunks = _chunk_subtitles_by_count(pair["srt"], chunk_count)
    if len(chunks) <= 1:
        return []

    episode_out_dir = Path(OUTPUT_DIR) / episode_name
    chunk_root = episode_out_dir / "_chunks"
    shutil.rmtree(chunk_root, ignore_errors=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    prepared = []
    print(
        f"[*] '{episode_name}' chia thanh {len(chunks)} phan theo SRT "
        f"de chay song song tren {GPU_WORKERS} GPU.",
        flush=True,
    )
    for idx, chunk in enumerate(chunks, start=1):
        chunk_name = f"{episode_name}__part{idx:03d}"
        chunk_dir = chunk_root / f"part{idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_media = chunk_dir / f"{chunk_name}.wav"
        chunk_srt = chunk_dir / f"{chunk_name}.srt"
        _write_rebased_srt(chunk["subs"], chunk["start"], chunk_srt)
        _cut_media_chunk(pair["media"], chunk["start"], chunk["end"], chunk_media)
        prepared.append({
            "name": chunk_name,
            "media": chunk_media,
            "srt": chunk_srt,
            "out_dir": chunk_dir / "output",
            "resume_dir": Path(RESUME_DIR) / episode_name / "_chunks",
            "part": idx,
            "total": len(chunks),
        })
    return prepared


def _merge_chunk_outputs(episode_name, chunks, original_srt_path):
    by_gender = {"male": [], "female": [], "child": [], "unknown": []}
    speaker_rows = []
    speaker_profiles = []
    for chunk in chunks:
        chunk_txt = _find_voiceblock_txt(chunk["out_dir"])
        if chunk_txt is None:
            continue
        chunk_gender = _read_gender_txt(chunk_txt)
        for gender, values in chunk_gender.items():
            by_gender[gender].extend(values)

        chunk_speaker_txt = _find_speaker_txt(chunk["out_dir"])
        for row in _read_speaker_txt(chunk_speaker_txt) if chunk_speaker_txt else []:
            row["speaker"] = f"part{chunk['part']:03d}/{row['speaker']}"
            speaker_rows.append(row)

        chunk_embed_json = _find_speaker_embed_json(chunk["out_dir"])
        for profile in _read_speaker_embed_json(chunk_embed_json):
            local_speaker = f"part{chunk['part']:03d}/{profile.get('speaker', '')}"
            profile["local_speaker"] = local_speaker
            speaker_profiles.append(profile)

    global_map = _global_speaker_map(speaker_profiles)
    for row in speaker_rows:
        if row["speaker"] in global_map:
            row["speaker"] = global_map[row["speaker"]]
    if global_map:
        print(
            f"[*] Da khop {len(global_map)} speaker chunk -> "
            f"{len(set(global_map.values()))} global speaker "
            f"(nguong {SPEAKER_MATCH_THRESHOLD:.2f}).",
            flush=True,
        )

    out_dir = Path(OUTPUT_DIR) / episode_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_txt = _voiceblock_txt_path(out_dir, original_srt_path)
    _write_gender_txt(final_txt, by_gender)
    _write_speaker_txt(_speaker_txt_path(out_dir, original_srt_path), _merge_speaker_rows(speaker_rows))

    gender_by_index = {}
    for gender, values in by_gender.items():
        for idx in values:
            gender_by_index[int(idx)] = gender

    speaker_by_index = {}
    for row in speaker_rows:
        for idx in row.get("indices", []):
            speaker_by_index[int(idx)] = row["speaker"]

    final_srt = out_dir / "annotated.srt"
    subs = _open_srt_fallback(original_srt_path)
    with open(final_srt, "w", encoding="utf-8") as f:
        for sub in subs:
            gender = gender_by_index.get(sub.index, "unknown")
            speaker = speaker_by_index.get(sub.index, "unknown")
            f.write(f"{sub.index}\n{sub.start} --> {sub.end}\n"
                    f"[{speaker}|{gender}] {sub.text}\n\n")
    return final_txt, final_srt


def clear_old_data():
    if _is_persistent_drive_path(ROOT_DIR):
        # ROOT_DIR nam trong Google Drive that (khong phai dia tam Kaggle) - day
        # la du lieu that cua nguoi dung, TUYET DOI khong duoc rmtree. Chi don
        # cac cap media+srt trong input/ ma output/ da xu ly xong (an toan xoa
        # vi ket qua da co san trong output/), khong dung shutil.rmtree tren
        # ca thu muc de tranh mot lenh xoa sach toan bo Drive cua ho.
        removed = []
        for episode_name, pair in _find_episode_pairs(INPUT_DIR).items():
            if _episode_done(episode_name):
                for p in (pair["media"], pair["srt"]):
                    try:
                        os.remove(p)
                        removed.append(p.name)
                    except OSError:
                        pass
        msg = (f"Da xoa {len(removed)} file input da xu ly xong (con giu nguyen "
               f"output/resume vi day la Google Drive that, khong the phuc hoi neu xoa nham).")
        if gr is None:
            return list_input_episodes(), msg
        return gr.update(choices=list_input_episodes()), msg

    for d in (INPUT_DIR, OUTPUT_DIR, RESUME_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    if gr is None:
        return [], "Da xoa du lieu cu trong input/output/resume."
    return gr.update(choices=[]), "Da xoa du lieu cu trong input/output/resume."


def _start_worker(media_path, srt_path, out_dir, resume_dir, worker_name, gpu_index,
                  lock_path=None, kind="episode", job_name=None, chunk=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "worker.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT), str(media_path), str(srt_path),
         str(out_dir), str(resume_dir), worker_name],
        stdout=log_fh, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[*] GPU {gpu_index}: khoi dong worker cho '{worker_name}' (log: {log_path})", flush=True)
    return {
        "proc": proc, "log_fh": log_fh, "log_path": log_path, "log_pos": 0,
        "lock_path": lock_path, "out_dir": out_dir, "episode_name": worker_name,
        "gpu_index": gpu_index, "kind": kind, "job_name": job_name, "chunk": chunk,
    }


def _dispatch_worker(pair, episode_name, gpu_index):
    """Khoi dong 1 subprocess xu ly rieng 'episode_name', gan cung 1 GPU qua
    CUDA_VISIBLE_DEVICES=gpu_index. Tra ve None neu episode dang bi khoa (vd
    nguoi dung vua bam nut xu ly thu cong episode nay tren UI)."""
    lock_path = _try_acquire_lock(episode_name)
    if lock_path is None:
        return None

    if RCLONE_RESUME_REMOTE:
        _rclone_pull_dir(f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}",
                         os.path.join(RESUME_DIR, episode_name))

    out_dir = Path(OUTPUT_DIR) / episode_name
    return _start_worker(pair["media"], pair["srt"], out_dir, RESUME_DIR,
                         episode_name, gpu_index, lock_path=lock_path)


def _dispatch_chunk_worker(job, gpu_index):
    chunk = job["pending"].pop(0)
    worker = _start_worker(
        chunk["media"], chunk["srt"], chunk["out_dir"], chunk["resume_dir"],
        chunk["name"], gpu_index, kind="chunk", job_name=job["episode_name"],
        chunk=chunk,
    )
    print(
        f"[*] GPU {gpu_index}: '{job['episode_name']}' chunk "
        f"{chunk['part']}/{chunk['total']} dang xu ly.",
        flush=True,
    )
    return worker


def _start_chunk_job(episode_name, pair):
    lock_path = _try_acquire_lock(episode_name)
    if lock_path is None:
        return None
    try:
        if RCLONE_RESUME_REMOTE:
            _rclone_pull_dir(f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}",
                             os.path.join(RESUME_DIR, episode_name))
        chunks = _prepare_episode_chunks(pair, episode_name)
        if not chunks:
            _release_lock(lock_path)
            return None
        return {
            "episode_name": episode_name,
            "pair": pair,
            "lock_path": lock_path,
            "pending": chunks[:],
            "done": [],
            "failed": False,
        }
    except Exception:
        _release_lock(lock_path)
        raise


def _finish_chunk_job(job):
    episode_name = job["episode_name"]
    try:
        if job["failed"]:
            print(f"[!] '{episode_name}' co chunk loi, se thu lai o vong sau.", flush=True)
            return

        txt_path, srt_path = _merge_chunk_outputs(episode_name, job["done"], job["pair"]["srt"])
        print(f"[*] '{episode_name}' da gop {len(job['done'])} chunk -> {txt_path}", flush=True)
        out_dir = Path(OUTPUT_DIR) / episode_name
        if RCLONE_REMOTE:
            _rclone_push_dir(str(out_dir), f"{RCLONE_REMOTE.rstrip('/')}/{episode_name}", label="output")
        if RCLONE_RESUME_REMOTE:
            _rclone_push_dir(os.path.join(RESUME_DIR, episode_name),
                             f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}",
                             label="resume")
        print(f"[*] '{episode_name}' xu ly xong bang multi-GPU chunks: {txt_path} | {srt_path}", flush=True)
    finally:
        _release_lock(job["lock_path"])


def _tail_new_lines(worker):
    """In them nhung dong log MOI cua 1 worker ra console chinh (log cua worker
    dang bi redirect vao file rieng nen khong tu hien o day)."""
    try:
        with open(worker["log_path"], "r", encoding="utf-8", errors="ignore") as f:
            f.seek(worker["log_pos"])
            new_text = f.read()
            worker["log_pos"] = f.tell()
    except OSError:
        return
    for line in new_text.splitlines():
        if line.strip():
            print(f"[GPU{worker['gpu_index']}] {line}", flush=True)


def _finish_worker(worker):
    _tail_new_lines(worker)
    worker["log_fh"].close()
    episode_name, out_dir, gpu_index = worker["episode_name"], worker["out_dir"], worker["gpu_index"]
    if worker.get("kind") == "chunk":
        if worker["proc"].returncode == 0:
            print(
                f"[*] GPU {gpu_index}: chunk '{episode_name}' xu ly xong.",
                flush=True,
            )
        else:
            print(
                f"[!] GPU {gpu_index}: chunk '{episode_name}' xu ly LOI "
                f"(xem {worker['log_path']}).",
                flush=True,
            )
        return

    _release_lock(worker["lock_path"])
    if worker["proc"].returncode == 0:
        if RCLONE_REMOTE:
            _rclone_push_dir(str(out_dir), f"{RCLONE_REMOTE.rstrip('/')}/{episode_name}", label="output")
        if RCLONE_RESUME_REMOTE:
            _rclone_push_dir(os.path.join(RESUME_DIR, episode_name),
                             f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}",
                             label="resume")
        print(f"[*] GPU {gpu_index}: '{episode_name}' xu ly xong.", flush=True)
    else:
        print(f"[!] GPU {gpu_index}: '{episode_name}' xu ly LOI (xem {worker['log_path']}), "
              f"se tu thu lai vong sau.", flush=True)


def _autowatch_loop():
    print(f"[*] Auto-watch dang bat: {INPUT_DIR} "
          f"(toi da {GPU_WORKERS} episode song song tren {GPU_WORKERS} GPU)", flush=True)
    active = {}  # gpu_index -> worker dict
    chunk_jobs = {}
    last_input_pull = 0.0
    last_idle_log = 0.0
    while True:
        try:
            # Tick 3s de tail log worker cho muot, nhung chi keo Drive input moi
            # AUTO_WATCH_INTERVAL - keo moi tick se spam log "Da keo tu Drive"
            # lien tuc trong khi worker con dang chay lau (vd diarization).
            now = time.time()
            if RCLONE_INPUT_REMOTE and now - last_input_pull >= AUTO_WATCH_INTERVAL:
                _rclone_pull_dir(RCLONE_INPUT_REMOTE, INPUT_DIR, skip_existing=True)
                last_input_pull = now

            for gpu_index in list(active.keys()):
                worker = active[gpu_index]
                _tail_new_lines(worker)
                if worker["proc"].poll() is not None:
                    if worker.get("kind") == "chunk":
                        job = chunk_jobs.get(worker.get("job_name"))
                        if job is not None:
                            if worker["proc"].returncode == 0:
                                job["done"].append(worker["chunk"])
                            else:
                                job["failed"] = True
                    _finish_worker(worker)
                    del active[gpu_index]

            for job_name in list(chunk_jobs.keys()):
                job = chunk_jobs[job_name]
                job_active = any(
                    worker.get("job_name") == job_name
                    for worker in active.values()
                )
                if (job["failed"] or (not job["pending"] and not job_active)):
                    _finish_chunk_job(job)
                    del chunk_jobs[job_name]

            pairs = sorted(_find_episode_pairs(INPUT_DIR).items())
            pending = []
            done_count = 0
            locked = []
            for name, pair in pairs:
                if _episode_done(name):
                    done_count += 1
                    continue
                _cleanup_stale_lock(name)
                if _episode_lock_path(name).exists():
                    locked.append(name)
                    continue
                pending.append((name, pair))

            for job in list(chunk_jobs.values()):
                for gpu_index in [i for i in range(GPU_WORKERS) if i not in active]:
                    if not job["pending"] or job["failed"]:
                        break
                    active[gpu_index] = _dispatch_chunk_worker(job, gpu_index)

            for gpu_index in [i for i in range(GPU_WORKERS) if i not in active]:
                if not pending or chunk_jobs:
                    break
                name, pair = pending.pop(0)
                if MULTI_GPU_CHUNKS and GPU_WORKERS > 1:
                    job = _start_chunk_job(name, pair)
                    if job:
                        chunk_jobs[name] = job
                        while job["pending"] and gpu_index not in active:
                            active[gpu_index] = _dispatch_chunk_worker(job, gpu_index)
                        for free_gpu in [i for i in range(GPU_WORKERS) if i not in active]:
                            if not job["pending"]:
                                break
                            active[free_gpu] = _dispatch_chunk_worker(job, free_gpu)
                        break
                worker = _dispatch_worker(pair, name, gpu_index)
                if worker:
                    active[gpu_index] = worker

            if not active and not pending and not chunk_jobs:
                if pairs and now - last_idle_log >= 30:
                    print(
                        f"[*] Auto-watch: {len(pairs)} episode, "
                        f"{done_count} da xong, {len(locked)} dang bi lock, "
                        f"0 dang cho xu ly.",
                        flush=True,
                    )
                    if locked:
                        print(f"[*] Episode dang bi lock: {', '.join(locked)}", flush=True)
                    last_idle_log = now
                _schedule_exit_after_done()
                time.sleep(AUTO_WATCH_INTERVAL)
            else:
                time.sleep(3)  # dang co worker chay -> kiem tra/tail log thuong xuyen hon
        except Exception:
            import traceback
            print(traceback.format_exc(), flush=True)
            time.sleep(AUTO_WATCH_INTERVAL)


def start_autowatch():
    if not AUTO_WATCH or not HF_TOKEN:
        return
    threading.Thread(target=_autowatch_loop, daemon=True).start()


demo = None
if not HEADLESS:
    with gr.Blocks(title="detach-voice-gender") as demo:
        gr.Markdown(
            "# detach-voice-gender\n"
            "Xac dinh **tung block SRT** la giong **nam hay nu**, dua tren speaker "
            "diarization (pyannote) + phan loai gioi tinh bang giong noi (wav2vec2).\n\n"
            f"Thu muc lam viec: `{ROOT_DIR}` (gom `input/`, `output/`, `resume/`).\n"
            "Bo file media + srt (cung ten, khac duoi) vao `input/` roi bam 'Lam moi', "
            "hoac upload truc tiep ben duoi.\n\n"
            f"Phat hien **{GPU_WORKERS} GPU** - auto-watch se xu ly toi da "
            f"{GPU_WORKERS} episode cung luc, moi episode 1 GPU rieng."
        )
        with gr.Row():
            with gr.Column():
                episode_dd = gr.Dropdown(
                    choices=list_input_episodes(), value=None,
                    label=f"Chon episode da co san trong {INPUT_DIR}",
                )
                with gr.Row():
                    refresh_btn = gr.Button("Lam moi danh sach", size="sm")
                    clear_btn = gr.Button("Xoa data cu", size="sm", variant="secondary")
                with gr.Row():
                    media_in = gr.File(label="... hoac Upload Video/Audio")
                    srt_in = gr.File(label="... hoac Upload file .srt")
                btn = gr.Button("Xac dinh gioi tinh", variant="primary")
            with gr.Column():
                txt_out = gr.File(label="<ten_srt>_voiceblock.txt")
                speaker_out = gr.File(label="<ten_srt>_speaker.txt")
                srt_out = gr.File(label="annotated.srt")
                log = gr.Textbox(label="Log / Trang thai", lines=10)

        refresh_btn.click(refresh_input_list, outputs=[episode_dd])
        clear_btn.click(clear_old_data, outputs=[episode_dd, log])
        btn.click(run_ui, inputs=[episode_dd, media_in, srt_in], outputs=[txt_out, speaker_out, srt_out, log])

if __name__ == "__main__":
    if HEADLESS:
        if not HF_TOKEN:
            raise RuntimeError("Thieu HF_TOKEN, khong the chay headless.")
        print("[*] Headless mode: khong mo Gradio, chay auto-watch truc tiep trong notebook.", flush=True)
        _autowatch_loop()
    else:
        start_autowatch()
        demo.queue().launch(share=os.environ.get("GENDERSFX_SHARE", "1") == "1")
