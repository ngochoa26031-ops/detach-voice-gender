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
from datetime import datetime
from pathlib import Path

import gradio as gr

from core import process_episode

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

EXIT_AFTER_DONE = os.environ.get("GENDERSFX_EXIT_AFTER_DONE", "0").strip().lower() in ("1", "true", "yes", "on")
EXIT_AFTER_DONE_DELAY = max(0, int(os.environ.get("GENDERSFX_EXIT_AFTER_DONE_DELAY", "15")))
AUTO_WATCH = os.environ.get("GENDERSFX_AUTO_WATCH", "1") != "0"
AUTO_WATCH_INTERVAL = int(os.environ.get("GENDERSFX_AUTO_WATCH_SEC", "20"))


def detect_gpu_count():
    """So GPU vat ly (vd Kaggle T4 x2 -> 2). Dung de chay song song nhieu
    episode, moi episode 1 subprocess rieng gan cung 1 GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return max(1, len([line for line in result.stdout.splitlines() if line.strip()]))
    except Exception:
        return 1


GPU_WORKERS = int(os.environ.get("GENDERSFX_GPU_WORKERS", "") or detect_gpu_count())


def _rclone_available():
    return shutil.which("rclone") is not None


def _rclone_push_dir(local_dir, remote):
    if not remote or not _rclone_available() or not os.path.isdir(local_dir):
        return
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        subprocess.run(
            ["rclone", "copy", "-q", local_dir, remote] + RCLONE_RATE_LIMIT_ARGS,
            check=True, timeout=1800,
        )
        print(f"[{ts}] [*] Da day len Drive: {local_dir} -> {remote}", flush=True)
    except Exception as exc:
        print(f"[{ts}] [!] Day len Drive LOI ({local_dir}): {exc}", flush=True)


def _rclone_pull_dir(remote, local_dir):
    if not remote or not _rclone_available():
        return
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        os.makedirs(local_dir, exist_ok=True)
        result = subprocess.run(
            ["rclone", "copy", "-q", remote, local_dir] + RCLONE_RATE_LIMIT_ARGS,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0:
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
    csv_path = Path(OUTPUT_DIR) / episode_name / "gender.csv"
    return csv_path.is_file() and csv_path.stat().st_size > 0


def _episode_lock_path(episode_name):
    return Path(OUTPUT_DIR) / episode_name / ".lock"


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
            f.write(str(os.getpid()))
        return lock_path
    except FileExistsError:
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
    csv_path, srt_out_path = process_episode(
        media_path, srt_path, out_dir, HF_TOKEN,
        resume_dir=RESUME_DIR, episode_name=episode_name, progress_cb=report,
    )

    if RCLONE_REMOTE:
        _rclone_push_dir(str(out_dir), f"{RCLONE_REMOTE.rstrip('/')}/{episode_name}")
    if RCLONE_RESUME_REMOTE:
        _rclone_push_dir(os.path.join(RESUME_DIR, episode_name),
                         f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}")
    return csv_path, srt_out_path


PROCESS_LOCK = threading.Lock()


def run_ui(episode_choice, media_upload, srt_upload, progress=gr.Progress()):
    import traceback
    try:
        if not HF_TOKEN:
            raise gr.Error(
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
                    raise gr.Error(f"Khong tim thay cap file cho '{episode_choice}' trong {INPUT_DIR}")
                media_path = pairs[episode_choice]["media"]
                srt_path = pairs[episode_choice]["srt"]
                episode_name = episode_choice
            else:
                raise gr.Error(
                    "Chon 1 episode tu danh sach input, hoac upload truc tiep "
                    "ca file media va file srt."
                )

            # Phoi hop voi auto-watch (co the dang xu ly episode nay tren 1
            # subprocess GPU khac) bang lock file, tranh 2 ben cung ghi 1 out_dir.
            lock_path = _try_acquire_lock(episode_name)
            if lock_path is None:
                raise gr.Error(
                    f"'{episode_name}' dang duoc auto-watch xu ly o mot worker khac, "
                    f"doi no xong roi thu lai (xem log console de biet tien do)."
                )
            try:
                csv_path, srt_out_path = _run_pipeline(media_path, srt_path, episode_name, progress)
            finally:
                _release_lock(lock_path)
            progress(1, desc="Xong!")
            if EXIT_AFTER_DONE:
                _schedule_exit_after_done()
            return str(csv_path), str(srt_out_path), f"OK: da xu ly xong '{episode_name}'."
    except gr.Error:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return None, None, f"LOI:\n{exc}\n\n--- chi tiet ---\n{tb[-3000:]}"


def refresh_input_list():
    if RCLONE_INPUT_REMOTE:
        _rclone_pull_dir(RCLONE_INPUT_REMOTE, INPUT_DIR)
    return gr.update(choices=list_input_episodes())


def _is_persistent_drive_path(path):
    """True neu path nam trong Google Drive that (Colab mount), khong phai dia
    tam/ephemeral. Dung de CHAN xoa nham du lieu that su cua nguoi dung."""
    normalized = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    return "/drive/MyDrive/" in normalized or normalized.endswith("/drive/MyDrive")


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
        return gr.update(choices=list_input_episodes()), msg

    for d in (INPUT_DIR, OUTPUT_DIR, RESUME_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    return gr.update(choices=[]), "Da xoa du lieu cu trong input/output/resume."


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
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "worker.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    proc = subprocess.Popen(
        [sys.executable, str(WORKER_SCRIPT), str(pair["media"]), str(pair["srt"]),
         str(out_dir), RESUME_DIR, episode_name],
        stdout=log_fh, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[*] GPU {gpu_index}: khoi dong worker cho '{episode_name}' (log: {log_path})", flush=True)
    return {
        "proc": proc, "log_fh": log_fh, "log_path": log_path, "log_pos": 0,
        "lock_path": lock_path, "out_dir": out_dir, "episode_name": episode_name,
        "gpu_index": gpu_index,
    }


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
    _release_lock(worker["lock_path"])
    episode_name, out_dir, gpu_index = worker["episode_name"], worker["out_dir"], worker["gpu_index"]
    if worker["proc"].returncode == 0:
        if RCLONE_REMOTE:
            _rclone_push_dir(str(out_dir), f"{RCLONE_REMOTE.rstrip('/')}/{episode_name}")
        if RCLONE_RESUME_REMOTE:
            _rclone_push_dir(os.path.join(RESUME_DIR, episode_name),
                             f"{RCLONE_RESUME_REMOTE.rstrip('/')}/{episode_name}")
        print(f"[*] GPU {gpu_index}: '{episode_name}' xu ly xong.", flush=True)
    else:
        print(f"[!] GPU {gpu_index}: '{episode_name}' xu ly LOI (xem {worker['log_path']}), "
              f"se tu thu lai vong sau.", flush=True)


def _autowatch_loop():
    print(f"[*] Auto-watch dang bat: {INPUT_DIR} "
          f"(toi da {GPU_WORKERS} episode song song tren {GPU_WORKERS} GPU)", flush=True)
    active = {}  # gpu_index -> worker dict
    while True:
        try:
            if RCLONE_INPUT_REMOTE:
                _rclone_pull_dir(RCLONE_INPUT_REMOTE, INPUT_DIR)

            for gpu_index in list(active.keys()):
                worker = active[gpu_index]
                _tail_new_lines(worker)
                if worker["proc"].poll() is not None:
                    _finish_worker(worker)
                    del active[gpu_index]

            pending = [
                (name, pair) for name, pair in sorted(_find_episode_pairs(INPUT_DIR).items())
                if not _episode_done(name) and not _episode_lock_path(name).exists()
            ]

            for gpu_index in [i for i in range(GPU_WORKERS) if i not in active]:
                if not pending:
                    break
                name, pair = pending.pop(0)
                worker = _dispatch_worker(pair, name, gpu_index)
                if worker:
                    active[gpu_index] = worker

            if not active and not pending:
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
            csv_out = gr.File(label="gender.csv")
            srt_out = gr.File(label="annotated.srt")
            log = gr.Textbox(label="Log / Trang thai", lines=10)

    refresh_btn.click(refresh_input_list, outputs=[episode_dd])
    clear_btn.click(clear_old_data, outputs=[episode_dd, log])
    btn.click(run_ui, inputs=[episode_dd, media_in, srt_in], outputs=[csv_out, srt_out, log])

if __name__ == "__main__":
    start_autowatch()
    demo.queue().launch(share=os.environ.get("GENDERSFX_SHARE", "1") == "1")
