"""Pipeline lõi: diarization + đối chiếu thư viện Voice DNA đã verify.

Pyannote chỉ xác định ai nói khi nào. Nhãn Nam/Nữ chỉ được gán khi embedding
speaker khớp đủ chắc với một mẫu trong thư viện DNA; còn lại là unknown.

Có resume: kết quả diarization (bước tốn GPU/thời gian nhất) được lưu cache theo
episode vào resume_dir; nếu bị ngắt giữa chừng, lần chạy sau dùng lại cache thay
vì chạy lại diarization từ đầu.
"""
import json
import hashlib
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysrt
import soundfile as sf
import torch
from pyannote.audio import Pipeline

GENDER_LABELS = ["female", "male", "child"]
GENDER_LABEL_VI = {
    "female": "Nu",
    "male": "Nam",
    "child": "Tre em",
    "unknown": "unknown",
}
GENDER_TXT_LABELS = {
    "male": "Nam",
    "female": "Nữ",
    "child": "Trẻ em",
    "unknown": "Không rõ",
}
GENDER_MAX_SEGMENTS_PER_SPEAKER = int(os.environ.get("GENDERSFX_MAX_GENDER_SEGMENTS", "24"))
GENDER_MAX_SECONDS_PER_SPEAKER = float(os.environ.get("GENDERSFX_MAX_GENDER_SECONDS", "90"))
GENDER_MIN_SEGMENT_SECONDS = float(os.environ.get("GENDERSFX_MIN_GENDER_SEGMENT_SECONDS", "0.6"))
DNA_MODEL_NAME = os.environ.get("GENDERSFX_DNA_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
DNA_LIBRARY_DIR = Path(os.environ.get(
    "GENDERSFX_DNA_DIR", Path(__file__).resolve().parent / "voice_dna",
))
DNA_MATCH_THRESHOLD = float(os.environ.get("GENDERSFX_DNA_MATCH_THRESHOLD", "0.45"))
DNA_GENDER_MARGIN = float(os.environ.get("GENDERSFX_DNA_GENDER_MARGIN", "0.08"))
DNA_WINDOW_SECONDS = float(os.environ.get("GENDERSFX_DNA_WINDOW_SECONDS", "8"))
PIPELINE_VERSION = "voice_dna_v1"
PIPELINE_MARKER_NAME = ".voice_dna_v1.done"

_device = None
_diarization_pipeline = None
_speaker_encoder = None
_dna_profiles = None
_dna_fingerprint = None


def _dna_library_fingerprint():
    global _dna_fingerprint
    if _dna_fingerprint is not None:
        return _dna_fingerprint
    digest = hashlib.sha256(PIPELINE_VERSION.encode("utf-8"))
    config_path = DNA_LIBRARY_DIR / "library.json"
    digest.update(config_path.read_bytes())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for group in config.get("groups", []):
        wav_path = DNA_LIBRARY_DIR / str(group.get("file", ""))
        digest.update(wav_path.name.encode("utf-8"))
        digest.update(wav_path.read_bytes())
    _dna_fingerprint = digest.hexdigest()
    return _dna_fingerprint


def _with_retry(fn, *args, retries=5, base_delay=10, progress_cb=None, **kwargs):
    """HuggingFace hub hay tra ve 429 (rate limit) khi nhieu nguoi cung tai model
    tu chung 1 dai IP (vd Kaggle). Thu lai voi backoff thay vi chet ngay."""
    import time
    from huggingface_hub.utils import HfHubHTTPError

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as exc:
            last_exc = exc
            is_429 = getattr(exc.response, "status_code", None) == 429
            if not is_429 or attempt == retries:
                raise
            delay = base_delay * attempt
            msg = f"HuggingFace tra ve 429 (rate limit), thu lai sau {delay}s ({attempt}/{retries})..."
            if progress_cb:
                progress_cb(msg)
            print(f"[!] {msg}", flush=True)
            time.sleep(delay)
    raise last_exc


def ensure_models(hf_token: str, progress_cb=None):
    """Tải diarization và encoder nhận dạng giọng một lần mỗi process."""
    global _device, _diarization_pipeline, _speaker_encoder

    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hf_token:
        # Dang nhap HF de MOI request (ke ca cua transformers, khong chi pyannote)
        # deu duoc xac thuc bang token thay vi tinh la request an danh - request
        # an danh de bi rate-limit 429 hon nhieu so voi request co token.
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception as exc:
            print(f"[!] huggingface_hub login loi (bo qua, van dung token qua env): {exc}", flush=True)

    if _diarization_pipeline is None:
        if progress_cb:
            progress_cb("Đang tải model diarization (pyannote)...")
        # pyannote.audio doi ten tham so use_auth_token -> token o ban moi; thu
        # token truoc, neu ban cu chua ho tro (TypeError) thi fallback use_auth_token.
        try:
            pipeline = _with_retry(
                Pipeline.from_pretrained, "pyannote/speaker-diarization-3.1",
                token=hf_token, progress_cb=progress_cb,
            )
        except TypeError:
            pipeline = _with_retry(
                Pipeline.from_pretrained, "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token, progress_cb=progress_cb,
            )
        _diarization_pipeline = pipeline.to(_device)

    if _speaker_encoder is None:
        if progress_cb:
            progress_cb("Đang tải model nhận dạng Voice DNA (ECAPA)...")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        _speaker_encoder = EncoderClassifier.from_hparams(
            source=DNA_MODEL_NAME,
            run_opts={"device": str(_device)},
        )

    return _device


def _to_wav_16k_mono(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst_path)],
        check=True, capture_output=True,
    )


def _encode_voice_segments(audio_segments, sr):
    """Tạo ECAPA embedding chuẩn hóa cho nhiều đoạn audio trong ít batch."""
    if sr != 16000:
        raise ValueError(f"Voice DNA can audio 16kHz, nhan duoc {sr}Hz")

    segments = []
    max_samples = max(1, int(DNA_WINDOW_SECONDS * sr))
    min_samples = int(GENDER_MIN_SEGMENT_SECONDS * sr)
    for segment in audio_segments:
        audio = np.asarray(segment, dtype=np.float32).reshape(-1)
        if len(audio) < min_samples:
            continue
        for start in range(0, len(audio), max_samples):
            chunk = audio[start:start + max_samples]
            if len(chunk) >= min_samples:
                segments.append(chunk)

    embeddings = []
    batch_size = 8
    for offset in range(0, len(segments), batch_size):
        batch = segments[offset:offset + batch_size]
        longest = max(len(item) for item in batch)
        waveforms = torch.zeros((len(batch), longest), dtype=torch.float32)
        lengths = torch.zeros(len(batch), dtype=torch.float32)
        for idx, item in enumerate(batch):
            waveforms[idx, :len(item)] = torch.from_numpy(item)
            lengths[idx] = len(item) / longest
        with torch.no_grad():
            encoded = _speaker_encoder.encode_batch(
                waveforms.to(_device), lengths.to(_device), normalize=True,
            )
        encoded = encoded.detach().cpu().numpy().reshape(len(batch), -1)
        embeddings.extend(encoded)
    return embeddings


def _mean_embedding(embeddings):
    if not embeddings:
        return None
    embedding = np.mean(embeddings, axis=0)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0 else None


def _load_dna_profiles(progress_cb=None):
    global _dna_profiles
    if _dna_profiles is not None:
        return _dna_profiles

    config_path = DNA_LIBRARY_DIR / "library.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Khong tim thay thu vien Voice DNA: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    groups = config.get("groups", [])
    profiles = []
    if progress_cb:
        progress_cb(f"Đang nạp {len(groups)} mẫu Voice DNA đã verify...")
    for group in groups:
        name = str(group.get("name", "")).strip()
        gender = str(group.get("gender", "")).strip().lower()
        wav_path = DNA_LIBRARY_DIR / str(group.get("file", ""))
        if not name or gender not in ("male", "female") or not wav_path.is_file():
            raise ValueError(f"Voice DNA khong hop le: {group}")
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        embedding = _mean_embedding(_encode_voice_segments([audio], sr))
        if embedding is None:
            raise ValueError(f"Voice DNA khong du audio de tao embedding: {name}")
        profiles.append({
            "name": name,
            "gender": gender,
            "embedding": embedding,
        })
    _dna_profiles = profiles
    return profiles


def _cosine_similarity(left, right):
    return float(np.dot(left, right))


def _match_voice_dna(embedding, dna_profiles):
    """Chỉ nhận nhãn khi đủ ngưỡng và tách rõ DNA của giới tính đối diện."""
    ranked = sorted(
        ((profile, _cosine_similarity(embedding, profile["embedding"]))
         for profile in dna_profiles),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return "unknown", 0.0, None, 0.0

    best, best_score = ranked[0]
    opposite_scores = [
        score for profile, score in ranked if profile["gender"] != best["gender"]
    ]
    opposite_score = max(opposite_scores, default=-1.0)
    accepted = (
        best_score >= DNA_MATCH_THRESHOLD
        and best_score - opposite_score >= DNA_GENDER_MARGIN
    )
    return (
        best["gender"] if accepted else "unknown",
        best_score,
        best["name"],
        best_score - opposite_score,
    )


def _sample_turns_for_gender(speaker_turns):
    """Chon mau turn dai/ro nhat cho tung speaker.

    Diarization cua file dai co the tao hang nghin turn nho. Phan loai gender
    tren tat ca turn lam buoc nay cham kinh khung, trong khi chi can mot mau
    du lon cho moi speaker la du on dinh.
    """
    grouped = defaultdict(list)
    for start, end, speaker in speaker_turns:
        duration = max(0.0, end - start)
        if duration >= GENDER_MIN_SEGMENT_SECONDS:
            grouped[speaker].append((duration, start, end, speaker))

    sampled = {}
    for speaker, turns in grouped.items():
        chosen = []
        total_seconds = 0.0
        for duration, start, end, _ in sorted(turns, reverse=True):
            if len(chosen) >= GENDER_MAX_SEGMENTS_PER_SPEAKER:
                break
            if total_seconds >= GENDER_MAX_SECONDS_PER_SPEAKER:
                break
            chosen.append((start, end, speaker))
            total_seconds += duration
        sampled[speaker] = sorted(chosen)
    return sampled


def _sub_to_seconds(sub_time):
    return (sub_time.hours * 3600 + sub_time.minutes * 60
            + sub_time.seconds + sub_time.milliseconds / 1000)


def _find_speaker(block_start, block_end, speaker_turns):
    best_speaker, best_overlap = None, 0.0
    for t_start, t_end, speaker in speaker_turns:
        overlap = min(block_end, t_end) - max(block_start, t_start)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, speaker
    return best_speaker


def _find_speakers_for_subs(subs, speaker_turns):
    """Gan speaker cho tat ca subtitle bang mot lan quet timeline.

    Cach cu goi _find_speaker() cho tung block va moi lan lai scan toan bo
    diarization turns. Voi file dai, day la doan CPU cham nhat sau gender.
    """
    turns = sorted(speaker_turns, key=lambda item: item[0])
    speakers = []
    active_start = 0
    for sub in subs:
        block_start = _sub_to_seconds(sub.start)
        block_end = _sub_to_seconds(sub.end)
        while active_start < len(turns) and turns[active_start][1] <= block_start:
            active_start += 1

        best_speaker, best_overlap = None, 0.0
        idx = active_start
        while idx < len(turns) and turns[idx][0] < block_end:
            t_start, t_end, speaker = turns[idx]
            overlap = min(block_end, t_end) - max(block_start, t_start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
            idx += 1
        speakers.append(best_speaker)
    return speakers


def _short_text(text, limit=80):
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[:limit - 3] + "..."


def _format_ranges(indices):
    if not indices:
        return ""
    sorted_indices = sorted(int(i) for i in indices)
    ranges = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _write_gender_ranges_txt(txt_path, results):
    by_gender = defaultdict(list)
    for row in results:
        by_gender[row["gender"]].append(row["index"])

    lines = []
    for gender in ("male", "female", "child", "unknown"):
        if gender not in by_gender:
            continue
        label = GENDER_TXT_LABELS.get(gender, gender)
        lines.append(f"{label}: {_format_ranges(by_gender[gender])}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _write_speaker_ranges_txt(txt_path, results):
    by_speaker = defaultdict(list)
    speaker_info = {}
    for row in results:
        speaker = row["speaker"]
        by_speaker[speaker].append(row["index"])
        speaker_info.setdefault(speaker, (
            row["gender"], row["confidence"], row.get("dna_name"),
            row.get("dna_margin", 0.0),
        ))

    lines = []
    for speaker in sorted(by_speaker):
        gender, confidence, dna_name, dna_margin = speaker_info.get(
            speaker, ("unknown", 0.0, None, 0.0),
        )
        gender_vi = GENDER_LABEL_VI.get(gender, gender)
        match_name = dna_name or "NO_MATCH"
        lines.append(
            f"{speaker} | {gender_vi} | dna {match_name} score {confidence:.3f} "
            f"margin {dna_margin:.3f} | "
            f"blocks {_format_ranges(by_speaker[speaker])}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _write_speaker_embeddings_json(json_path, speaker_profiles):
    rows = []
    for speaker in sorted(speaker_profiles):
        profile = speaker_profiles[speaker]
        embedding = profile.get("embedding")
        if embedding is None:
            continue
        rows.append({
            "speaker": speaker,
            "gender": profile["label"],
            "confidence": round(profile["confidence"], 6),
            "dna_name": profile.get("dna_name"),
            "dna_margin": round(float(profile.get("dna_margin", 0.0)), 6),
            "samples": profile.get("samples", 0),
            "embedding": [round(float(x), 8) for x in embedding],
        })
    json_path.write_text(json.dumps(rows), encoding="utf-8")


def _voiceblock_txt_path(out_dir, srt_path):
    return Path(out_dir) / f"{Path(srt_path).stem}_voiceblock.txt"


def _speaker_txt_path(out_dir, srt_path):
    return Path(out_dir) / f"{Path(srt_path).stem}_speaker.txt"


def _speaker_embed_json_path(out_dir, srt_path):
    return Path(out_dir) / f"{Path(srt_path).stem}_speaker_embed.json"


def _extract_speaker_turns(diarization):
    """Lay danh sach turn tu output pyannote 3.x/4.x.

    pyannote.audio 3.x tra ve Annotation co itertracks(). pyannote.audio 4.x
    tra ve DiarizeOutput, Annotation nam trong speaker_diarization.
    """
    annotation = getattr(diarization, "speaker_diarization", diarization)
    if not hasattr(annotation, "itertracks"):
        raise TypeError(
            "Khong doc duoc output diarization cua pyannote: thieu itertracks() "
            "va speaker_diarization."
        )
    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _diarization_cache_path(resume_dir, episode_name):
    if not resume_dir:
        return None
    return Path(resume_dir) / episode_name / "diarization.json"


def _load_cached_diarization(cache_path):
    if not cache_path or not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [(t["start"], t["end"], t["speaker"]) for t in data]
    except Exception:
        return None


def _save_diarization_cache(cache_path, speaker_turns):
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"start": s, "end": e, "speaker": sp} for s, e, sp in speaker_turns]
    cache_path.write_text(json.dumps(data), encoding="utf-8")


def _gender_cache_path(resume_dir, episode_name):
    if not resume_dir:
        return None
    fingerprint = _dna_library_fingerprint()[:12]
    return Path(resume_dir) / episode_name / f"dna_profiles_v1_{fingerprint}.json"


def _load_cached_gender_profiles(cache_path):
    if not cache_path or not cache_path.is_file():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    profiles = {}
    for speaker, profile in data.items():
        if not isinstance(profile, dict):
            continue
        label = profile.get("label")
        if label not in (*GENDER_LABELS, "unknown"):
            continue
        profiles[speaker] = {
            "label": label,
            "confidence": float(profile.get("confidence", 0.0)),
            "embedding": profile.get("embedding"),
            "samples": int(profile.get("samples", 0)),
            "dna_name": profile.get("dna_name"),
            "dna_margin": float(profile.get("dna_margin", 0.0)),
        }
    return profiles


def _save_gender_profiles_cache(cache_path, profiles):
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for speaker, profile in profiles.items():
        data[speaker] = {
            "label": profile["label"],
            "confidence": round(float(profile.get("confidence", 0.0)), 6),
            "embedding": profile.get("embedding"),
            "samples": int(profile.get("samples", 0)),
            "dna_name": profile.get("dna_name"),
            "dna_margin": round(float(profile.get("dna_margin", 0.0)), 6),
        }
    tmp_path = cache_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp_path, cache_path)


def process_episode(media_path, srt_path, out_dir, hf_token, resume_dir=None,
                     episode_name=None, progress_cb=None):
    """Chạy toàn bộ pipeline cho 1 cặp media+srt, trả về (txt_path, annotated_srt_path).

    resume_dir: nếu truyền vào, kết quả diarization được cache theo episode_name
    (mặc định = tên file media không đuôi) để lần chạy sau tái sử dụng nếu bị
    ngắt giữa chừng (ví dụ mất session Kaggle) thay vì chạy lại từ đầu.
    """
    media_path, srt_path, out_dir = Path(media_path), Path(srt_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_name = episode_name or media_path.stem

    ensure_models(hf_token, progress_cb)

    if progress_cb:
        progress_cb("Đang chuyển audio sang 16kHz mono...")
    wav_path = out_dir / "audio_16k.wav"
    _to_wav_16k_mono(media_path, wav_path)

    cache_path = _diarization_cache_path(resume_dir, episode_name)
    speaker_turns = _load_cached_diarization(cache_path)
    if speaker_turns is not None:
        if progress_cb:
            progress_cb("Dùng lại kết quả diarization đã lưu (resume)...")
    else:
        if progress_cb:
            progress_cb("Đang tách giọng theo từng speaker (diarization)...")
        # ProgressHook cua pyannote in tien do tung buoc (segmentation/embedding/
        # clustering) trong luc chay - neu khong co hook, lenh nay chay lien 1
        # mach khong in gi ca cho toi khi xong, de nham la bi treo voi file dai.
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                diarization = _diarization_pipeline(str(wav_path), hook=hook)
        except ImportError:
            diarization = _diarization_pipeline(str(wav_path))
        speaker_turns = _extract_speaker_turns(diarization)
        _save_diarization_cache(cache_path, speaker_turns)

    sampled_turns = _sample_turns_for_gender(speaker_turns)
    speaker_names = sorted(sampled_turns)
    total_samples = sum(len(turns) for turns in sampled_turns.values())
    dna_profiles = _load_dna_profiles(progress_cb)
    gender_cache_path = _gender_cache_path(resume_dir, episode_name)
    speaker_gender = _load_cached_gender_profiles(gender_cache_path)
    missing_speakers = [speaker for speaker in speaker_names if speaker not in speaker_gender]
    if progress_cb:
        cached_count = len(speaker_names) - len(missing_speakers)
        progress_cb(
            f"Đang đối chiếu Voice DNA cho {len(speaker_names)} speaker "
            f"({total_samples} đoạn mẫu, tối đa {GENDER_MAX_SEGMENTS_PER_SPEAKER}/speaker, "
            f"resume {cached_count}/{len(speaker_names)})..."
        )
    if missing_speakers:
        full_audio, sr = sf.read(str(wav_path))
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=1)

        for speaker_idx, speaker in enumerate(speaker_names, start=1):
            if speaker not in missing_speakers:
                continue
            turns = sampled_turns[speaker]
            if progress_cb:
                progress_cb(
                    f"Đối chiếu DNA {speaker} ({speaker_idx}/{len(speaker_names)}), "
                    f"{len(turns)} đoạn mẫu..."
                )
            segments = [
                full_audio[int(start * sr):int(end * sr)]
                for start, end, _ in turns
            ]
            embeddings = _encode_voice_segments(segments, sr)
            emb = _mean_embedding(embeddings)
            if emb is None:
                speaker_gender[speaker] = {
                    "label": "unknown", "confidence": 0.0,
                    "embedding": None, "samples": 0,
                    "dna_name": None, "dna_margin": 0.0,
                }
                _save_gender_profiles_cache(gender_cache_path, speaker_gender)
                continue

            label, score, dna_name, dna_margin = _match_voice_dna(emb, dna_profiles)
            speaker_gender[speaker] = {
                "label": label,
                "confidence": score,
                "embedding": emb.tolist(),
                "samples": len(embeddings),
                "dna_name": dna_name,
                "dna_margin": dna_margin,
            }
            if progress_cb:
                gender_vi = GENDER_LABEL_VI.get(label, label)
                progress_cb(
                    f"DNA {speaker}: {gender_vi}, mau {dna_name or 'NO_MATCH'}, "
                    f"score {score:.3f}, margin {dna_margin:.3f}"
                )
            _save_gender_profiles_cache(gender_cache_path, speaker_gender)
    elif progress_cb:
        progress_cb("Dùng lại toàn bộ Voice DNA profiles đã lưu (resume).")

    if progress_cb:
        progress_cb("Đang gán nhãn từng block SRT...")
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    sub_speakers = _find_speakers_for_subs(subs, speaker_turns)
    results = []
    total_subs = len(subs)
    for idx, sub in enumerate(subs, start=1):
        speaker = sub_speakers[idx - 1]
        info = speaker_gender.get(speaker, {
            "label": "unknown", "confidence": 0.0, "dna_name": None,
            "dna_margin": 0.0,
        })
        results.append({
            "index": sub.index, "start": str(sub.start), "end": str(sub.end),
            "text": sub.text, "speaker": speaker or "unknown",
            "gender": info["label"], "confidence": round(info["confidence"], 3),
            "dna_name": info.get("dna_name"),
            "dna_margin": round(float(info.get("dna_margin", 0.0)), 3),
        })
        if progress_cb and (idx == 1 or idx == total_subs or idx % 100 == 0):
            gender_vi = GENDER_LABEL_VI.get(info["label"], info["label"])
            progress_cb(
                f"Đã gán {idx}/{total_subs} block "
                f"(block {sub.index} -> {speaker or 'unknown'} | {gender_vi} "
                f"| DNA {info.get('dna_name') or 'NO_MATCH'} "
                f"| score {info['confidence']:.3f})"
            )

    # Ghi ra file .tmp roi os.replace() (atomic) thay vi ghi thang vao *_voiceblock.txt/
    # annotated.srt: neu session bi ngat dung luc dang ghi, out_dir se khong bao
    # gio co 1 file voiceblock "do dang" nhung size > 0 - thu de bi _episode_done()
    # (app.py) hieu nham la da xu ly xong roi bo qua vinh vien o session sau.
    txt_path = _voiceblock_txt_path(out_dir, srt_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    _write_gender_ranges_txt(txt_tmp, results)
    os.replace(txt_tmp, txt_path)
    if progress_cb:
        progress_cb(f"Da ghi ket qua TXT: {txt_path}")

    speaker_txt_path = _speaker_txt_path(out_dir, srt_path)
    speaker_tmp = speaker_txt_path.with_suffix(".txt.tmp")
    _write_speaker_ranges_txt(speaker_tmp, results)
    os.replace(speaker_tmp, speaker_txt_path)
    if progress_cb:
        progress_cb(f"Da ghi speaker TXT: {speaker_txt_path}")

    speaker_embed_path = _speaker_embed_json_path(out_dir, srt_path)
    speaker_embed_tmp = speaker_embed_path.with_suffix(".json.tmp")
    _write_speaker_embeddings_json(speaker_embed_tmp, speaker_gender)
    os.replace(speaker_embed_tmp, speaker_embed_path)

    annotated_srt_path = out_dir / "annotated.srt"
    srt_tmp = annotated_srt_path.with_suffix(".srt.tmp")
    with open(srt_tmp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['index']}\n{r['start']} --> {r['end']}\n"
                    f"[{r['speaker']}|{r['gender']}] {r['text']}\n\n")
    os.replace(srt_tmp, annotated_srt_path)
    if progress_cb:
        progress_cb(f"Da ghi annotated SRT: {annotated_srt_path}")

    (out_dir / PIPELINE_MARKER_NAME).write_text(
        _dna_library_fingerprint(), encoding="ascii",
    )
    wav_path.unlink(missing_ok=True)
    if progress_cb:
        progress_cb("Da don file audio tam, episode local da xu ly xong.")
    return txt_path, annotated_srt_path
