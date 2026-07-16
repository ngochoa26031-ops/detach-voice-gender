"""Pipeline lõi: diarization (ai nói khi nào) + gender classification (nam/nữ),
map kết quả vào từng block SRT theo overlap thời gian. Không phụ thuộc Gradio/Kaggle
để có thể test độc lập hoặc tái sử dụng từ app.py / run_kaggle.py.

Có resume: kết quả diarization (bước tốn GPU/thời gian nhất) được lưu cache theo
episode vào resume_dir; nếu bị ngắt giữa chừng, lần chạy sau dùng lại cache thay
vì chạy lại diarization từ đầu.
"""
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysrt
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pyannote.audio import Pipeline
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

GENDER_MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
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

_device = None
_diarization_pipeline = None
_processor = None
_gender_model = None


class _ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel(nn.Module):
    """Plain nn.Module (KHONG subclass Wav2Vec2PreTrainedModel/goi .from_pretrained()
    tren chinh class nay): ban transformers moi doi noi bo _finalize_model_loading()
    (doi hoi thuoc tinh all_tied_weights_keys) khong tuong thich voi custom head nay,
    crash ngay luc load. Tu tai checkpoint + load_state_dict() thu cong de khong dinh
    lieu vao duong from_pretrained() dang gay loi, bat ke ban transformers nao sau nay."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _ModelHead(config, 1)
        self.gender = _ModelHead(config, 3)  # female, male, child

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return self.age(hidden_states), self.gender(hidden_states)

    @classmethod
    def load(cls, model_name, token=None):
        config = Wav2Vec2Config.from_pretrained(model_name, token=token)
        model = cls(config)
        try:
            ckpt_path = hf_hub_download(model_name, "model.safetensors", token=token)
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        except Exception:
            ckpt_path = hf_hub_download(model_name, "pytorch_model.bin", token=token)
            state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[!] Gender model thieu key khi load (bo qua): {missing}", flush=True)
        if unexpected:
            print(f"[!] Gender model co key thua khi load (bo qua): {unexpected}", flush=True)
        return model


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
    """Tải/khởi tạo model 1 lần, dùng lại cho các lần gọi process_episode sau
    trong cùng session (HF hub cache còn giữ file trên đĩa qua các lần gọi)."""
    global _device, _diarization_pipeline, _processor, _gender_model

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

    if _gender_model is None:
        if progress_cb:
            progress_cb("Đang tải model phân loại giới tính (wav2vec2)...")
        _processor = _with_retry(
            Wav2Vec2Processor.from_pretrained, GENDER_MODEL_NAME,
            token=hf_token or None, progress_cb=progress_cb,
        )
        _gender_model = _with_retry(
            _AgeGenderModel.load, GENDER_MODEL_NAME,
            token=hf_token or None, progress_cb=progress_cb,
        ).to(_device).eval()

    return _device


def _to_wav_16k_mono(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst_path)],
        check=True, capture_output=True,
    )


def _classify_gender(audio_segment, sr):
    if len(audio_segment) < sr * 0.3:  # quá ngắn (<0.3s) để tin cậy
        return None
    inputs = _processor(audio_segment, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        _, logits = _gender_model(inputs.input_values.to(_device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs  # [female, male, child]


def _classify_gender_with_embedding(audio_segment, sr):
    if len(audio_segment) < sr * 0.3:
        return None, None
    inputs = _processor(audio_segment, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hidden = _gender_model.wav2vec2(inputs.input_values.to(_device))[0]
        pooled = torch.mean(hidden, dim=1)
        logits = _gender_model.gender(pooled)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        embedding = pooled.cpu().numpy()[0]
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding = embedding / norm
    return probs, embedding


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
        speaker_info.setdefault(speaker, (row["gender"], row["confidence"]))

    lines = []
    for speaker in sorted(by_speaker):
        gender, confidence = speaker_info.get(speaker, ("unknown", 0.0))
        gender_vi = GENDER_LABEL_VI.get(gender, gender)
        lines.append(
            f"{speaker} | {gender_vi} | conf {confidence:.3f} | "
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
    if progress_cb:
        progress_cb(
            f"Đang phân loại giới tính cho {len(speaker_names)} speaker "
            f"({total_samples} đoạn mẫu, tối đa {GENDER_MAX_SEGMENTS_PER_SPEAKER}/speaker)..."
        )
    full_audio, sr = sf.read(str(wav_path))
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)

    speaker_probs = defaultdict(list)
    speaker_embeddings = defaultdict(list)
    for speaker_idx, speaker in enumerate(speaker_names, start=1):
        turns = sampled_turns[speaker]
        if progress_cb:
            progress_cb(
                f"Phân loại {speaker} ({speaker_idx}/{len(speaker_names)}), "
                f"{len(turns)} đoạn mẫu..."
            )
        for start, end, _ in turns:
            seg = full_audio[int(start * sr):int(end * sr)]
            probs, embedding = _classify_gender_with_embedding(seg, sr)
            if probs is not None:
                speaker_probs[speaker].append(probs)
            if embedding is not None:
                speaker_embeddings[speaker].append(embedding)

    speaker_gender = {}
    for speaker, probs_list in speaker_probs.items():
        avg = np.mean(probs_list, axis=0)
        emb = None
        if speaker_embeddings.get(speaker):
            emb = np.mean(speaker_embeddings[speaker], axis=0)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
        speaker_gender[speaker] = {
            "label": GENDER_LABELS[int(np.argmax(avg))],
            "confidence": float(avg.max()),
            "embedding": emb.tolist() if emb is not None else None,
            "samples": len(probs_list),
        }

    if progress_cb:
        progress_cb("Đang gán nhãn từng block SRT...")
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    sub_speakers = _find_speakers_for_subs(subs, speaker_turns)
    results = []
    total_subs = len(subs)
    for idx, sub in enumerate(subs, start=1):
        speaker = sub_speakers[idx - 1]
        info = speaker_gender.get(speaker, {"label": "unknown", "confidence": 0.0})
        results.append({
            "index": sub.index, "start": str(sub.start), "end": str(sub.end),
            "text": sub.text, "speaker": speaker or "unknown",
            "gender": info["label"], "confidence": round(info["confidence"], 3),
        })
        if progress_cb and (idx == 1 or idx == total_subs or idx % 100 == 0):
            gender_vi = GENDER_LABEL_VI.get(info["label"], info["label"])
            progress_cb(
                f"Đã gán {idx}/{total_subs} block "
                f"(block {sub.index} -> {speaker or 'unknown'} | {gender_vi} "
                f"| conf {info['confidence']:.3f})"
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

    wav_path.unlink(missing_ok=True)
    if progress_cb:
        progress_cb("Da don file audio tam, episode local da xu ly xong.")
    return txt_path, annotated_srt_path
