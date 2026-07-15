"""Pipeline lõi: diarization (ai nói khi nào) + gender classification (nam/nữ),
map kết quả vào từng block SRT theo overlap thời gian. Không phụ thuộc Gradio/Kaggle
để có thể test độc lập hoặc tái sử dụng từ app.py / run_kaggle.py.

Có resume: kết quả diarization (bước tốn GPU/thời gian nhất) được lưu cache theo
episode vào resume_dir; nếu bị ngắt giữa chừng, lần chạy sau dùng lại cache thay
vì chạy lại diarization từ đầu.
"""
import csv
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
    "female": "nu",
    "male": "nam",
    "child": "tre_em",
    "unknown": "unknown",
}

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


def _short_text(text, limit=80):
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[:limit - 3] + "..."


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
    """Chạy toàn bộ pipeline cho 1 cặp media+srt, trả về (csv_path, annotated_srt_path).

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

    if progress_cb:
        progress_cb(f"Đang phân loại giới tính cho {len(set(s for _, _, s in speaker_turns))} speaker...")
    full_audio, sr = sf.read(str(wav_path))
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)

    speaker_probs = defaultdict(list)
    for start, end, speaker in speaker_turns:
        seg = full_audio[int(start * sr):int(end * sr)]
        probs = _classify_gender(seg, sr)
        if probs is not None:
            speaker_probs[speaker].append(probs)

    speaker_gender = {}
    for speaker, probs_list in speaker_probs.items():
        avg = np.mean(probs_list, axis=0)
        speaker_gender[speaker] = {
            "label": GENDER_LABELS[int(np.argmax(avg))],
            "confidence": float(avg.max()),
        }

    if progress_cb:
        progress_cb("Đang gán nhãn từng block SRT...")
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    results = []
    for sub in subs:
        b_start = _sub_to_seconds(sub.start)
        b_end = _sub_to_seconds(sub.end)
        speaker = _find_speaker(b_start, b_end, speaker_turns)
        info = speaker_gender.get(speaker, {"label": "unknown", "confidence": 0.0})
        results.append({
            "index": sub.index, "start": str(sub.start), "end": str(sub.end),
            "text": sub.text, "speaker": speaker or "unknown",
            "gender": info["label"], "confidence": round(info["confidence"], 3),
        })
        if progress_cb:
            gender_vi = GENDER_LABEL_VI.get(info["label"], info["label"])
            progress_cb(
                f"Block {sub.index} -> {speaker or 'unknown'} | {gender_vi} "
                f"| conf {info['confidence']:.3f} | {_short_text(sub.text)}"
            )

    # Ghi ra file .tmp roi os.replace() (atomic) thay vi ghi thang vao gender.csv/
    # annotated.srt: neu session bi ngat dung luc dang ghi, out_dir se khong bao
    # gio co 1 file gender.csv "do dang" nhung size > 0 - thu de bi _episode_done()
    # (app.py) hieu nham la da xu ly xong roi bo qua vinh vien o session sau.
    csv_path = out_dir / "gender.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with open(csv_tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "start", "end", "speaker", "gender", "confidence", "text"])
        writer.writeheader()
        writer.writerows(results)
    os.replace(csv_tmp, csv_path)

    annotated_srt_path = out_dir / "annotated.srt"
    srt_tmp = annotated_srt_path.with_suffix(".srt.tmp")
    with open(srt_tmp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['index']}\n{r['start']} --> {r['end']}\n"
                    f"[{r['speaker']}|{r['gender']}] {r['text']}\n\n")
    os.replace(srt_tmp, annotated_srt_path)

    wav_path.unlink(missing_ok=True)
    return csv_path, annotated_srt_path
