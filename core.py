"""Pipeline lõi: diarization (ai nói khi nào) + gender classification (nam/nữ),
map kết quả vào từng block SRT theo overlap thời gian. Không phụ thuộc Gradio/Kaggle
để có thể test độc lập hoặc tái sử dụng từ app.py / run_kaggle.py.
"""
import csv
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysrt
import soundfile as sf
import torch
import torch.nn as nn
from pyannote.audio import Pipeline
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

GENDER_MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
GENDER_LABELS = ["female", "male", "child"]

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


class _AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _ModelHead(config, 1)
        self.gender = _ModelHead(config, 3)  # female, male, child
        self.init_weights()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return self.age(hidden_states), self.gender(hidden_states)


def ensure_models(hf_token: str, progress_cb=None):
    """Tải/khởi tạo model 1 lần, dùng lại cho các lần gọi process_episode sau
    trong cùng session (HF hub cache còn giữ file trên đĩa qua các lần gọi)."""
    global _device, _diarization_pipeline, _processor, _gender_model

    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _diarization_pipeline is None:
        if progress_cb:
            progress_cb("Đang tải model diarization (pyannote)...")
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token,
        ).to(_device)

    if _gender_model is None:
        if progress_cb:
            progress_cb("Đang tải model phân loại giới tính (wav2vec2)...")
        _processor = Wav2Vec2Processor.from_pretrained(GENDER_MODEL_NAME)
        _gender_model = _AgeGenderModel.from_pretrained(GENDER_MODEL_NAME).to(_device).eval()

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


def process_episode(media_path, srt_path, out_dir, hf_token, progress_cb=None):
    """Chạy toàn bộ pipeline cho 1 cặp media+srt, trả về (csv_path, annotated_srt_path)."""
    media_path, srt_path, out_dir = Path(media_path), Path(srt_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_models(hf_token, progress_cb)

    if progress_cb:
        progress_cb("Đang chuyển audio sang 16kHz mono...")
    wav_path = out_dir / "audio_16k.wav"
    _to_wav_16k_mono(media_path, wav_path)

    if progress_cb:
        progress_cb("Đang tách giọng theo từng speaker (diarization)...")
    diarization = _diarization_pipeline(str(wav_path))
    speaker_turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

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

    csv_path = out_dir / "gender.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "start", "end", "speaker", "gender", "confidence", "text"])
        writer.writeheader()
        writer.writerows(results)

    annotated_srt_path = out_dir / "annotated.srt"
    with open(annotated_srt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['index']}\n{r['start']} --> {r['end']}\n"
                    f"[{r['speaker']}|{r['gender']}] {r['text']}\n\n")

    wav_path.unlink(missing_ok=True)
    return csv_path, annotated_srt_path
