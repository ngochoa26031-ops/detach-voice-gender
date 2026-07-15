# ============================================================================
# SRT block -> speaker gender detection, fully automatic on Kaggle.
#
# ONE-TIME SETUP (only needed the very first time you use this notebook):
#   1. Add-ons -> Secrets -> add secret named HF_TOKEN with your HuggingFace
#      access token (https://huggingface.co/settings/tokens).
#   2. On huggingface.co, accept the user agreement for:
#        - pyannote/speaker-diarization-3.1
#        - pyannote/segmentation-3.0
#   3. Settings -> Accelerator -> GPU T4 x2 (or any GPU).
#
# EVERY TIME YOU WANT TO PROCESS A NEW EPISODE:
#   - Add Input -> upload your .mp4/.mp3 + matching .srt as a Kaggle dataset
#     (or drop them into an existing input dataset folder). File names don't
#     matter, just make sure exactly one audio/video file and one .srt sit
#     together (per subfolder, if you upload several episodes at once).
#   - Click "Run All". Nothing else to edit.
#
# Output per episode appears in /kaggle/working/output/<episode_name>/:
#   - gender.csv          (index, start, end, speaker, gender, confidence, text)
#   - annotated.srt       (original srt with [speakerX|gender] prefix per line)
# ============================================================================

# %% [1] Install dependencies only if missing (fast on repeat runs)
import importlib.util
import subprocess
import sys


def ensure(pip_name, import_name=None):
    if importlib.util.find_spec(import_name or pip_name) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)


ensure("pyannote.audio", "pyannote.audio")
ensure("speechbrain")
ensure("transformers")
ensure("librosa")
ensure("pysrt")
ensure("soundfile")
# torch / torchaudio already ship in the Kaggle GPU image

# %% [2] Read HF token from Kaggle Secrets (set once via Add-ons -> Secrets)
import os

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    raise RuntimeError(
        "HF_TOKEN secret not found. Add-ons -> Secrets -> add HF_TOKEN "
        "with your HuggingFace access token, then Run All again."
    ) from e

# %% [3] Auto-discover episode(s): each is one audio/video file + one .srt
# that live in the same folder under /kaggle/input/.
from pathlib import Path

AUDIO_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".mkv"}
episodes = {}  # folder -> {"media": path, "srt": path}

for p in Path("/kaggle/input").rglob("*"):
    if p.is_dir():
        continue
    folder = p.parent
    if p.suffix.lower() == ".srt":
        episodes.setdefault(folder, {})["srt"] = p
    elif p.suffix.lower() in AUDIO_EXTS:
        episodes.setdefault(folder, {})["media"] = p

episodes = {
    folder: files for folder, files in episodes.items()
    if "media" in files and "srt" in files
}

if not episodes:
    raise RuntimeError(
        "No (audio/video + .srt) pair found under /kaggle/input. "
        "Add Input with your media file and matching .srt in the same folder."
    )

print(f"Found {len(episodes)} episode(s) to process:")
for folder, files in episodes.items():
    print(f"  - {files['media'].name} + {files['srt'].name}  (in {folder})")

# %% [4] Load models once, reused across all episodes
import torch
import torch.nn as nn
from pyannote.audio import Pipeline
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HF_TOKEN"],
).to(device)


# audeering/wav2vec2-large-robust-24-ft-age-gender: wav2vec2 fine-tuned on
# age + gender + arousal/valence, a strong open-source SOTA baseline for
# speech-based gender classification, robust across languages (works on
# Chinese speech even though trained mostly on English/German corpora,
# because it models acoustic/prosodic cues rather than language content).
class ModelHead(nn.Module):
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


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)  # female, male, child
        self.init_weights()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return self.age(hidden_states), self.gender(hidden_states)


GENDER_MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
processor = Wav2Vec2Processor.from_pretrained(GENDER_MODEL_NAME)
gender_model = AgeGenderModel.from_pretrained(GENDER_MODEL_NAME).to(device).eval()
GENDER_LABELS = ["female", "male", "child"]

# %% [5] Helper functions

import csv
from collections import defaultdict

import numpy as np
import pysrt
import soundfile as sf


def to_wav_16k_mono(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst_path)],
        check=True, capture_output=True,
    )


def classify_gender(audio_segment, sr):
    if len(audio_segment) < sr * 0.3:  # too short (<0.3s) to trust
        return None
    inputs = processor(audio_segment, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        _, logits = gender_model(inputs.input_values.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs  # [female, male, child]


def sub_to_seconds(sub_time):
    return (sub_time.hours * 3600 + sub_time.minutes * 60
            + sub_time.seconds + sub_time.milliseconds / 1000)


def find_speaker(block_start, block_end, speaker_turns):
    best_speaker, best_overlap = None, 0.0
    for t_start, t_end, speaker in speaker_turns:
        overlap = min(block_end, t_end) - max(block_start, t_start)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, speaker
    return best_speaker


def process_episode(media_path, srt_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio_16k.wav"
    to_wav_16k_mono(media_path, wav_path)

    # --- diarization: who spoke when ---
    diarization = diarization_pipeline(str(wav_path))
    speaker_turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    print(f"  {len(set(s for _, _, s in speaker_turns))} speakers, "
          f"{len(speaker_turns)} turns")

    # --- gender per speaker, aggregated across all their segments ---
    full_audio, sr = sf.read(str(wav_path))
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)

    speaker_probs = defaultdict(list)
    for start, end, speaker in speaker_turns:
        seg = full_audio[int(start * sr):int(end * sr)]
        probs = classify_gender(seg, sr)
        if probs is not None:
            speaker_probs[speaker].append(probs)

    speaker_gender = {}
    for speaker, probs_list in speaker_probs.items():
        avg = np.mean(probs_list, axis=0)
        speaker_gender[speaker] = {
            "label": GENDER_LABELS[int(np.argmax(avg))],
            "confidence": float(avg.max()),
        }
    for speaker, info in sorted(speaker_gender.items()):
        print(f"    {speaker}: {info['label']} (confidence {info['confidence']:.2f})")

    # --- map each srt block to a speaker/gender by time overlap ---
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    results = []
    for sub in subs:
        b_start = sub_to_seconds(sub.start)
        b_end = sub_to_seconds(sub.end)
        speaker = find_speaker(b_start, b_end, speaker_turns)
        info = speaker_gender.get(speaker, {"label": "unknown", "confidence": 0.0})
        results.append({
            "index": sub.index, "start": str(sub.start), "end": str(sub.end),
            "text": sub.text, "speaker": speaker or "unknown",
            "gender": info["label"], "confidence": round(info["confidence"], 3),
        })

    with open(out_dir / "gender.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "start", "end", "speaker", "gender", "confidence", "text"])
        writer.writeheader()
        writer.writerows(results)

    with open(out_dir / "annotated.srt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['index']}\n{r['start']} --> {r['end']}\n"
                    f"[{r['speaker']}|{r['gender']}] {r['text']}\n\n")

    wav_path.unlink()  # free disk, keep only csv/srt output
    return out_dir / "gender.csv", out_dir / "annotated.srt"


# %% [6] Run for every discovered episode
OUTPUT_ROOT = Path("/kaggle/working/output")

for folder, files in episodes.items():
    name = files["media"].stem
    print(f"\nProcessing: {name}")
    csv_path, srt_path_out = process_episode(files["media"], files["srt"], OUTPUT_ROOT / name)
    print(f"  -> {csv_path}")
    print(f"  -> {srt_path_out}")

print("\nAll episodes done. Download results from the Kaggle 'Output' tab "
      "under /kaggle/working/output/<episode_name>/.")
