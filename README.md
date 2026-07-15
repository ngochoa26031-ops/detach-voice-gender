# detach-voice-gender

Detect speaker gender (male/female) per SRT subtitle block from an mp4/mp3
recording, using speaker diarization + speech-based gender classification.
Designed to run on Kaggle Notebooks with GPU.

## Pipeline

1. **Speaker diarization** — [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
   splits the audio into "who spoke when" turns.
2. **Gender classification** — [`audeering/wav2vec2-large-robust-24-ft-age-gender`](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
   classifies each speaker's segments as female/male/child, aggregated per
   speaker for stability (not guessed block-by-block).
3. Each SRT block is mapped to the diarization turn it overlaps most, and
   inherits that speaker's gender label.

## One-time setup (per Kaggle account)

1. Add-ons → Secrets → add secret named `HF_TOKEN` with your
   [HuggingFace access token](https://huggingface.co/settings/tokens).
2. On huggingface.co, accept the user agreement for:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
3. Notebook Settings → Accelerator → GPU (e.g. T4 x2).

## Usage

1. Add Input → upload your `.mp4`/`.mp3` + matching `.srt` (any file names,
   just keep them in the same folder/dataset).
2. Click **Run All**.
3. Results appear in `/kaggle/working/output/<episode_name>/`:
   - `gender.csv` — index, start, end, speaker, gender, confidence, text
   - `annotated.srt` — original srt with `[speakerX|gender]` prefix per line

Multiple episodes can be processed in one run if uploaded as separate
input folders — the script auto-discovers every media+srt pair under
`/kaggle/input`.

## Files

- `gender_pipeline.py` — the full pipeline, paste into a Kaggle Notebook
  (cells are separated by `# %%`) or run as a single script with Run All.
- `sample/` — a 30-minute trimmed sample (audio excluded from git, srt only)
  for quick testing.
