"""Gradio app: upload media + srt ngay trên web, bấm chạy, tải kết quả ngay trên web.
Chạy độc lập (python app.py) hoặc được run_kaggle.py gọi lại sau khi bootstrap môi trường.
"""
import os
from pathlib import Path

import gradio as gr

from core import process_episode

OUTPUT_DIR = Path(os.environ.get("GENDERSFX_OUTPUT", "./gendersfx_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()


def run(media_file, srt_file, progress=gr.Progress()):
    if not HF_TOKEN:
        raise gr.Error(
            "Thiếu HF_TOKEN. Trên Kaggle: Add-ons -> Secrets -> thêm secret tên "
            "HF_TOKEN (token HuggingFace, đã accept license pyannote/speaker-diarization-3.1 "
            "và pyannote/segmentation-3.0)."
        )
    if media_file is None or srt_file is None:
        raise gr.Error("Cần upload cả file video/audio và file .srt")

    media_path = Path(media_file if isinstance(media_file, str) else media_file.name)
    srt_path = Path(srt_file if isinstance(srt_file, str) else srt_file.name)
    out_dir = OUTPUT_DIR / media_path.stem

    def report(msg):
        progress(0, desc=msg)
        print(f"[*] {msg}", flush=True)

    csv_path, srt_out_path = process_episode(
        media_path, srt_path, out_dir, HF_TOKEN, progress_cb=report,
    )
    progress(1, desc="Xong!")
    return str(csv_path), str(srt_out_path)


with gr.Blocks(title="detach-voice-gender - Xác định giới tính người nói") as demo:
    gr.Markdown(
        "# 🎙️ detach-voice-gender\n"
        "Xác định **từng block SRT** là giọng **nam hay nữ** đang nói, dựa trên "
        "speaker diarization (pyannote) + phân loại giới tính bằng giọng nói (wav2vec2).\n\n"
        "Upload file video/audio + file .srt tương ứng, bấm chạy, tải kết quả về."
    )
    with gr.Row():
        media_in = gr.File(label="🎬 Video/Audio (mp4/mp3/wav/m4a...)")
        srt_in = gr.File(label="📝 File .srt")
    btn = gr.Button("▶ Xác định giới tính", variant="primary")
    with gr.Row():
        csv_out = gr.File(label="📥 gender.csv")
        srt_out = gr.File(label="📥 annotated.srt")
    btn.click(run, inputs=[media_in, srt_in], outputs=[csv_out, srt_out])

if __name__ == "__main__":
    demo.queue().launch(share=os.environ.get("GENDERSFX_SHARE", "1") == "1")
