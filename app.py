# -*- coding: utf-8 -*-
"""
detach-voice-gender - Xac dinh tung block SRT la giong nam hay nu, tu 1 file
video/audio + srt tach bang speech-to-text.
Dung pyannote (diarization) + wav2vec2 age/gender (audeering) de phan loai.

Ca 3 thu muc input/output/resume nam chung duoi 1 thu muc cha (mac dinh
"detach-voice-gender" trong /kaggle/working hoac cwd), giong cach to chuc cua
keepsfx, de de quan ly va de backup len Google Drive qua rclone.
"""
import glob
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from core import process_episode

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
        subprocess.run(
            ["rclone", "copy", "-q", remote, local_dir] + RCLONE_RATE_LIMIT_ARGS,
            check=True, timeout=1800,
        )
        print(f"[{ts}] [*] Da keo tu Drive: {remote} -> {local_dir}", flush=True)
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

            csv_path, srt_out_path = _run_pipeline(media_path, srt_path, episode_name, progress)
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


def clear_old_data():
    for d in (INPUT_DIR, OUTPUT_DIR, RESUME_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    return gr.update(choices=[]), "Da xoa du lieu cu trong input/output/resume."


def _autowatch_loop():
    print(f"[*] Auto-watch dang bat: {INPUT_DIR} (moi {AUTO_WATCH_INTERVAL}s)", flush=True)
    while True:
        all_done_this_pass = True
        try:
            if RCLONE_INPUT_REMOTE:
                _rclone_pull_dir(RCLONE_INPUT_REMOTE, INPUT_DIR)
            for episode_name, pair in sorted(_find_episode_pairs(INPUT_DIR).items()):
                if _episode_done(episode_name):
                    continue
                all_done_this_pass = False
                print(f"[*] Auto-watch: '{episode_name}' chua co output -> tu dong xu ly...", flush=True)
                with PROCESS_LOCK:
                    try:
                        _run_pipeline(pair["media"], pair["srt"], episode_name)
                    except Exception:
                        import traceback
                        print(traceback.format_exc(), flush=True)
                        print(f"[!] Auto-watch: '{episode_name}' xu ly LOI, se tu thu lai vong sau.", flush=True)
        except Exception:
            import traceback
            print(traceback.format_exc(), flush=True)
            all_done_this_pass = False
        if all_done_this_pass:
            _schedule_exit_after_done()
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
        "hoac upload truc tiep ben duoi."
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
