"""Worker xu ly 1 episode doc lap trong 1 subprocess rieng - dung de chay
song song nhieu episode tren nhieu GPU (moi worker duoc gan dung 1 GPU qua
bien moi truong CUDA_VISIBLE_DEVICES do tien trinh cha set truoc khi spawn).

Goi: python process_worker.py <media_path> <srt_path> <out_dir> <resume_dir> <episode_name>
HF_TOKEN doc tu bien moi truong (tien trinh cha truyen qua env khi Popen).
"""
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import process_episode


def main():
    media_path, srt_path, out_dir, resume_dir, episode_name = sys.argv[1:6]
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    heartbeat_sec = max(10, int(os.environ.get("GENDERSFX_HEARTBEAT_SEC", "30")))
    started_at = time.time()
    done = threading.Event()
    last_status = {"text": "dang khoi dong worker"}

    def report(msg):
        last_status["text"] = msg
        print(f"[{episode_name}] {msg}", flush=True)

    def heartbeat():
        while not done.wait(heartbeat_sec):
            elapsed = int(time.time() - started_at)
            mins, secs = divmod(elapsed, 60)
            print(
                f"[{episode_name}] van dang chay {mins}m{secs:02d}s - "
                f"{last_status['text']}",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        csv_path, srt_out_path = process_episode(
            media_path, srt_path, out_dir, hf_token,
            resume_dir=resume_dir, episode_name=episode_name, progress_cb=report,
        )
        done.set()
        print(f"[{episode_name}] XONG: {csv_path} | {srt_out_path}", flush=True)
    except Exception as exc:
        done.set()
        print(f"[{episode_name}] LOI: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
