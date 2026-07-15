"""Worker xu ly 1 episode doc lap trong 1 subprocess rieng - dung de chay
song song nhieu episode tren nhieu GPU (moi worker duoc gan dung 1 GPU qua
bien moi truong CUDA_VISIBLE_DEVICES do tien trinh cha set truoc khi spawn).

Goi: python process_worker.py <media_path> <srt_path> <out_dir> <resume_dir> <episode_name>
HF_TOKEN doc tu bien moi truong (tien trinh cha truyen qua env khi Popen).
"""
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import process_episode


def main():
    media_path, srt_path, out_dir, resume_dir, episode_name = sys.argv[1:6]
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    def report(msg):
        print(f"[{episode_name}] {msg}", flush=True)

    try:
        csv_path, srt_out_path = process_episode(
            media_path, srt_path, out_dir, hf_token,
            resume_dir=resume_dir, episode_name=episode_name, progress_cb=report,
        )
        print(f"[{episode_name}] XONG: {csv_path} | {srt_out_path}", flush=True)
    except Exception as exc:
        print(f"[{episode_name}] LOI: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
