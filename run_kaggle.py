"""Bootstrap chạy trên Kaggle: cài thư viện (chỉ nếu thiếu), đọc HF_TOKEN từ
Kaggle Secret, rồi mở app Gradio. Notebook run_kaggle.ipynb luôn tải bản mới
nhất của file này từ GitHub nên không cần sửa notebook mỗi khi code cập nhật.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working")
OUTPUT_DIR = ROOT / "gendersfx_output"


def run(cmd, **kwargs):
    print("[*]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def install_requirements(app_dir: Path):
    need_mods = ["gradio", "pyannote.audio", "speechbrain", "transformers", "pysrt"]
    if all(importlib.util.find_spec(m) for m in need_mods):
        print("[*] Thu vien Python da co, bo qua cai dat.", flush=True)
        return
    print("[*] Dang cai thu vien Python lan dau trong session nay...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(app_dir / "requirements.txt")])


def load_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError(
            "Khong tim thay HF_TOKEN. Add-ons -> Secrets -> them secret ten HF_TOKEN "
            "(token HuggingFace, da accept license pyannote/speaker-diarization-3.1 "
            "va pyannote/segmentation-3.0)."
        ) from exc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app_dir = ROOT / "detach-voice-gender"

    # Moi lan chay deu lay code moi nhat tu GitHub, nen notebook Kaggle co the giu nguyen.
    subprocess.run(["rm", "-rf", str(app_dir)], check=False)
    run(["git", "clone", "-q",
         "https://github.com/ngochoa26031-ops/detach-voice-gender.git", str(app_dir)])

    install_requirements(app_dir)

    os.environ["HF_TOKEN"] = load_hf_token()
    os.environ["GENDERSFX_OUTPUT"] = str(OUTPUT_DIR)
    os.environ["GENDERSFX_SHARE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    print(f"[*] Ket qua se luu tai: {OUTPUT_DIR}", flush=True)
    print("[*] Dang mo Gradio app. Upload media + srt truc tiep tren web UI.", flush=True)

    os.chdir(app_dir)
    run([sys.executable, "-u", "app.py"])


if __name__ == "__main__":
    main()
