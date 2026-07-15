"""Bootstrap chay tren Kaggle: cai thu vien (chi neu thieu), doc HF_TOKEN tu
Kaggle Secret, cau hinh rclone (tuy chon) de backup input/output/resume len
Google Drive, roi mo app Gradio. Notebook run_kaggle.ipynb luon tai ban moi
nhat cua file nay tu GitHub nen khong can sua notebook moi khi code cap nhat.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working")
RCLONE_CONF_PATH = Path.home() / ".config" / "rclone" / "rclone.conf"


def run(cmd, **kwargs):
    print("[*]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def _module_installed(name: str) -> bool:
    # find_spec("pyannote.audio") nem ModuleNotFoundError (thay vi tra ve None)
    # neu package cha "pyannote" chua cai - phai bat exception, khong chi check gia tri.
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def install_requirements(app_dir: Path):
    need_mods = ["gradio", "pyannote.audio", "speechbrain", "transformers", "pysrt"]
    if all(_module_installed(m) for m in need_mods):
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


def setup_rclone_from_secret():
    """Tuy chon: neu co Kaggle Secret 'RCLONE_CONF_B64', cai rclone + nap config
    de backup input/output/resume len Google Drive. Bo qua neu khong co secret nay
    hoac khong bien GENDERSFX_RCLONE_*_REMOTE nao duoc dat."""
    has_remote_env = any(
        os.environ.get(k, "").strip()
        for k in ("GENDERSFX_RCLONE_REMOTE", "GENDERSFX_RCLONE_INPUT_REMOTE",
                   "GENDERSFX_RCLONE_RESUME_REMOTE")
    )
    if not has_remote_env:
        return
    if shutil.which("rclone") is None:
        print("[*] Dang cai rclone de day file len Drive...", flush=True)
        run(["bash", "-c", "curl -s https://rclone.org/install.sh | bash"])
    try:
        from kaggle_secrets import UserSecretsClient
        import base64
        conf_b64 = UserSecretsClient().get_secret("RCLONE_CONF_B64")
        RCLONE_CONF_PATH.parent.mkdir(parents=True, exist_ok=True)
        RCLONE_CONF_PATH.write_bytes(base64.b64decode(conf_b64))
        print("[*] Da nap rclone.conf tu Kaggle Secret.", flush=True)
    except Exception as exc:
        print(f"[!] Khong nap duoc rclone.conf tu Secret ({exc}). "
              f"Se KHONG tu backup len Drive duoc.", flush=True)


def main():
    app_dir = ROOT / "detach-voice-gender-src"

    # Moi lan chay deu lay code moi nhat tu GitHub, nen notebook Kaggle co the giu nguyen.
    subprocess.run(["rm", "-rf", str(app_dir)], check=False)
    run(["git", "clone", "-q",
         "https://github.com/ngochoa26031-ops/detach-voice-gender.git", str(app_dir)])

    install_requirements(app_dir)
    setup_rclone_from_secret()

    os.environ["HF_TOKEN"] = load_hf_token()
    os.environ["GENDERSFX_ROOT"] = str(ROOT / "detach-voice-gender")
    os.environ["GENDERSFX_SHARE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    print(f"[*] Thu muc lam viec: {os.environ['GENDERSFX_ROOT']} (input/output/resume)", flush=True)
    print("[*] Dang mo Gradio app...", flush=True)

    os.chdir(app_dir)
    run([sys.executable, "-u", "app.py"])


if __name__ == "__main__":
    main()
