"""Bootstrap chay tren Kaggle HOAC Colab: cai thu vien (chi neu thieu), doc
HF_TOKEN tu Secret cua tung nen tang, cau hinh luu tru lau dai roi mo app
Gradio. Notebook (run_kaggle.ipynb) luon tai ban moi nhat cua file nay tu
GitHub nen khong can sua notebook moi khi code cap nhat.

Tren Kaggle: du lieu nam trong /kaggle/working (mat khi het session), co the
bat rclone (tuy chon) de backup len Google Drive.
Tren Colab: tu mount Google Drive that (khong can rclone), du lieu nam thang
trong MyDrive/detach-voice-gender nen khong bao gio mat.
"""
import importlib.util
import os
import runpy
import signal
import shutil
import subprocess
import sys
from pathlib import Path

RCLONE_CONF_PATH = Path.home() / ".config" / "rclone" / "rclone.conf"
REPO_URL = "https://github.com/ngochoa26031-ops/detach-voice-gender.git"


def _ts():
    import time
    return time.strftime("%H:%M:%S")


def run(cmd, **kwargs):
    print(f"[{_ts()}] [*]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=kwargs.pop("timeout", None), **kwargs)
    print(f"[{_ts()}] [*] Lenh xong:", " ".join(str(x) for x in cmd), flush=True)


def run_foreground(cmd, **kwargs):
    print(f"[{_ts()}] [*]", " ".join(str(x) for x in cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, timeout=kwargs.pop("timeout", None), **kwargs)
    except subprocess.CalledProcessError as exc:
        print(f"[{_ts()}] [!] Lenh loi code {exc.returncode}: {' '.join(str(x) for x in cmd)}", flush=True)
        raise


def run_app_in_current_process(app_path: Path):
    print(f"[{_ts()}] [*] Chay app trong cung process: {app_path}", flush=True)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(app_path)]
        runpy.run_path(str(app_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def cleanup_old_tool_processes():
    """Stop/Run All tren Kaggle co the de sot worker subprocess cu.

    Khong kill python rong rai de tranh dung vao ipykernel; chi kill process co
    command line ro rang thuoc tool nay.
    """
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except Exception as exc:
        print(f"[!] Khong quet duoc process cu (bo qua): {exc}", flush=True)
        return

    current_pid = os.getpid()
    targets = []
    markers = (
        "detach-voice-gender-src/app.py",
        "detach-voice-gender-src/process_worker.py",
        "detach_voice_gender_loader/run_kaggle.py",
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if any(marker in args for marker in markers):
            targets.append((pid, args))

    if not targets:
        return

    print(f"[*] Don {len(targets)} process cu cua tool truoc khi chay lai...", flush=True)
    for pid, args in targets:
        try:
            print(f"[*] Stop process cu pid={pid}: {args[:180]}", flush=True)
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _module_installed(name: str) -> bool:
    # find_spec("pyannote.audio") nem ModuleNotFoundError (thay vi tra ve None)
    # neu package cha "pyannote" chua cai - phai bat exception, khong chi check gia tri.
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _pyannote_is_stale_pin() -> bool:
    """Ban cu tung ghim pyannote.audio<4.0 (da bo, xem git history) khong tuong
    thich torchaudio moi tren Kaggle/Colab. Neu runtime con dinh ban <4 do lan
    chay truoc, phai ep nang cap thay vi tin 'module da co la xong'."""
    try:
        from importlib.metadata import version
        return int(version("pyannote.audio").split(".")[0]) < 4
    except Exception:
        return False


def install_requirements(app_dir: Path):
    need_mods = ["gradio", "pyannote.audio", "speechbrain", "transformers", "pysrt"]
    if all(_module_installed(m) for m in need_mods) and not _pyannote_is_stale_pin():
        print("[*] Thu vien Python da co, bo qua cai dat.", flush=True)
        return
    if _pyannote_is_stale_pin():
        print("[*] pyannote.audio dang la ban cu <4.0 (khong hop torchaudio moi), nang cap...", flush=True)
    else:
        print("[*] Dang cai thu vien Python lan dau trong session nay...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "-r", str(app_dir / "requirements.txt")])


def load_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    # Colab Secrets (icon chia khoa o sidebar)
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    raise RuntimeError(
        "Khong tim thay HF_TOKEN. Tren Kaggle: Add-ons -> Secrets -> them secret "
        "ten HF_TOKEN. Tren Colab: bam icon chia khoa o sidebar trai -> them secret "
        "ten HF_TOKEN. (Token HuggingFace da accept license "
        "pyannote/speaker-diarization-3.1 va pyannote/segmentation-3.0.)"
    )


def setup_rclone_from_secret():
    """Chi dung tren Kaggle: neu co Kaggle Secret 'RCLONE_CONF_B64', cai rclone +
    nap config de backup input/output/resume len Google Drive. Bo qua neu khong
    co secret nay hoac khong bien GENDERSFX_RCLONE_*_REMOTE nao duoc dat."""
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


def detect_platform_dirs():
    """Tra ve (code_root, data_root):
    - code_root: noi git clone code, luon la dia tam thoi (khong can ben vung).
    - data_root: noi chua input/output/resume.
        + Kaggle: /kaggle/working/detach-voice-gender (mat khi het session,
          co the bat rclone backup len Drive - xem setup_rclone_from_secret).
        + Colab: tu mount Google Drive that, du lieu nam thang trong
          MyDrive/detach-voice-gender nen KHONG mat khi het session, khong
          can rclone.
        + Khac (may local): thu muc hien tai.
    """
    if Path("/kaggle/working").is_dir():
        root = Path("/kaggle/working")
        print("[*] Phat hien Kaggle.", flush=True)
        return root, root / "detach-voice-gender"

    if Path("/content").is_dir():
        print("[*] Phat hien Google Colab.", flush=True)
        drive_root = Path("/content/drive")
        my_drive = drive_root / "MyDrive"
        if not my_drive.is_dir():
            try:
                from google.colab import drive
                print("[*] Dang mount Google Drive...", flush=True)
                drive.mount(str(drive_root))
            except Exception as exc:
                print(f"[!] Khong mount duoc Google Drive ({exc}). "
                      f"Du lieu se chi luu tam trong may ao Colab, mat khi het session.",
                      flush=True)
        code_root = Path("/content")
        data_root = (my_drive if my_drive.is_dir() else code_root) / "detach-voice-gender"
        return code_root, data_root

    cwd = Path.cwd()
    return cwd, cwd / "detach-voice-gender"


def main():
    code_root, data_root = detect_platform_dirs()
    code_root.mkdir(parents=True, exist_ok=True)
    os.chdir(code_root)
    cleanup_old_tool_processes()
    app_dir = code_root / "detach-voice-gender-src"

    # Moi lan chay deu lay code moi nhat tu GitHub, nen notebook co the giu nguyen.
    print(f"[*] Xoa source cu neu co: {app_dir}", flush=True)
    subprocess.run(["rm", "-rf", str(app_dir)], check=False)
    print(f"[*] Dang clone source moi tu GitHub: {REPO_URL}", flush=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(app_dir)], timeout=300)
    print("[*] Clone source xong.", flush=True)
    subprocess.run(["git", "-C", str(app_dir), "log", "-1", "--oneline"], check=False)

    install_requirements(app_dir)
    setup_rclone_from_secret()

    os.environ["HF_TOKEN"] = load_hf_token()
    os.environ["GENDERSFX_ROOT"] = str(data_root)
    os.environ["GENDERSFX_SHARE"] = "1"
    os.environ.setdefault("GENDERSFX_HEADLESS", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"

    print(f"[*] Thu muc lam viec: {data_root} (input/output/resume)", flush=True)
    if os.environ.get("GENDERSFX_HEADLESS", "0") == "1":
        print("[*] Dang chay headless, khong mo Gradio/web UI.", flush=True)
    else:
        print("[*] Dang mo Gradio app...", flush=True)

    os.chdir(app_dir)
    run_app_in_current_process(app_dir / "app.py")


if __name__ == "__main__":
    main()
