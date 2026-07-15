@echo off
setlocal
set RCLONE_CONF=%APPDATA%\rclone\rclone.conf

if not exist "%RCLONE_CONF%" (
    echo Khong tim thay file: %RCLONE_CONF%
    echo Hay chay "rclone config" truoc de tao remote Google Drive.
    pause
    exit /b 1
)

echo ============================================================
echo Cac remote (tai khoan Google Drive) dang co trong file config:
echo ============================================================
findstr /R "^\[.*\]$" "%RCLONE_CONF%"
echo (^^^ ten trong [ ] o tren la ten remote, dung nhu "ten:thu_muc" trong env var)
echo.
echo ============================================================
echo Chuoi base64 ben duoi la GOP CHUNG tat ca remote o tren vao 1 file.
echo No KHONG rieng cho tung remote - copy het, dan vao Kaggle Secret RCLONE_CONF_B64.
echo Chon remote nao se dung o cac bien GENDERSFX_RCLONE_*_REMOTE trong notebook,
echo vi du GENDERSFX_RCLONE_REMOTE=ten_remote:detach-voice-gender/output
echo (KHONG chia se chuoi nay cho ai, no la quyen truy cap Drive cua ban)
echo ============================================================
echo.
powershell -NoProfile -Command "[Convert]::ToBase64String([IO.File]::ReadAllBytes('%RCLONE_CONF%'))"
echo.
echo ============================================================
pause
