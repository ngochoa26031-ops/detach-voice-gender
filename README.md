# detach-voice-gender

Xác định **từng block SRT** là giọng **nam hay nữ** đang nói, từ 1 file
video/audio + srt tách bằng speech-to-text (tiếng Trung hay ngôn ngữ khác đều
được, vì model dựa trên đặc trưng âm học chứ không phải nội dung lời nói).

- **Speaker diarization** — [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  tách audio thành "ai nói khi nào".
- **Gender classification** — [`audeering/wav2vec2-large-robust-24-ft-age-gender`](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
  phân loại nam/nữ/trẻ em cho từng speaker, gộp xác suất qua tất cả đoạn nói
  của speaker đó để ổn định (không đoán riêng lẻ từng block).
- Mỗi block SRT được gán vào turn diarization overlap nhiều nhất, thừa hưởng
  nhãn giới tính của speaker đó.
- **Resume:** kết quả diarization (bước tốn thời gian/GPU nhất) được lưu cache
  theo từng episode; nếu phiên Kaggle bị ngắt giữa chừng, lần chạy sau dùng
  lại cache thay vì tách giọng lại từ đầu.

## Cấu trúc thư mục làm việc

Cả 3 thư mục nằm chung dưới **1 thư mục cha `detach-voice-gender`** (mặc định
`/kaggle/working/detach-voice-gender` trên Kaggle):

```
detach-voice-gender/
├── input/     bỏ file media + srt (cùng tên, khác đuôi) vào đây
├── output/    gender.csv + annotated.srt của từng episode
└── resume/    cache diarization để resume nếu bị ngắt giữa chừng
```

> ⚠️ `/kaggle/working` mất khi session Kaggle kết thúc. Nếu không cấu hình
> backup Google Drive (xem bên dưới), nhớ tải `output/` về máy trước khi dừng
> session.

---

## 🚀 Chạy trên Kaggle (upload → chạy → tải về)

1. Tạo notebook mới trên Kaggle (hoặc import [`run_kaggle.ipynb`](run_kaggle.ipynb)).
2. **Add-ons → Secrets** → thêm secret tên `HF_TOKEN` (token HuggingFace,
   xem [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
   Nhớ **accept license** cho `pyannote/speaker-diarization-3.1` và
   `pyannote/segmentation-3.0` trên trang HuggingFace của 2 model đó.
   *(Chỉ cần làm 1 lần cho mỗi tài khoản Kaggle.)*
3. `Settings → Accelerator → GPU`, `Settings → Internet → On`.
4. Bấm **Run**. Notebook tự clone code mới nhất, cài thư viện (chỉ lần đầu
   trong session), mở app **Gradio**.
5. Trong app: chọn episode từ danh sách (nếu đã bỏ file vào `input/`), hoặc
   **upload trực tiếp cặp file media+srt**, bấm **Xác định giới tính**, tải
   `gender.csv` và `annotated.srt` ngay trên web khi xong.

## (Tùy chọn) Backup lên Google Drive qua rclone

Không bắt buộc, nhưng giúp không mất dữ liệu khi hết session Kaggle và tránh
tải lại model resume. Cách bật:

1. Trên máy local: chạy `rclone config` để tạo 1 remote Google Drive (chỉ 1 lần).
2. Chạy [`get_rclone_secret.bat`](get_rclone_secret.bat) → copy chuỗi base64
   → Kaggle **Add-ons → Secrets** → tạo secret tên `RCLONE_CONF_B64`, dán vào.
3. Trong notebook, set các biến môi trường trước khi gọi `run_kaggle.py` (tùy
   chọn dùng phần nào):
   ```python
   import os
   os.environ["GENDERSFX_RCLONE_REMOTE"] = "ten_remote:detach-voice-gender/output"
   os.environ["GENDERSFX_RCLONE_INPUT_REMOTE"] = "ten_remote:detach-voice-gender/input"
   os.environ["GENDERSFX_RCLONE_RESUME_REMOTE"] = "ten_remote:detach-voice-gender/resume"
   ```
4. App sẽ tự: kéo file mới từ Drive input về, đẩy output/resume lên Drive sau
   mỗi episode xử lý xong.

## Chạy local / máy khác có GPU

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx   # đã accept license 2 model pyannote ở trên
python app.py
```

Mở link Gradio hiện trong terminal, dùng như trên Kaggle. Thư mục làm việc
mặc định là `./detach-voice-gender/{input,output,resume}` trong thư mục hiện tại.

## Output

Mỗi episode có 1 thư mục `output/<tên_episode>/` chứa:
- `gender.csv` — `index, start, end, speaker, gender, confidence, text`
- `annotated.srt` — srt gốc, thêm tiền tố `[speakerX|gender]` mỗi dòng thoại

## Cấu trúc mã nguồn

| File | Vai trò |
|------|---------|
| `core.py` | Pipeline lõi: diarization + gender classification + resume cache + map vào SRT |
| `app.py` | Gradio UI: input/output/resume folder, dropdown episode, upload, auto-watch, backup Drive |
| `run_kaggle.py` | Bootstrap Kaggle: clone code mới nhất, cài thư viện, đọc `HF_TOKEN`/rclone secret, chạy `app.py` |
| `run_kaggle.ipynb` | Notebook Kaggle 1-cell, luôn gọi `run_kaggle.py` mới nhất |
| `get_rclone_secret.bat` | Lấy chuỗi base64 rclone.conf để dán vào Kaggle Secret `RCLONE_CONF_B64` |
| `requirements.txt` | Thư viện cần cho `app.py`/`core.py` |
| `sample/` | Mẫu 30 phút đầu để test nhanh (chỉ giữ `.srt`, audio loại khỏi git) |

## Ghi chú

- Model gender/diarization tải qua HuggingFace hub cache — trong 1 session
  Kaggle, gọi nhiều lần không tải lại; giữa các session khác nhau vẫn phải
  tải lại (vài trăm MB, vài phút) vì `/kaggle/working` không tồn tại xuyên
  session.
- Đây là pipeline nghiên cứu, độ chính xác gender phụ thuộc chất lượng
  diarization (giọng nói chồng lấn, nhiễu nền có thể làm giảm độ chính xác).
