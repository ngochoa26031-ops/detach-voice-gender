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
5. Trong app: upload file video/audio + file `.srt` tương ứng ngay trên web,
   bấm **▶ Xác định giới tính**, tải `gender.csv` và `annotated.srt` ngay
   trên web khi xong.

> ⚠️ `/kaggle/working` mất khi session kết thúc — nhớ tải kết quả về máy
> trước khi dừng session.

## Chạy local / máy khác có GPU

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx   # đã accept license 2 model pyannote ở trên
python app.py
```

Mở link Gradio hiện trong terminal, upload file, chạy như trên Kaggle.

## Output

- `gender.csv` — `index, start, end, speaker, gender, confidence, text`
- `annotated.srt` — srt gốc, thêm tiền tố `[speakerX|gender]` mỗi dòng thoại

## Cấu trúc

| File | Vai trò |
|------|---------|
| `core.py` | Pipeline lõi: diarization + gender classification + map vào SRT |
| `app.py` | Gradio UI: upload media+srt, chạy, tải kết quả |
| `run_kaggle.py` | Bootstrap Kaggle: clone code mới nhất, cài thư viện, đọc `HF_TOKEN` từ Secret, chạy `app.py` |
| `run_kaggle.ipynb` | Notebook Kaggle 1-cell, luôn gọi `run_kaggle.py` mới nhất |
| `requirements.txt` | Thư viện cần cho `app.py`/`core.py` |
| `sample/` | Mẫu 30 phút đầu để test nhanh (chỉ giữ `.srt`, audio loại khỏi git) |

## Ghi chú

- Model gender/diarization tải qua HuggingFace hub cache — trong 1 session
  Kaggle, gọi nhiều lần không tải lại; giữa các session khác nhau vẫn phải
  tải lại (vài trăm MB, vài phút) vì `/kaggle/working` không tồn tại xuyên
  session.
- Đây là pipeline nghiên cứu, độ chính xác gender phụ thuộc chất lượng
  diarization (giọng nói chồng lấn, nhiễu nền có thể làm giảm độ chính xác).
