# Nhận diện loài hoa bằng CNN và Gradio

Ứng dụng web chạy cục bộ, cho phép tải ảnh lên và nhận diện một trong năm
loài hoa: hoa cúc, bồ công anh, hoa hồng, hướng dương và tulip. Ảnh không
được gửi ra Internet.

## Yêu cầu

- Python 3.9–3.11 (môi trường hiện tại đã được kiểm tra với Python 3.9.13)
- File mô hình `flowers_cnn_model.h5` đặt tại thư mục gốc của dự án
- Windows, macOS hoặc Linux; không cần GPU

Thứ tự nhãn đã được xác minh từ script huấn luyện và trực tiếp trên model:
`daisy`, `dandelion`, `rose`, `sunflower`, `tulip`.

## Cài đặt

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

File `.h5` được bỏ qua bởi Git để tránh đưa mô hình lớn vào lịch sử. Sau khi
clone repo, hãy tự chép `flowers_cnn_model.h5` vào cạnh `app.py`.

## Chạy ứng dụng

```powershell
python app.py
```

Trình duyệt sẽ mở `http://127.0.0.1:7860`. Ứng dụng chỉ lắng nghe trên máy
cục bộ và không tạo liên kết chia sẻ công khai. Dừng ứng dụng bằng `Ctrl+C`.

Nếu cổng 7860 đang được sử dụng, đổi `server_port` trong `app.py`.

## Kiểm thử

```powershell
python -m pytest -q
```

Các bài test đối chiếu pipeline Gradio RGB → BGR với pipeline OpenCV gốc trên
10 ảnh mẫu. Sai khác xác suất cho phép nhỏ hơn `1e-4`.

## Tiền xử lý khi dự đoán

1. Chuyển ảnh Gradio từ RGB sang BGR.
2. Resize về `180 × 180`.
3. Chuyển sang `float32` và chia `255.0`.
4. Thêm chiều batch thành `(1, 180, 180, 3)`.
5. Áp dụng softmax cho 5 logits của model.

Mô hình được tải đúng một lần khi ứng dụng khởi động. Vì chỉ biết năm lớp,
mô hình vẫn sẽ gán ảnh ngoài phạm vi vào một lớp; giao diện cảnh báo khi xác
suất cao nhất dưới 50%.

## Huấn luyện lại

Dataset không được lưu trong Git. Để chạy `flowers_cnn.py`, tải Flowers
Recognition dataset và đặt ảnh theo cấu trúc:

```text
datasets/
└── flowers/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
```

Sau đó chạy:

```powershell
python flowers_cnn.py
```

Độ chính xác validation tham khảo khoảng 74%; kết quả thực tế có thể sai do
mô hình còn overfit.
