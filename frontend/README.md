# ABSA Streamlit Frontend — review handoff

## Cách chạy

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Khởi động FastAPI trước tại `http://127.0.0.1:8000` hoặc đổi địa chỉ ở thanh bên.

## API frontend giả định (cần đối chiếu API_CONTRACT.md / mã FastAPI)

- `GET /api/v1/health` trả JSON, HTTP 2xx.
- `POST /api/v1/analyze` nhận `{ "text": "..." }`.
- `POST /api/v1/reviews` nhận `{ "product": "...", "text": "..." }`, trả kết quả phân tích và đã lưu.
- `GET /api/v1/reviews` trả `{ "reviews": [...] }`, có thể lọc `product`, `from_date`, `to_date`.
- `GET /api/v1/statistics` trả `total_reviews`, `overall_distribution`, `aspect_statistics`.
- Result phân tích: `text`, `overall_sentiment`, `overall_confidence`, `aspects` (mỗi item có `start`, `end`, `category`, `polarity`, `confidence`, `text`), `truncated`.

Không có Backend trong ZIP gốc nên **chưa xác nhận schema và luồng chạy thực tế**.

## Sửa đổi từ ZIP gốc

- Xóa mock `du_lieu_gia.py` vốn không có trong ZIP và chế độ dữ liệu giả, tránh lỗi import và demo nhầm.
- Chấp nhận mọi HTTP 2xx (ví dụ 201 khi tạo review), vẫn hiển thị lỗi cho 4xx/5xx.
- Nhãn ngoài Positive/Neutral/Negative không bị gắn nhầm thành Neutral.
- Chọn rỗng khoảng ngày không gây IndexError.
- Bổ sung requirements và hướng dẫn chạy.

## Checklist demo

1. `GET /api/v1/health` kết nối thành công.
2. Một review mới cho kết quả Overall và các span có offset đúng trên text gốc.
3. `POST /api/v1/reviews` thực sự lưu; Dashboard cập nhật thống kê.
4. Lọc sản phẩm và ngày áp dụng nhất quán cho reviews và statistics.
5. Thử đầu vào rỗng, review dài, sai kết nối và review không có span.

## Sửa lỗi kết quả review cũ (24/10/2026)

- Snapshot `text`, `product`, `multi` trong callback bấm nút **trước khi widget bị khóa**.
- Khi nội dung đang nhập khác với lần gửi gần nhất, ẩn kết quả cũ và yêu cầu phân tích lại.
- Nếu `text` trong response của FastAPI khác review vừa gửi, chặn hiển thị kết quả sai review.
- Khi click không hợp lệ (rỗng hoặc thiếu tên sản phẩm nếu Lưu), xóa kết quả cũ.
- Không đổi CSS, bảng màu, cấu trúc API hay logic dashboard.

**Nâng cấp:** Thay `frontend/app.py` trong project bằng file mới này, giữ nguyên các file còn lại. Sau khi lưu, Streamlit sẽ tự reload; nếu không, dừng bằng `Ctrl+C`, chạy lại `python -m streamlit run frontend/app.py`.
- Input widgets không chuyển trạng thái enabled/disabled khi submit, tránh reset không mong muốn; chỉ khóa các nút trong lần thực thi yêu cầu.
