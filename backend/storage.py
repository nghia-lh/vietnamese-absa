import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

# File lưu các review đã phân tích
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "dashboard_data" / "reviews.json"

# Tránh nhiều request cùng ghi file một lúc
_WRITE_LOCK = Lock()


def load_reviews():
    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_review(product, result):
    # Tạo bản ghi review mới
    review = {
        "id": str(uuid4()),
        "product": product.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo đọc, cập nhật và ghi không bị chồng chéo
    with _WRITE_LOCK:
        reviews = load_reviews()
        reviews.append(review)

        # Mỗi lần ghi sử dụng một file tạm riêng
        temp_file = DATA_FILE.with_name(
            f"{DATA_FILE.name}.{uuid4().hex}.tmp"
        )

        try:
            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(
                    reviews,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            temp_file.replace(DATA_FILE)

        finally:
            temp_file.unlink(missing_ok=True)

    return review