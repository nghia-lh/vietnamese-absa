
"""Chạy ablation cho baseline Overall Sentiment."""

import json
import platform
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_jsonl
from .train_baseline import metrics


DATA = Path("data/overall_visfd")
OUTPUT = Path("outputs/ablation_baseline/reproduced_metrics.json")
REFERENCE = Path("outputs/ablation_baseline/metrics.json")

CONFIGS = {
    "original": {
        "ngram_range": (1, 2),
        "class_weight": "balanced",
    },
    "no_bigram": {
        "ngram_range": (1, 1),
        "class_weight": "balanced",
    },
    "no_class_weight": {
        "ngram_range": (1, 2),
        "class_weight": None,
    },
}


def main():
    # Đọc ba tập dữ liệu
    data = {
        split: load_jsonl(DATA / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }

    x_train = [r["text"] for r in data["train"]]
    y_train = [r["overall"] for r in data["train"]]

    result = {
        "dataset": "UIT-ViSFD",
        "seed": 42,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "split_sizes": {
            k: len(v) for k, v in data.items()
        },
        "experiments": {},
    }

    for name, config in CONFIGS.items():
        print(f"\nĐang chạy: {name}")

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=config["ngram_range"],
                min_df=1,
                max_features=60000,
            )),
            ("clf", LogisticRegression(
                C=0.1,
                max_iter=1200,
                class_weight=config["class_weight"],
                random_state=42,
            )),
        ])

        model.fit(x_train, y_train)

        scores = {}
        for split in ("dev", "test"):
            x = [r["text"] for r in data[split]]
            y = [r["overall"] for r in data[split]]
            pred = model.predict(x)

            m = metrics(y, pred)
            scores[split] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
            }

        result["experiments"][name] = {
            "ngram_range": list(config["ngram_range"]),
            "class_weight": config["class_weight"],
            "C": 0.1,
            **scores,
        }

        print("Dev Macro-F1:", scores["dev"]["macro_f1"])
        print("Test Macro-F1:", scores["test"]["macro_f1"])

    # Lưu kết quả mới
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nĐã lưu:", OUTPUT)

    # So sánh với kết quả cũ
    if REFERENCE.exists():
        old = json.loads(REFERENCE.read_text(encoding="utf-8"))

        print("\n=== SO SÁNH KẾT QUẢ CŨ ===")
        for name in CONFIGS:
            expected = old[name]["test_macro_f1"]
            actual = result["experiments"][name]["test"]["macro_f1"]

            matched = abs(expected - actual) < 1e-8

            print(
                f"{name}: "
                f"cũ={expected:.10f}, "
                f"mới={actual:.10f}, "
                f"khớp={matched}"
            )


if __name__ == "__main__":
    main()
