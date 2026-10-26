# Phân tích cảm xúc và khía cạnh trong đánh giá sản phẩm tiếng Việt

Đây là đồ án môn Xử lý ngôn ngữ tự nhiên (NLP), thực hiện bài toán phân tích cảm xúc trong các bài đánh giá sản phẩm bằng tiếng Việt.

Nhóm xây dựng hai mô hình để giải quyết hai bài toán:

- **Overall Sentiment Classification:** Dự đoán cảm xúc chung của một bài đánh giá, gồm 3 nhãn `positive`, `neutral` và `negative`.
- **Aspect-Based Sentiment Analysis (ABSA):** Tìm các đoạn văn bản nói về từng khía cạnh của sản phẩm, xác định loại khía cạnh và cảm xúc tương ứng.

Ngoài phần huấn luyện và đánh giá mô hình, nhóm còn xây dựng API bằng FastAPI để tích hợp với giao diện web. Các mô hình chạy trực tiếp trên máy chủ, không sử dụng API của các mô hình ngôn ngữ bên ngoài để phân tích review.

## 1. Dữ liệu sử dụng

Nhóm sử dụng hai bộ dữ liệu tiếng Việt. Dataset gốc được cung cấp bởi nhóm tác giả tương ứng:

- **[UIT-ViSD4SA](https://github.com/kimkim00/UIT-ViSD4SA):** cung cấp nội dung review, vị trí các opinion spans, aspect category, polarity và các split dữ liệu cho bài toán Aspect.
- **[UIT-ViSFD](https://github.com/LuongPhan/UIT-ViSFD):** cung cấp thông tin số sao (`n_star`) để nhóm đối chiếu và tạo nhãn Overall. Tệp dữ liệu gốc có thể tải từ `UIT-ViSFD.zip` trong repository này.

**Lưu ý:** Nhóm không sử dụng toàn bộ UIT-ViSFD như một tập dữ liệu Overall độc lập. Nhóm dùng các review từ UIT-ViSD4SA, đối chiếu nội dung đã chuẩn hóa với rating của UIT-ViSFD, loại bản ghi không khớp hoặc mâu thuẫn và giữ cách chia Train/Dev/Test gốc.

| Bài toán | Bộ dữ liệu | Train | Dev | Test |
|---|---|---:|---:|---:|
| Overall Sentiment | UIT-ViSD4SA (review) + UIT-ViSFD (rating) | 7.057 | 1.004 | 2.022 |
| Aspect-Based Sentiment | UIT-ViSD4SA | 7.360 | 1.037 | 2.076 |

### 1.1. Overall Sentiment

Đối với Overall Sentiment, nhóm sử dụng thông tin số sao đánh giá để tạo nhãn cảm xúc:

- 1–2 sao: `negative`
- 3 sao: `neutral`
- 4–5 sao: `positive`

Do nhãn được suy ra từ số sao nên không phải lúc nào cũng phản ánh hoàn toàn nội dung văn bản. Ví dụ, một người có thể đánh giá 5 sao nhưng vẫn phàn nàn về một vài tính năng của sản phẩm.

Dữ liệu sau xử lý được lưu tại:

```text
data/overall_visfd/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── preprocessing_report.json
```

### 1.2. Aspect-Based Sentiment

Với bài toán ABSA, nhóm sử dụng UIT-ViSD4SA. Mỗi review có thể chứa nhiều khía cạnh, mỗi khía cạnh gồm:

- `start`, `end`: vị trí đoạn văn bản.
- `category`: loại khía cạnh như BATTERY, CAMERA, DESIGN, PERFORMANCE...
- `polarity`: cảm xúc của khía cạnh (`positive`, `neutral`, `negative`).

Ví dụ:

```text
Review: "Pin rất tốt nhưng camera hơi kém."

Aspect 1:
  Text: "Pin rất tốt"
  Category: BATTERY
  Polarity: positive

Aspect 2:
  Text: "camera hơi kém"
  Category: CAMERA
  Polarity: negative
```

Nhóm sử dụng cách gán nhãn BIO kết hợp category và polarity để huấn luyện mô hình nhận diện các đoạn khía cạnh.

Dữ liệu sau xử lý nằm trong:

```text
data/processed_v2/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
├── tags.json
└── preprocessing_report.json
```

Lưu ý: Overall và Aspect được đánh giá trên hai tập dữ liệu khác nhau.

## 2. Các mô hình sử dụng

### 2.1. Baseline

Nhóm xây dựng baseline cho Overall Sentiment bằng TF-IDF kết hợp với ba thuật toán:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

Các cấu hình được so sánh trên tập Dev bằng Macro-F1. Cấu hình tốt nhất trong thực nghiệm là Logistic Regression với `C=0.1`, TF-IDF unigram và bigram, có sử dụng `class_weight=balanced`.

### 2.2. PhoBERT Overall

Nhóm fine-tune `vinai/phobert-base-v2` cho bài toán phân loại cảm xúc chung.

Đầu vào là nội dung review và đầu ra là một trong ba nhãn cảm xúc.

### 2.3. PhoBERT Aspect

Nhóm sử dụng một mô hình PhoBERT khác cho bài toán BIO sequence labeling.

Mô hình dự đoán nhãn cho từng token, sau đó chuyển các nhãn BIO thành danh sách aspect spans gồm vị trí, category và polarity.

Hai mô hình PhoBERT được huấn luyện riêng và kết hợp khi chạy inference.

## 3. Cấu trúc thư mục

Các thư mục chính của project:

```text
data/
    overall_visfd/          # Dữ liệu Overall
    processed_v2/           # Dữ liệu Aspect
    visd4sa/                # Dữ liệu review gốc
    visfd_raw/              # Dữ liệu rating gốc

src/
    download_data.py        # Tải dữ liệu
    audit.py                # Kiểm tra dữ liệu
    prepare.py              # Tiền xử lý Aspect
    prepare_overall_visfd.py
    segmentation.py         # Tách từ tiếng Việt
    encoding.py
    decode.py
    train_baseline.py       # Train baseline
    train_phobert.py        # Train PhoBERT
    evaluate_overall.py
    evaluate_model.py
    evaluate_aspects.py
    analyze_errors.py       # Phân tích lỗi
    inference.py            # Chạy hai mô hình

backend/
    main.py                 # FastAPI
    storage.py              # Lưu review

models/
    phobert/
        overall/best/
        aspect/best/
    vncorenlp/

outputs/
    baseline_visfd/
    evaluation/
    analysis/
    benchmark/
```

Một số thư mục dữ liệu và checkpoint có dung lượng lớn nên không nhất thiết nằm trong gói source code.

## 4. Cài đặt môi trường

Project được chạy trên Python 3.12. Phần xử lý dữ liệu và backend đã được thực hiện trên macOS.

Tạo môi trường ảo:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài thư viện:

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Các thư viện chính gồm PyTorch, Transformers, Accelerate, scikit-learn, pandas, NumPy, VnCoreNLP, seqeval, matplotlib, FastAPI và Uvicorn.

Nhóm đã kiểm tra cài đặt trên môi trường hiện tại bằng:

```bash
python3 -m pip check
```

Kết quả:

```text
No broken requirements found.
```

Kiểm tra import các thư viện chính cũng thành công.

Nếu chưa có bộ tách từ VnCoreNLP, tải bằng lệnh:

```bash
python3 -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='models/vncorenlp')"
```

VnCoreNLP cần có Java để chạy. Ngoài ra, phiên bản thư viện giữa máy local và môi trường huấn luyện trên GPU có thể khác nhau, nên cần kiểm tra lại nếu cài trên máy mới.

## 5. Chuẩn bị dữ liệu

Nếu đã có các thư mục `data/overall_visfd` và `data/processed_v2` thì có thể dùng trực tiếp, không cần tiền xử lý lại.

### 5.1. Tải dữ liệu gốc

Dữ liệu gốc và checkpoint fine-tuned **không được đính kèm trong source code repository**. Để chuẩn bị dữ liệu từ đầu:

1. Tải ba file `train.jsonl`, `dev.jsonl`, `test.jsonl` trong thư mục [`data/` của UIT-ViSD4SA](https://github.com/kimkim00/UIT-ViSD4SA/tree/main/data), đặt dưới thư mục `data/visd4sa/` của project.
2. Tải [`UIT-ViSFD.zip`](https://github.com/LuongPhan/UIT-ViSFD/blob/main/UIT-ViSFD.zip), giải nén dữ liệu rating vào `data/visfd_raw/`. Kiểm tra rằng các tệp CSV/split có tên và định dạng mà `src/prepare_overall_visfd.py` yêu cầu, đặc biệt các cột review (`comment`) và số sao (`n_star`).
3. Cài đặt VnCoreNLP theo mục 4 và xác nhận các file model đã nằm trong `models/vncorenlp/`.

Cấu trúc thư mục dữ liệu đầu vào dự kiến:

```text
data/
├── visd4sa/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── visfd_raw/
    └── ...                 # Các CSV sau khi giải nén UIT-ViSFD.zip
```

Có thể tham khảo `src/download_data.py` nếu muốn sử dụng script tải dữ liệu đi kèm project; cần kiểm tra tham số của script trước khi chạy.

### 5.2. Chạy preprocessing

Nếu muốn tạo lại dữ liệu từ nguồn gốc, chạy:

```bash
python3 -m src.audit --data data/visd4sa

python3 -m src.prepare \
  --data data/visd4sa \
  --vncorenlp models/vncorenlp \
  --output data/processed_v2

python3 -m src.prepare_overall_visfd
```

Script `prepare_overall_visfd.py` sử dụng dữ liệu rating từ `data/visfd_raw` và review tương ứng từ `data/visd4sa`.

Trong preprocessing Overall, nhãn được tạo theo quy tắc 1–2 sao → `negative`, 3 sao → `neutral`, 4–5 sao → `positive`. Các review không thể đối chiếu chắc chắn hoặc có rating không nhất quán bị loại; những nhãn này là **weak labels**. Với Aspect, cần giữ đúng ranh giới ký tự của gold spans và không ép căn chỉnh các trường hợp không hợp lệ.

Sau khi chạy, hãy xem `preprocessing_report.json` ở mỗi thư mục đầu ra và kiểm tra số lượng mẫu. Nếu không khớp số liệu trong paper, cần kiểm tra phiên bản dataset, tham số của script và số bản ghi bị loại, không tự điều chỉnh số lượng mẫu.

Sau khi tiền xử lý, nhóm đã kiểm tra số lượng mẫu, dữ liệu trùng trong từng split và giữa các split.

Kết quả kiểm tra trên dữ liệu đã xử lý:

| Nội dung | Overall | Aspect |
|---|---:|---:|
| Train | 7.057 | 7.360 |
| Dev | 1.004 | 1.037 |
| Test | 2.022 | 2.076 |
| Review trùng trong từng split | 0 | 0 |
| Review trùng giữa các split | 0 | 0 |

Các file dự đoán trên Test cũng đã được kiểm tra khớp thứ tự với dữ liệu tương ứng. Với Aspect, không phát hiện trường hợp độ dài chuỗi BIO không hợp lệ trong lần kiểm tra này.

## 6. Huấn luyện mô hình

Nếu đã có checkpoint trong thư mục `models/` thì không cần train lại, có thể chuyển thẳng sang phần đánh giá hoặc chạy API.

### 6.1. Train Baseline

```bash
python3 -m src.train_baseline \
  --overall-source visfd \
  --data data/overall_visfd \
  --output outputs/baseline_visfd \
  --seed 42
```

Nhóm chọn cấu hình dựa trên Macro-F1 của tập Dev, sau đó mới đánh giá trên Test.

### 6.2. Train PhoBERT Overall

```bash
python3 -m src.train_phobert \
  --task overall \
  --overall-source visfd \
  --data data/overall_visfd \
  --vncorenlp models/vncorenlp \
  --output models/phobert \
  --model vinai/phobert-base-v2 \
  --epochs 4 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 2e-5 \
  --max-length 256 \
  --seed 42 \
  --fp16
```

Checkpoint được lưu trong:

```text
models/phobert/overall/best/
```

### 6.3. Train PhoBERT Aspect

```bash
python3 -m src.train_phobert \
  --task aspect \
  --processed data/processed_v2 \
  --output models/phobert \
  --model vinai/phobert-base-v2 \
  --epochs 4 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 2e-5 \
  --max-length 256 \
  --seed 42 \
  --fp16
```

Checkpoint được lưu trong:

```text
models/phobert/aspect/best/
```

Lưu ý: checkpoint Aspect hiện tại là kết quả của lần huấn luyện lại. Nhóm sử dụng metrics tương ứng với checkpoint này khi báo cáo, không trộn với kết quả từ lần train cũ.

Tham số `--fp16` được sử dụng khi huấn luyện trên GPU CUDA. Khi chạy trên CPU hoặc máy không hỗ trợ CUDA cần điều chỉnh tham số này.

### 6.4. Trạng thái checkpoint và cách sử dụng

### 6.4. Trạng thái checkpoint và cách sử dụng

Hai mô hình sử dụng backbone gốc [`vinai/phobert-base-v2`](https://huggingface.co/vinai/phobert-base-v2), nhưng **checkpoint đã fine-tune cho Overall và Aspect là hai bộ trọng số riêng**, không thể thay bằng checkpoint PhoBERT gốc nếu muốn tái tạo kết quả trong paper.

Vị trí checkpoint để đánh giá và chạy inference:

```text
models/phobert/
├── overall/best/     # Checkpoint PhoBERT Overall đã fine-tune
└── aspect/best/      # Checkpoint PhoBERT Aspect đã fine-tune
```

**Trạng thái phát hành:** Repository cung cấp mã nguồn huấn luyện và đánh giá. Hai checkpoint PhoBERT đã fine-tune được chia sẻ qua [Google Drive](https://drive.google.com/file/d/1JWf3KrzuXr6NE5aZMxxEF6hapkTre_9r/view?usp=drive_link) do dung lượng lớn, không được lưu trực tiếp trên GitHub.

Hướng dẫn tải và giải nén checkpoint được trình bày tại **Mục 12 – Ghi chú về khả năng tái lập**.

Để chạy đánh giá hoặc inference, cần bảo đảm mỗi thư mục `best/` chứa đầy đủ trọng số mô hình, cấu hình và các file tokenizer tương ứng.

## 7. Đánh giá mô hình

### 7.1. Overall Sentiment

Chạy đánh giá:

```bash
python3 -m src.evaluate_overall
```

Kết quả được lưu trong:

```text
outputs/evaluation/overall/
    metrics.json
    predictions.jsonl
    confusion_matrix.png
```

Kết quả trên Test của UIT-ViSFD:

| Mô hình | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7408506429277942 | 0.6454101401965168 |
| PhoBERT Overall | 0.7932739861523245 | 0.6569389079619072 |

PhoBERT có kết quả tốt hơn baseline ở cả Accuracy và Macro-F1 trên tập Test này.

### 7.2. Aspect-Based Sentiment

Chạy đánh giá:

```bash
python3 -m src.evaluate_model \
  --processed data/processed_v2 \
  --checkpoint models/phobert/aspect/best \
  --split test \
  --output outputs/evaluation/aspect_kaggle/test_predictions.jsonl
```

Nhóm sử dụng hai cách đánh giá chính:

- **Exact Span:** Span phải trùng chính xác vị trí bắt đầu và kết thúc.
- **Aspect–Sentiment Pair:** Phải trùng vị trí, category và polarity.

Kết quả trên Test của UIT-ViSD4SA:

| Chỉ số | Precision | Recall | F1 |
|---|---:|---:|---:|
| Exact Span | 0.4866344195519348 | 0.5888786198398028 | 0.5328965709506552 |
| Aspect–Sentiment Pair | 0.43228105906313646 | 0.5231053604436229 | 0.47337608028993583 |

Ngoài ra, kết quả token classification của lần huấn luyện lại là:

- Token Accuracy: `0.7108680702496826`
- Token Macro-F1: `0.2237983556578706`

Do bài toán cần trích xuất đúng cả đoạn văn bản và nhãn, nhóm sử dụng Exact Span F1 và Pair F1 làm các chỉ số chính cho phần đánh giá Aspect.

### 7.3. Ablation của Baseline

Nhóm thử bỏ một số thành phần của cấu hình Logistic Regression tốt nhất để xem ảnh hưởng đến kết quả:

| Cấu hình | Test Macro-F1 |
|---|---:|
| Cấu hình ban đầu | 0.6454101401965168 |
| Không dùng bigram | 0.6429267125935124 |
| Không dùng class weighting | 0.45170397648350935 |

Trong thực nghiệm này, việc bỏ class weighting làm Macro-F1 giảm nhiều hơn so với bỏ bigram.

Kết quả được lưu tại:

```text
outputs/ablation_baseline/metrics.json
```
### 7.4. Kết quả thực nghiệm đã lưu

Thư mục `results/` chứa các file metrics từ những thí nghiệm được báo cáo trong paper. Các file này giúp kiểm tra kết quả mà không cần huấn luyện lại mô hình.

| Nội dung trong paper | File kết quả |
|---|---|
| Table 2 – TF-IDF Baseline (Dev) | `results/baseline_visfd_metrics.json` |
| Table 3 – Overall Sentiment (Test) | `results/baseline_visfd_metrics.json`, `results/phobert_overall_test_metrics.json` |
| Table 4 – Per-class Overall F1 | `results/baseline_visfd_metrics.json`, `results/phobert_overall_test_metrics.json` |
| Table 5 – Aspect Extraction | `results/aspect_nb_baseline_metrics.json`, `results/phobert_aspect_test_metrics.json` |
| Table 7 – Baseline Ablation | `results/baseline_ablation_metrics.json` |
| External Tiki Evaluation | `results/external_tiki_metrics.json` |

Thông tin cấu hình huấn luyện và kết quả token-level được lưu trong:

- `results/phobert_overall_training_metrics.json`
- `results/phobert_aspect_training_metrics.json`

Riêng Table 6 (inference latency) được mô tả trong mục 10.3. Chưa có file benchmark tương ứng trong `results/`.

Kết quả PhoBERT Aspect được tính từ `data/processed_v2/test.jsonl` và `outputs/evaluation/aspect_kaggle/test_predictions.jsonl`, sử dụng exact character-span matching và full span–category–polarity matching. Các file dữ liệu và dự đoán gốc không được đóng gói trong `results/`.

## 8. Phân tích lỗi

Nhóm phân tích các review dự đoán sai để tìm hiểu những trường hợp mô hình còn gặp khó khăn.

### 8.1. Overall Sentiment

Từ 2.022 review trong tập Test, mô hình dự đoán sai 418 review.

Nhóm phân tích thêm các trường hợp có phủ định, teencode, emoji và câu ngắn.

| Nhóm review | Số mẫu | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Toàn bộ Test | 2.022 | 79,33% | 65,69% |
| Có phủ định | 1.128 | 74,91% | 64,20% |
| Có emoji | 77 | 84,42% | 58,51% |
| Có teencode | 676 | 77,22% | 66,32% |
| Câu rất ngắn | 4 | 100,00% | 33,33% |

Các nhóm này được lọc bằng biểu thức chính quy và có thể trùng nhau. Ví dụ, một review vừa chứa phủ định vừa chứa teencode.

Ở nhóm câu rất ngắn, chỉ có 4 mẫu nên kết quả chưa đủ để rút ra nhận xét chung.

Một trường hợp Overall dự đoán sai là review có cả lời khen và chê về pin, màn hình, camera và giá bán. Nhãn từ rating là `positive` nhưng mô hình dự đoán `neutral`.

Nhóm nhận thấy những review có nhiều ý kiến trái chiều có thể gây khó khăn cho việc xác định một nhãn cảm xúc chung. Bên cạnh đó, vì nhãn Overall được tạo từ rating nên cũng có trường hợp cảm xúc trong văn bản không hoàn toàn giống nhãn.

### 8.2. Aspect-Based Sentiment

Kết quả đánh giá theo một số nhóm review:

| Nhóm review | Số mẫu | Exact Span F1 | Pair F1 |
|---|---:|---:|---:|
| Toàn bộ Test | 2.076 | 53,29% | 47,34% |
| Có phủ định | 1.158 | 49,02% | 42,92% |
| Có emoji | 65 | 48,05% | 43,75% |
| Có teencode | 689 | 49,73% | 43,68% |
| Có từ 2 khía cạnh trở lên | 1.690 | 55,30% | 49,13% |
| Có nhiều polarity | 826 | 53,61% | 45,67% |

Ngoài ra, nhóm sử dụng cách ghép các span có phần văn bản trùng nhau để phân loại lỗi gần đúng.

| Loại | Số lần ghi nhận |
|---|---:|
| Dự đoán đúng hoàn toàn | 3.396 |
| Sai ranh giới span | 2.216 |
| Sai category | 655 |
| Sai polarity | 454 |
| Gold span không được ghép | 453 |
| Predicted span không được ghép | 1.817 |

Đây là kết quả phân nhóm tham khảo bằng phương pháp ghép span theo overlap. Một span có thể có nhiều lỗi cùng lúc nên các số lượng trên không phải những nhóm độc lập.

Qua việc xem các ví dụ thực tế, nhóm nhận thấy có trường hợp mô hình xác định đúng category và polarity nhưng lấy thiếu hoặc thừa một vài ký tự ở ranh giới span. Những trường hợp này vẫn bị tính sai khi sử dụng exact-match.

Nhóm cũng ghi nhận những khó khăn liên quan đến cách viết không chuẩn, từ viết tắt, emoji và các review có nhiều cảm xúc khác nhau.

Kết quả phân tích được lưu tại:

```text
outputs/analysis/
    error_analysis.json
    aspect_error_groups.json
    language_error_examples.json
```

Các trường hợp có khả năng mang ý nghĩa mỉa mai cần được xem xét thủ công, không thể kết luận chỉ dựa vào emoji hoặc từ khóa.

## 9. Đánh giá trên dữ liệu bên ngoài

Nhóm thử đánh giá PhoBERT Overall trên các review từ Tiki để xem kết quả khi sử dụng dữ liệu ngoài tập Test ban đầu.

Dữ liệu ban đầu có 35.632 dòng. Sau các bước làm sạch, lọc dữ liệu trùng và xử lý những trường hợp không phù hợp với pipeline, còn 15.542 review đủ điều kiện theo quy trình đã sử dụng.

Trong đó có 718 review bị loại ở bước lọc ký tự Unicode ngoài BMP do vấn đề gặp phải khi xử lý bằng bộ tách từ. Đây là cách xử lý tạm thời của thực nghiệm, không phải cách giải quyết chung cho mọi ký tự Unicode.

Nhóm lấy 100 review với seed 42, gồm:

- 30 review negative
- 30 review neutral
- 40 review positive

Kết quả:

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0.69 |
| Macro-F1 | 0.6373737373737374 |

Confusion matrix theo thứ tự `negative`, `neutral`, `positive`:

```text
[[21, 5, 4],
 [14, 9, 7],
 [ 1, 0, 39]]
```

Nhãn của dữ liệu Tiki cũng được suy ra từ rating. Vì vậy kết quả này chỉ dùng để đánh giá thêm mô hình Overall trên nguồn review khác, chưa thể coi là đánh giá bằng nhãn cảm xúc được gán thủ công.

File kết quả:

```text
outputs/evaluation/external_tiki/
    metrics.json
    predictions.jsonl
    failed_reviews.jsonl
```

Đoạn script đánh giá Tiki trước đó được chạy riêng trong Terminal. Để người khác chạy lại hoàn toàn, nhóm vẫn cần bổ sung script này vào source code.

## 10. API và ứng dụng

Nhóm sử dụng FastAPI để kết nối mô hình với ứng dụng web.

### 10.1. Chạy backend

Trước tiên cần có checkpoint Overall, Aspect và bộ tách từ VnCoreNLP trong đúng thư mục.

Chạy server:

```bash
python3 -m uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Sau khi chạy, mở Swagger UI tại:

```text
http://127.0.0.1:8000/docs
```

### 10.2. Các API

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/api/v1/health` | Kiểm tra API |
| POST | `/api/v1/analyze` | Phân tích một review |
| POST | `/api/v1/reviews` | Phân tích và lưu review |
| GET | `/api/v1/reviews` | Lấy danh sách review đã lưu |
| GET | `/api/v1/statistics` | Thống kê review |

Ví dụ gọi API phân tích:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Pin rất tốt nhưng camera hơi kém."}'
```

API trả về cảm xúc chung và danh sách khía cạnh, bao gồm:

- `overall_sentiment`
- `overall_confidence`
- `aspects`
- `truncated`

Trong `aspects` có thông tin về đoạn văn bản, vị trí `start`, `end`, category, polarity và confidence.

Các vị trí ký tự sử dụng quy ước `[start, end)` của Python. Khi hiển thị phần highlight trên React cần chú ý cách xử lý Unicode giữa Python và JavaScript.

### 10.3. Benchmark inference

Nhóm đã đo tốc độ suy luận trên MacBook Apple Silicon, sử dụng CPU.

Thực nghiệm dùng 50 review từ UIT-ViSD4SA Test, seed 42, có 5 lượt warm-up.

| Chỉ số | Inference trực tiếp | FastAPI |
|---|---:|---:|
| Mean | 95.893 ms | 95.330 ms |
| Median | 90.957 ms | 90.510 ms |
| P95 | 123.708 ms | 124.712 ms |
| Throughput tuần tự | 10.428 review/s | 10.490 request/s |

Thời gian load mô hình được ghi nhận là `8.951572624966502` giây trong phép đo inference.

Các phép đo được thực hiện riêng nên không thể kết luận HTTP nhanh hơn chạy trực tiếp chỉ dựa vào sự chênh lệch nhỏ này.

File kết quả:

```text
outputs/benchmark/
    inference_latency.json
    api_latency.json
```

Benchmark chưa bao gồm thời gian render frontend, độ trễ Internet hoặc tải từ nhiều người dùng đồng thời.

### 10.4. Frontend

Backend đã có API phục vụ các chức năng phân tích review, lưu dữ liệu và thống kê. Phần tích hợp frontend và kiểm thử giao diện sẽ được cập nhật sau khi hoàn thành.

Dữ liệu review của ứng dụng đang được lưu dưới dạng JSON trong `dashboard_data`, chưa sử dụng cơ sở dữ liệu riêng.

## 11. Hạn chế và hướng phát triển

Qua quá trình thực hiện, nhóm nhận thấy một số hạn chế:

- **Nhãn Overall:** Được suy ra từ số sao nên có thể khác cảm xúc thể hiện trong nội dung review.
- **Ranh giới Aspect:** Mô hình đôi khi lấy thiếu hoặc thừa ký tự, làm giảm kết quả Exact Span F1.
- **Tiếng Việt không chuẩn:** Review có thể chứa teencode, lỗi chính tả, từ viết tắt và emoji, gây khó khăn cho tiền xử lý và dự đoán.
- **Review nhiều cảm xúc:** Một bài đánh giá có thể khen và chê nhiều khía cạnh cùng lúc, trong khi Overall chỉ dự đoán một nhãn.
- **Giới hạn dữ liệu:** Mô hình Aspect được học trên taxonomy của UIT-ViSD4SA nên chưa đánh giá đầy đủ khả năng áp dụng cho các loại sản phẩm ngoài phạm vi dữ liệu.
- **Review dài:** Việc giới hạn đầu vào ở `max_length=256` có thể khiến một phần nội dung bị cắt.
- **Bộ tách từ:** Một số ký tự đặc biệt từng gây lỗi trong quá trình xử lý dữ liệu Tiki.

Nếu có thêm thời gian, nhóm có thể thử cải thiện việc chuẩn hóa từ viết tắt, bổ sung dữ liệu gán nhãn thủ công cho Overall, cải thiện xử lý ranh giới span và đánh giá trên nhiều nguồn review khác.

Các hướng trên mới là đề xuất, nhóm chưa thực hiện thí nghiệm để kiểm chứng hiệu quả.

## 12. Ghi chú về khả năng tái lập

Repository cung cấp mã nguồn cho quy trình tiền xử lý dữ liệu, huấn luyện, đánh giá và suy luận của hai tác vụ: phân loại cảm xúc tổng thể (Overall Sentiment Classification) và trích xuất ý kiến theo khía cạnh (Aspect-Level Opinion Extraction).

Các thí nghiệm sử dụng hai bộ dữ liệu UIT-ViSD4SA và UIT-ViSFD. Dữ liệu gốc không được đóng gói trực tiếp trong repository; hướng dẫn chuẩn bị dữ liệu được trình bày ở Mục 5.

Các file kết quả thực nghiệm được cung cấp trong thư mục `results/` và được đối chiếu với các bảng kết quả trong paper tại Mục 7.4.

### Tải checkpoint PhoBERT

Hai checkpoint PhoBERT đã fine-tune được cung cấp qua Google Drive do dung lượng lớn, không lưu trực tiếp trong GitHub.

**Link tải:** [PhoBERT Checkpoints – Google Drive](https://drive.google.com/file/d/1JWf3KrzuXr6NE5aZMxxEF6hapkTre_9r/view?usp=drive_link)

File `phobert_checkpoints.zip` chứa hai mô hình:

- **PhoBERT Overall:** Phân loại cảm xúc tổng thể (Negative, Neutral, Positive).
- **PhoBERT Aspect:** Trích xuất opinion spans, aspect categories và sentiment polarities.

Sau khi tải, đặt file ZIP vào thư mục gốc của project và giải nén:

```bash
unzip phobert_checkpoints.zip
```

Sau khi giải nén, hai checkpoint nằm tại:

```text
models/phobert/
├── overall/
│   └── best/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
└── aspect/
    └── best/
        ├── config.json
        ├── model.safetensors
        └── ...
```

Việc tái lập toàn bộ quá trình huấn luyện yêu cầu dữ liệu đã tiền xử lý, các thư viện phụ thuộc và môi trường tính toán phù hợp. Kết quả thực nghiệm có thể chịu ảnh hưởng bởi phiên bản thư viện và phần cứng sử dụng.

## 13. Ghi chú

Các kết quả trong README được lấy từ những lần thực nghiệm đã thực hiện. Nhóm sử dụng kết quả của checkpoint hiện tại khi báo cáo, đặc biệt là mô hình PhoBERT Aspect đã huấn luyện lại.

Project đã chạy trên môi trường phát triển hiện tại, nhưng chưa kiểm tra lại toàn bộ quy trình từ đầu trên một máy sạch. Nếu muốn tái lập hoàn toàn, cần chuẩn bị đúng dữ liệu nguồn, checkpoint, phiên bản thư viện và môi trường huấn luyện tương ứng.

**Code availability và reproducibility:** Repository công khai source code để kiểm tra cách chuẩn bị dữ liệu, huấn luyện, đánh giá và suy luận. Tuy nhiên, việc clone repository **chưa đủ để tái tạo tức thì các kết quả trong paper**, vì dataset gốc, checkpoint fine-tuned và một số tài nguyên/thí nghiệm phụ cần được chuẩn bị riêng. Khi sử dụng hai dataset, vui lòng trích dẫn các công trình nghiên cứu gốc được dẫn trong các repository tương ứng.
