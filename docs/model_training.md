# Model Training

## KKBOX Customer Churn Prediction Platform

Tài liệu này mô tả quy trình **QT4 — Model Development & Training** của dự án KKBOX Customer Churn Prediction.

Nguồn kết quả chính:

```text
notebooks/03_model_training.ipynb
```

Production training pipeline:

```text
src/kkbox_churn_prediction/modeling/
├── data.py
├── preprocessing.py
├── metrics.py
└── train.py
```

Training được chạy bằng:

```bash
uv run python scripts/train_models.py
```

---

## 1. Mục tiêu QT4

QT4 có các mục tiêu chính:

1. Chia processed dataset thành Train, Validation và Test.
2. Giữ nguyên class distribution bằng stratified split.
3. Fit preprocessing chỉ trên Train.
4. Xây dựng baseline và các candidate models.
5. Đánh giá model trên Validation bằng PR-AUC và ROC-AUC.
6. Kiểm tra train-validation generalization gap.
7. Lưu model artifacts để phục vụ QT5.
8. Giữ Test set hoàn toàn độc lập cho final evaluation.

QT4 không thực hiện:

- Threshold tuning.
- Precision/Recall/F1 optimization.
- Final confusion matrix.
- Probability calibration.
- SHAP/error analysis đầy đủ.
- Final test evaluation.

Các bước trên thuộc QT5.

---

## 2. Input Dataset

QT4 sử dụng dataset đã được tạo ở QT3:

```text
data/processed/train_features.parquet
```

Thông tin dataset:

| Thuộc tính | Giá trị |
|---|---:|
| Users | 970,960 |
| Raw columns | 42 |
| Model candidate features | 40 |
| Identifier | `msno` |
| Target | `is_churn` |
| Churn rate | ~8.99% |

`msno` chỉ được sử dụng làm identifier và không được đưa vào model.

Target:

```text
0 = non-churn
1 = churn
```

Dataset bị class imbalance rõ rệt:

```text
Non-churn ≈ 91.01%
Churn     ≈ 8.99%
```

---

## 3. Train / Validation / Test Split

Dataset được chia theo tỷ lệ:

```text
70% Train
15% Validation
15% Test
```

Kết quả thực tế:

| Split | Rows | Churn rate |
|---|---:|---:|
| Train | 679,672 | 8.99% |
| Validation | 145,644 | 8.99% |
| Test | 145,644 | 8.99% |

Các split sử dụng:

```text
RANDOM_STATE = 42
```

và:

```text
stratify = is_churn
```

để giữ class distribution gần như giống nhau giữa ba tập.

Split assignment được lưu tại:

```text
data/processed/model_splits.parquet
```

Validation từ notebook:

- Tổng rows trong split: **970,960**
- Unique `msno`: **970,960**
- Duplicate `msno`: **0**
- Missing `msno`: **0**

Mục đích của `model_splits.parquet` là đảm bảo QT5 sử dụng chính xác cùng một Test set, không tạo split mới.

---

## 4. Vai trò của từng split

### Train

Train được sử dụng để:

- Fit median imputer.
- Fit missing indicators.
- Fit categorical encoder.
- Fit scaler.
- Train model.

### Validation

Validation được sử dụng để:

- So sánh models.
- Theo dõi generalization.
- Early stopping cho LightGBM.
- Chọn candidate model.

### Test

Test set được khóa trong QT4.

Metadata xác nhận:

```text
test_evaluated = False
```

Không có test metric nào được sử dụng để lựa chọn model hoặc điều chỉnh hyperparameters.

---

## 5. Preprocessing Pipeline

QT3 cung cấp:

```text
40 raw model candidate features
```

Sau preprocessing:

```text
88 processed features
```

### 5.1 Numeric Features

Numeric pipeline:

```text
Missing values
      ↓
Median Imputation
      ↓
Missing Indicators
      ↓
StandardScaler
```

Median imputation được fit chỉ trên Train.

`add_indicator=True` cho phép model nhận biết feature nào ban đầu bị missing.

StandardScaler đặc biệt quan trọng đối với Logistic Regression vì các numeric features có scale rất khác nhau.

### 5.2 Categorical Features

Categorical features:

```text
city
gender
registered_via
```

Pipeline:

```text
Missing values
      ↓
__missing__
      ↓
OneHotEncoder
```

Encoder sử dụng:

```text
handle_unknown = "ignore"
```

để tránh lỗi nếu future data chứa category chưa xuất hiện trong training data.

### 5.3 Leakage Prevention

Quy trình đúng:

```text
Train
  ↓
fit_transform()

Validation
  ↓
transform()
```

Validation và Test không được dùng để fit preprocessing parameters.

Preprocessor được lưu tại:

```text
models/preprocessor.joblib
```

Processed feature names được lưu tại:

```text
models/preprocessed_feature_names.json
```

---

## 6. Evaluation Metrics

Do churn chỉ chiếm khoảng 8.99%, Accuracy không được sử dụng làm primary model-selection metric.

Primary metric:

```text
PR-AUC
```

Secondary metric:

```text
ROC-AUC
```

### PR-AUC

PR-AUC phù hợp cho imbalanced classification vì tập trung vào khả năng ranking positive class.

Dummy baseline PR-AUC dự kiến gần positive prevalence:

```text
~0.0899
```

### ROC-AUC

ROC-AUC được sử dụng như secondary ranking metric để bổ sung cho PR-AUC.

---

## 7. Models

QT4 huấn luyện bốn model:

1. DummyClassifier.
2. Logistic Regression.
3. Random Forest.
4. LightGBM.

---

## 8. DummyClassifier

DummyClassifier được sử dụng làm lower-bound baseline.

Kết quả:

| Metric | Train | Validation |
|---|---:|---:|
| PR-AUC | 0.0899 | 0.0899 |
| ROC-AUC | 0.5000 | 0.5000 |

Kết quả đúng với kỳ vọng:

```text
PR-AUC ≈ churn prevalence
ROC-AUC = 0.5
```

Điều này xác nhận evaluation pipeline hoạt động hợp lý.

---

## 9. Logistic Regression

Logistic Regression đóng vai trò linear baseline.

Cấu hình chính:

```text
solver = saga
class_weight = balanced
```

Kết quả:

| Metric | Train | Validation |
|---|---:|---:|
| PR-AUC | 0.8030 | 0.8120 |
| ROC-AUC | 0.9695 | 0.9695 |

Generalization gap:

```text
PR-AUC gap  = -0.0090
ROC-AUC gap =  0.0001
```

Model vượt Dummy baseline rất mạnh, cho thấy feature set có predictive signal đáng kể ngay cả với linear classifier.

Training xuất hiện `ConvergenceWarning`, nghĩa là solver đạt `max_iter` trước khi coefficients hội tụ hoàn toàn. Vì Logistic Regression chỉ đóng vai trò baseline, model vẫn được giữ để benchmark nhưng không phải candidate chính.

---

## 10. Random Forest

Random Forest được sử dụng để học nonlinear relationships và feature interactions.

Kết quả:

| Metric | Train | Validation |
|---|---:|---:|
| PR-AUC | 0.9352 | 0.9175 |
| ROC-AUC | 0.9901 | 0.9865 |

Generalization gap:

```text
PR-AUC gap  = 0.0176
ROC-AUC gap = 0.0037
```

Random Forest vượt Logistic Regression rõ rệt.

Train performance cao hơn Validation performance, cho thấy overfitting nhẹ, nhưng gap chưa ở mức nghiêm trọng.

Random Forest được giữ làm **secondary candidate** cho QT5.

---

## 11. LightGBM

LightGBM là gradient boosting model chính cho tabular dataset.

Class imbalance được xử lý bằng:

```text
scale_pos_weight = 10.1183
```

Train class counts:

```text
Negative samples = 618,541
Positive samples = 61,131
```

Early stopping sử dụng:

```text
average_precision
```

và đạt:

```text
best_iteration = 786
```

Kết quả:

| Metric | Train | Validation |
|---|---:|---:|
| PR-AUC | 0.9505 | 0.9345 |
| ROC-AUC | 0.9939 | 0.9898 |

Generalization gap:

```text
PR-AUC gap  = 0.0160
ROC-AUC gap = 0.0041
```

LightGBM đạt cả Validation PR-AUC và Validation ROC-AUC cao nhất.

Do đó LightGBM được chọn làm **primary candidate model** cho QT5.

---

## 12. Model Comparison

Validation ranking:

| Rank | Model | Validation PR-AUC | Validation ROC-AUC |
|---:|---|---:|---:|
| 1 | LightGBM | **0.9345** | **0.9898** |
| 2 | Random Forest | 0.9175 | 0.9865 |
| 3 | Logistic Regression | 0.8120 | 0.9695 |
| 4 | DummyClassifier | 0.0899 | 0.5000 |

PR-AUC lift so với Dummy:

| Model | PR-AUC lift |
|---|---:|
| LightGBM | 10.3892x |
| Random Forest | 10.2011x |
| Logistic Regression | 9.0277x |
| Dummy | 1.0000x |

LightGBM vượt Random Forest khoảng:

```text
0.9345 - 0.9175 ≈ 0.0170 PR-AUC
```

---

## 13. Generalization Analysis

| Model | PR-AUC Gap | ROC-AUC Gap |
|---|---:|---:|
| Random Forest | 0.0176 | 0.0037 |
| LightGBM | 0.0160 | 0.0041 |
| Dummy | ~0.0000 | 0.0000 |
| Logistic Regression | -0.0090 | 0.0001 |

### LightGBM

Có overfitting nhẹ nhưng validation performance vẫn rất cao.

### Random Forest

Overfitting nhẹ và tương đương LightGBM.

### Logistic Regression

Train và Validation performance gần như tương đương.

### DummyClassifier

Không có meaningful train-validation gap vì model không học predictive patterns.

Không model nào cho thấy overfitting nghiêm trọng trong QT4.

---

## 14. Saved Artifacts

Sau training, các artifact chính gồm:

```text
models/
├── dummy_classifier.joblib
├── logistic_regression.joblib
├── random_forest.joblib
├── lightgbm.joblib
├── preprocessor.joblib
├── preprocessed_feature_names.json
├── validation_results.json
└── training_metadata.json
```

Split assignment:

```text
data/processed/model_splits.parquet
```

Các artifacts này đảm bảo reproducibility và cho phép QT5 tiếp tục mà không retrain từ đầu.

---

## 15. Automated Tests

Test suite được chạy bằng:

```bash
uv run pytest -v
```

Kết quả:

```text
11 passed in 0.99s
```

Các tests xác nhận:

- Split không làm mất rows.
- Train / Validation / Test không overlap.
- Class ratio được giữ bằng stratification.
- `msno` và target không được đưa vào model features.
- Processed dataset tồn tại.
- `1 msno = 1 row`.
- Row count đúng.
- Target không missing.
- Target binary.
- Transaction features không vượt cutoff.
- User-log features không vượt cutoff.

---

## 16. QT4 Completion Criteria

QT4 được xem là hoàn thành vì:

- Processed dataset load thành công.
- Dataset split đúng 70/15/15.
- Stratified split hoạt động đúng.
- Split assignment đã được lưu.
- Preprocessing được fit chỉ trên Train.
- 40 raw features được chuyển thành 88 processed features.
- Dummy baseline đã được train.
- Logistic Regression đã được train.
- Random Forest đã được train.
- LightGBM đã được train.
- PR-AUC và ROC-AUC đã được tính trên Validation.
- LightGBM là model validation tốt nhất.
- Model artifacts đã được lưu.
- Test set chưa được đánh giá.
- 11 automated tests đều pass.
- `03_model_training.ipynb` đã hoàn thành.

---

## 17. Candidate Models for QT5

Primary candidate:

```text
LightGBM
```

Validation:

```text
PR-AUC  = 0.9345
ROC-AUC = 0.9898
```

Secondary candidate:

```text
Random Forest
```

Validation:

```text
PR-AUC  = 0.9175
ROC-AUC = 0.9865
```

LightGBM chưa được gọi là final model vì Test set chưa được mở.

---

## 18. Next Step — QT5

QT5 sẽ sử dụng Validation set để:

1. Phân tích Precision-Recall Curve.
2. Phân tích ROC Curve.
3. Lựa chọn classification threshold.
4. Tính Precision.
5. Tính Recall.
6. Tính F1-score.
7. Phân tích Confusion Matrix.
8. Phân tích False Positive và False Negative.
9. Kiểm tra probability calibration.
10. Tính Brier Score.
11. Thực hiện error analysis.
12. Phân tích feature importance / SHAP.
13. So sánh LightGBM và Random Forest nếu cần.

Sau khi khóa:

```text
model
hyperparameters
threshold
calibration strategy
```

Test set mới được sử dụng cho final evaluation.
