# Stage 3 Findings — Model Development & Training

## KKBOX Customer Churn Prediction Platform

Tài liệu này tổng hợp các kết quả chính của **QT4 — Model Development & Training** dựa trên output thực tế của `03_model_training.ipynb`.

---

## 1. Executive Summary

QT4 đã hoàn thành toàn bộ pipeline model development từ processed dataset đến candidate-model selection.

Dataset:

```text
970,960 users
40 raw candidate features
```

Sau preprocessing:

```text
88 processed features
```

Split:

```text
Train       679,672
Validation  145,644
Test        145,644
```

Churn rate của cả ba split:

```text
~8.99%
```

Primary metric:

```text
PR-AUC
```

Secondary metric:

```text
ROC-AUC
```

Model tốt nhất trên Validation:

```text
LightGBM
```

với:

```text
Validation PR-AUC  = 0.9345
Validation ROC-AUC = 0.9898
Best iteration     = 786
```

Test set chưa được sử dụng.

---

## 2. Dataset Split Findings

Stratified split hoạt động đúng.

| Split | Rows | Churn rate |
|---|---:|---:|
| Train | 679,672 | 8.99% |
| Validation | 145,644 | 8.99% |
| Test | 145,644 | 8.99% |

Integrity:

- Total split rows: **970,960**
- Unique users: **970,960**
- Duplicate users: **0**
- Missing `msno`: **0**

Split assignment được lưu tại:

```text
data/processed/model_splits.parquet
```

Điều này đảm bảo QT5 tái sử dụng đúng cùng Test set.

---

## 3. Preprocessing Findings

Modeling dataset có:

```text
40 raw features
```

Sau preprocessing:

```text
88 features
```

Numeric pipeline:

```text
Median Imputation
+
Missing Indicators
+
StandardScaler
```

Categorical pipeline:

```text
Missing Category
+
OneHotEncoder
```

Preprocessing được fit chỉ trên Train.

Validation chỉ được transform.

Test không được sử dụng để fit preprocessing.

Không phát hiện preprocessing leakage trong QT4.

---

## 4. Dummy Baseline Findings

DummyClassifier:

```text
Train PR-AUC       = 0.0899
Validation PR-AUC  = 0.0899
Train ROC-AUC      = 0.5000
Validation ROC-AUC = 0.5000
```

PR-AUC gần churn prevalence (~8.99%), đúng với kỳ vọng của lower-bound baseline.

Các model thực tế đều vượt Dummy rất lớn.

---

## 5. Logistic Regression Findings

Logistic Regression:

```text
Train PR-AUC       = 0.8030
Validation PR-AUC  = 0.8120
Train ROC-AUC      = 0.9695
Validation ROC-AUC = 0.9695
```

Generalization:

```text
PR-AUC gap  = -0.0090
ROC-AUC gap =  0.0001
```

### Interpretation

- Linear model đã đạt performance cao.
- Feature set chứa predictive signal mạnh.
- Train và Validation gần nhau.
- Không có dấu hiệu overfitting đáng kể.
- Training xuất hiện `ConvergenceWarning`, nên Logistic Regression được giữ làm baseline thay vì candidate chính.

---

## 6. Random Forest Findings

Random Forest:

```text
Train PR-AUC       = 0.9352
Validation PR-AUC  = 0.9175
Train ROC-AUC      = 0.9901
Validation ROC-AUC = 0.9865
```

Generalization:

```text
PR-AUC gap  = 0.0176
ROC-AUC gap = 0.0037
```

### Interpretation

Random Forest vượt Logistic Regression rõ rệt.

Điều này cho thấy nonlinear relationships và feature interactions đóng vai trò quan trọng trong churn prediction.

Model có overfitting nhẹ nhưng validation performance vẫn rất mạnh.

Random Forest được giữ làm secondary candidate.

---

## 7. LightGBM Findings

LightGBM:

```text
Train PR-AUC       = 0.9505
Validation PR-AUC  = 0.9345
Train ROC-AUC      = 0.9939
Validation ROC-AUC = 0.9898
```

Generalization:

```text
PR-AUC gap  = 0.0160
ROC-AUC gap = 0.0041
```

Imbalance handling:

```text
scale_pos_weight = 10.1183
```

Early stopping:

```text
metric         = average_precision
best_iteration = 786
```

### Interpretation

LightGBM đạt Validation PR-AUC cao nhất và Validation ROC-AUC cao nhất.

Performance cao hơn Random Forest trong cả hai ranking metrics.

Generalization gap cho thấy overfitting nhẹ nhưng chưa đáng lo ngại.

LightGBM được chọn làm **primary candidate** cho QT5.

---

## 8. Final Validation Ranking

| Rank | Model | Validation PR-AUC | Validation ROC-AUC |
|---:|---|---:|---:|
| 1 | LightGBM | **0.9345** | **0.9898** |
| 2 | Random Forest | 0.9175 | 0.9865 |
| 3 | Logistic Regression | 0.8120 | 0.9695 |
| 4 | DummyClassifier | 0.0899 | 0.5000 |

PR-AUC lift over Dummy:

| Model | Lift |
|---|---:|
| LightGBM | 10.3892x |
| Random Forest | 10.2011x |
| Logistic Regression | 9.0277x |
| Dummy | 1.0000x |

---

## 9. Generalization Findings

Train-validation PR-AUC gaps:

```text
Random Forest       0.0176
LightGBM            0.0160
Dummy              ~0.0000
Logistic Regression -0.0090
```

Không model nào cho thấy train-validation gap đủ lớn để kết luận overfitting nghiêm trọng.

LightGBM và Random Forest đều có overfitting nhẹ, nhưng Validation scores vẫn cao.

---

## 10. Model Selection Decision

### Primary candidate

```text
LightGBM
```

Lý do:

1. Validation PR-AUC cao nhất: **0.9345**.
2. Validation ROC-AUC cao nhất: **0.9898**.
3. Early stopping hoạt động trên `average_precision`.
4. Generalization gap vẫn hợp lý.
5. Phù hợp với large tabular dataset.

### Secondary candidate

```text
Random Forest
```

Random Forest vẫn được giữ để so sánh nếu QT5 phát hiện LightGBM có vấn đề về threshold behavior, calibration hoặc business trade-offs.

### Không gọi LightGBM là final model

Model chưa thể được gọi là final vì:

```text
Test set chưa được đánh giá.
```

---

## 11. Test-set Protection

Training metadata xác nhận:

```text
test_evaluated = False
```

Test set không được sử dụng cho:

- Model selection.
- Hyperparameter adjustment.
- Early stopping.
- Threshold tuning.
- Calibration decisions.

Điều này bảo toàn Test set như unbiased final evaluation set cho QT5.

---

## 12. Automated Test Results

Lệnh:

```bash
uv run pytest -v
```

Kết quả:

```text
11 passed in 0.99s
```

Test suite xác nhận:

- Split giữ đủ rows.
- Không overlap giữa Train / Validation / Test.
- Class ratio được bảo toàn.
- Identifier và target không đi vào X.
- Processed dataset tồn tại.
- One row per user.
- Expected row count đúng.
- Target đầy đủ.
- Target binary.
- Transaction leakage check pass.
- User-log leakage check pass.

---

## 13. QT4 Validation Checks

`03_model_training.ipynb` xác nhận toàn bộ checks:

```text
split_file_exists                 True
preprocessor_exists               True
training_metadata_exists          True
training_results_exists           True
all_users_in_splits               True
unique_users_in_splits            True
raw_feature_count_40              True
processed_feature_count_88        True
test_not_evaluated                True
lightgbm_best_validation_model    True
```

---

## 14. QT4 Completion Decision

QT4 được xem là **hoàn thành**.

Đã hoàn thành:

```text
Processed data loading             ✓
Train/Validation/Test split        ✓
Stratification                     ✓
Split reproducibility              ✓
Preprocessing                      ✓
Imbalance handling                 ✓
Dummy baseline                     ✓
Logistic Regression               ✓
Random Forest                      ✓
LightGBM                           ✓
PR-AUC comparison                  ✓
ROC-AUC comparison                 ✓
Generalization analysis            ✓
Candidate selection                ✓
Model artifact persistence         ✓
Automated tests                    ✓
Test-set protection                ✓
03_model_training.ipynb            ✓
```

---

## 15. Known Limitation

### Logistic Regression convergence

Logistic Regression training xuất hiện:

```text
ConvergenceWarning
```

Model đạt `max_iter` trước khi solver hội tụ hoàn toàn.

Tuy nhiên:

- Logistic Regression chỉ được sử dụng làm linear baseline.
- Candidate chính là LightGBM.
- Candidate phụ là Random Forest.

Do đó warning này không ngăn QT4 được xem là hoàn thành, nhưng nên được ghi lại để đảm bảo reproducibility.

### Probability calibration

LightGBM sử dụng:

```text
scale_pos_weight = 10.1183
```

nên raw `predict_proba()` chưa được mặc định xem là calibrated business probability.

Calibration sẽ được kiểm tra ở QT5.

---

## 16. Handoff to QT5

QT5 bắt đầu với:

```text
Primary candidate:
LightGBM

Secondary candidate:
Random Forest
```

Validation sẽ được sử dụng để:

1. Precision-Recall Curve.
2. ROC Curve.
3. Threshold selection.
4. Precision / Recall / F1.
5. Confusion Matrix.
6. False Positive / False Negative analysis.
7. Probability calibration.
8. Brier Score.
9. Error analysis.
10. Feature importance / SHAP.

Sau khi toàn bộ decisions được khóa:

```text
model
hyperparameters
threshold
calibration
```

Test set mới được mở để thực hiện final evaluation.
