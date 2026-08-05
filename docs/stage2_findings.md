# Stage 2 Findings — Data Processing & Feature Engineering

## KKBOX Customer Churn Prediction Platform

Tài liệu này tổng hợp kết quả thực tế của **QT3 — Data Processing & Feature Engineering**, dựa trên `02_feature_engineering.ipynb` và các feature artifacts được tạo bởi production pipeline.

---

## 1. Executive Summary

QT3 đã chuyển 4 nguồn raw data của KKBOX thành một training dataset ở user-level:

```text
data/processed/train_features.parquet
```

Kết quả cuối:

- **970,960 rows**
- **970,960 unique users**
- **42 columns**
- **40 model candidate features**
- `msno` không missing
- `is_churn` không missing
- Target chỉ chứa `0/1`
- Không phát hiện time-based leakage
- Các feature tables đều đảm bảo 1 row/user

Invariant quan trọng đã đạt:

```text
1 msno = 1 row
```

---

## 2. Pipeline Architecture

Production logic được tách khỏi notebook:

```text
src/kkbox_churn_prediction/
├── data/
│   └── validation.py
└── features/
    ├── members.py
    ├── transactions.py
    ├── user_logs.py
    └── build_dataset.py
```

Pipeline được chạy qua:

```bash
uv run python scripts/build_features.py
```

Artifacts:

```text
data/interim/
├── members_features.parquet
├── transactions_features.parquet
└── user_logs_features.parquet

data/processed/
└── train_features.parquet
```

`02_feature_engineering.ipynb` được dùng để kiểm tra và document kết quả, không duplicate production feature logic.

---

## 3. Final Dataset Integrity

Validation từ notebook:

| Check | Result |
|---|---:|
| Rows | 970,960 |
| Unique users | 970,960 |
| Missing `msno` | 0 |
| Missing target | 0 |
| Invalid target | 0 |
| One row per user | Pass |

Kết luận:

> Dataset cuối giữ nguyên toàn bộ training population và không phát sinh duplicate user sau merge.

---

## 4. Source Coverage

Final dataset giữ lại source availability bằng các feature:

- `has_member_data`
- `has_transaction_data`
- `has_log_data`

Coverage:

| Source | Coverage |
|---|---:|
| Members | **88.67%** |
| Transactions | **96.15%** |
| User logs | **77.71%** |

Các tỷ lệ này khớp với QT2, xác nhận aggregation và LEFT JOIN hoạt động đúng.

### Implication

Không phải user nào cũng có dữ liệu ở mọi source.

Missing source data được giữ như thông tin thay vì drop training samples.

---

## 5. Member Processing Findings

Cleaning chính:

- Chỉ giữ age trong miền `1–100`.
- Invalid age được chuyển thành missing.
- Missing gender trong member record được chuyển thành `unknown`.
- Registration date được chuyển thành `account_age_days` tương đối với cutoff.
- Không drop user vì missing profile fields.

### Validation result

- Age min: **1**
- Age median: **28**
- Age mean: **29.904**
- Age max: **100**
- Invalid age sau cleaning: **0**
- Missing age: **584,245 users (60.17%)**
- Median account age: **1,034 days**
- Negative account age: **0**

Gender sau processing:

- `unknown`: **472,062 users (48.62%)**
- `male`: **204,561 (21.07%)**
- `female`: **184,344 (18.99%)**
- NULL do không có member profile: **109,993 (11.33%)**

### Finding

Missing/invalid member data vẫn là hạn chế lớn nhất của nhóm profile features, đặc biệt là `age`.

---

## 6. Transaction Feature Engineering Findings

Transaction history được aggregate thành một record/user.

Intermediate table:

- **1,197,050 rows**
- **1,197,050 unique users**

Feature groups:

- Frequency
- Payment
- Plan
- Discount
- Auto-renew
- Cancellation
- Recency

### Summary

- Median transaction count: **1**
- Mean transaction count: **1.195**
- Max transaction count: **208**
- Median average payment: **149**
- Mean average payment: **299.201**
- Mean auto-renew rate: **0.770**
- Mean cancel rate: **0.017**
- Median days since last transaction: **17**
- Max transaction recency: **820 days**
- Invalid auto-renew rate: **0**
- Invalid cancel rate: **0**

`membership_expire_date` chưa được đưa vào feature V1 vì QT2 đã phát hiện các giá trị rất xa trong tương lai.

### Missing

Khoảng **37,382 training users (3.85%)** không có các transaction-derived averages/rates/recency.

Các count feature có business meaning phù hợp được giữ ở `0`.

---

## 7. User Log Feature Engineering Findings

Listening logs được aggregate thành user-level activity features.

Intermediate table:

- **1,103,894 rows**
- **1,103,894 unique users**

Feature groups:

- Activity frequency
- Listening duration
- Song diversity
- Song completion behavior
- Activity recency
- Recent 7-day window
- Recent 30-day window
- Extreme-listening quality signal

### Activity summary

- Active days median: **18**
- Active days mean: **16.665**
- Active days max: **31**
- Median total listening seconds: **73,828.011**
- Mean total listening seconds: **131,559.486**
- Median days since last activity: **1**
- Mean active days 7d: **3.770**
- Mean active days 30d: **16.143**

### Window validation

All checks passed:

- `active_days_7d <= 7`
- `active_days_30d <= 30`
- `active_days_7d <= active_days_30d`
- `listening_secs_7d <= listening_secs_30d`

Invalid count cho tất cả các rule trên: **0**.

### Completion behavior

- Min completion rate: **0**
- Median: **0.731**
- Mean: **0.679**
- Max: **1**
- Invalid completion rate: **0**

### Extreme listening

QT2 đã phát hiện `total_secs` có extreme values.

QT3 giữ lại tín hiệu này qua:

```text
extreme_listening_day_count
```

Kết quả:

- Users có ít nhất một extreme listening day: **2,638**
- Max extreme days/user: **31**

### Missing

**216,409 training users (22.29%)** thiếu các log-derived averages/recency vì không có log data.

---

## 8. Missing-value Findings

Final dataset chủ động giữ một số missing values thay vì impute trên toàn dataset.

Các missing đáng chú ý:

| Feature | Missing users | Missing % |
|---|---:|---:|
| `age` | 584,245 | 60.17% |
| `avg_listening_secs_per_day` | 216,409 | 22.29% |
| `median_listening_secs_per_day` | 216,409 | 22.29% |
| `avg_unique_songs` | 216,409 | 22.29% |
| `completion_rate` | 216,409 | 22.29% |
| `days_since_last_activity` | 216,409 | 22.29% |
| `city` | 109,993 | 11.33% |
| `gender` | 109,993 | 11.33% |
| `registered_via` | 109,993 | 11.33% |
| `account_age_days` | 109,994 | 11.33% |
| Transaction averages/rates | ~37,382 | 3.85% |

### Decision

QT3 **không fit median/mean imputation trên toàn dataset**.

Statistical imputation sẽ được fit chỉ trên training split ở QT4.

Điều này tránh để validation/test distribution ảnh hưởng preprocessing parameters.

---

## 9. Feature Distribution Findings

Một số median đáng chú ý trong final training dataset:

| Feature | Median |
|---|---:|
| `age` | 28 |
| `account_age_days` | 1,034 |
| `transaction_count` | 1 |
| `avg_paid` | 149 |
| `cancel_rate` | 0 |
| `active_days` | 14 |
| `total_listening_secs` | 49,982.129 |
| `days_since_last_activity` | 1 |

Selected percentiles:

| Feature | P50 | P95 | P99 |
|---|---:|---:|---:|
| `total_paid` | 149 | 298 | 1,299 |
| `total_listening_secs` | 49,982.129 | 412,943.803 | 848,807.756 |

Một số numeric features có phân phối lệch mạnh. QT3 chưa thực hiện scaling/log transformation vì các transformation phụ thuộc model và phải được fit đúng trên training split ở QT4.

---

## 10. Leakage Validation

Cutoff:

```text
CUTOFF_DATE = 2017-04-01
```

Results:

| Check | Result |
|---|---|
| Latest transaction used | `2017-03-31` |
| Transactions at/after cutoff | 0 |
| Latest activity used | `2017-03-31` |
| Activities at/after cutoff | 0 |

Kết luận:

> Không phát hiện time-based leakage trong transaction hoặc listening feature pipeline.

---

## 11. Target-based Sanity Check

Phần này chỉ được dùng để xác nhận aggregation vẫn giữ các pattern quan sát từ QT2, không dùng để feature-select trên toàn dataset.

| Metric | Non-churn | Churn |
|---|---:|---:|
| Users | 883,630 | 87,330 |
| Avg transaction count | 1.153 | 1.301 |
| Avg auto-renew rate | 0.935 | 0.568 |
| Avg cancel rate | 0.006 | 0.243 |
| Avg active days | 14.097 | 12.327 |
| Median listening secs | 51,167.386 | 38,419.620 |
| Avg days since last activity | 3.291 | 7.919 |

### Interpretation

Feature dataset vẫn giữ được các pattern chính đã quan sát ở QT2:

- Churn users có auto-renew thấp hơn rõ rệt.
- Churn users có cancel rate cao hơn rõ rệt.
- Churn users ít active days hơn.
- Churn users có tổng listening thấp hơn.
- Churn users có activity recency lớn hơn, tức lâu không hoạt động hơn.

Đây là **association**, không phải quan hệ nhân quả.

---

## 12. Final Feature Set

Notebook xác nhận:

```text
Total model candidate features: 40
Missing expected: set()
Unexpected: set()
```

Feature groups:

- Member features: **6**
- Transaction features: **15**
- User log features: **19**

Total:

```text
40 candidate features
```

`msno` và `is_churn` không nằm trong candidate feature list.

---

## 13. Final Validation Status

Final notebook checks:

| Check | Status |
|---|---|
| Processed dataset exists | PASS |
| Member feature artifact exists | PASS |
| Transaction feature artifact exists | PASS |
| Log feature artifact exists | PASS |
| One row per user | PASS |
| Target complete | PASS |
| Target valid | PASS |
| Transaction leakage free | PASS |
| Log leakage free | PASS |

Tất cả final checks đều trả về:

```text
True
```

Ngoài notebook validation, automated QT3 test suite trước đó cũng đã pass toàn bộ.

---

## 14. Known Limitations

1. `age` missing rất lớn dù invalid raw age đã được clean.
2. Member profile coverage chỉ khoảng 88.67%.
3. User-log coverage chỉ khoảng 77.71%.
4. Một số payment/listening variables có distribution lệch mạnh.
5. `membership_expire_date` chưa được sử dụng ở feature V1.
6. Current feature set chưa chứa activity trend giữa các consecutive windows.
7. Missing-value imputation chưa được fit — đây là chủ đích và thuộc QT4.

---

## 15. QT3 Completion Decision

QT3 được xem là hoàn thành vì:

- Raw data không bị thay đổi.
- Feature artifacts đã được tạo.
- Final processed dataset đã được tạo.
- `1 msno = 1 row`.
- Target hợp lệ.
- Source coverage được bảo toàn.
- Time-based leakage checks pass.
- Feature ranges/windows hợp lệ.
- Expected feature set đầy đủ.
- Final checks đều pass.

Output chính để chuyển sang QT4:

```text
data/processed/train_features.parquet
```

---

## 16. Next Step — QT4 Model Development

QT4 nên bắt đầu từ processed dataset, không quay lại train trực tiếp từ raw CSV.

Thứ tự đề xuất:

1. Load `train_features.parquet`.
2. Tách identifier, target và candidate features.
3. Train / validation / test split.
4. Phân loại numeric/categorical features.
5. Fit imputer chỉ trên training split.
6. Encode categorical features khi cần.
7. Xây `DummyClassifier` baseline.
8. Train Logistic Regression.
9. Train tree-based model.
10. Đánh giá bằng PR-AUC, ROC-AUC, Recall, Precision và F1.
11. Tune threshold và model sau khi baseline rõ ràng.
