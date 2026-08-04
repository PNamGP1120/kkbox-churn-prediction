# Data Understanding Findings

## KKBOX Customer Churn Prediction Platform

Tài liệu này tổng hợp kết quả của **QT2 — Thu thập, tìm hiểu và phân tích dữ liệu**, dựa trên output thực tế của `01_data_understanding.ipynb`.

---

## 1. Executive Summary

Bốn nguồn dữ liệu KKBOX có cấu trúc phù hợp để xây dựng bài toán churn prediction, nhưng chưa thể đưa trực tiếp vào mô hình.

Các vấn đề chính được phát hiện:

1. Target `is_churn` mất cân bằng: churn chỉ chiếm **8.99%**.
2. `members_v3.csv` có vấn đề lớn ở `bd` và `gender`.
3. `transactions_v2.csv` và `user_logs_v2.csv` là bảng one-to-many theo user và bắt buộc phải aggregate.
4. Không phải mọi training user đều xuất hiện trong mọi source table.
5. `user_logs_v2.csv` chứa extreme values rất lớn ở `total_secs`.
6. Một số giá trị thời gian / subscription cần được kiểm tra thêm trước Feature Engineering.
7. QT3 phải kiểm soát `cutoff_date` để tránh data leakage.

---

## 2. Train Dataset

### Observations

- Total users: **970,960**
- Unique `msno`: **970,960**
- Duplicate `msno`: **0**
- Missing `msno`: **0**
- Missing target: **0**
- Target values: `{0, 1}`

### Target distribution

| Class | Users | Percentage |
|---|---:|---:|
| Non-churn (`0`) | 883,630 | 91.01% |
| Churn (`1`) | 87,330 | 8.99% |

### Finding

Target có **class imbalance rõ ràng**.

### Implication

Ở QT5 không nên chỉ dùng Accuracy. Cần ưu tiên các metric phù hợp với imbalanced classification như Precision, Recall, F1-score, ROC-AUC và PR-AUC.

---

## 3. Members Dataset

### Observations

- Rows: **6,769,473**
- Unique users: **6,769,473**
- Duplicate users: **0**
- Missing `gender`: **4,429,505 rows (65.43%)**
- Invalid age (`bd <= 0` hoặc `bd > 100`): **4,545,866 rows (67.15%)**
- Age range quan sát: **-7168 → 2016**
- `registered_via` có một giá trị `-1`
- Registration date range: **2004-03-26 → 2017-04-29**
- Invalid parsed registration dates: **0**

### Findings

- `bd` không thể sử dụng trực tiếp làm age feature.
- `gender` có missing rate rất cao nên cần chiến lược xử lý rõ ràng.
- `registered_via = -1` cần được xác minh trước khi encode.
- `registration_init_time` cần chuyển sang datetime.

### QT3 actions

- Chuyển invalid age sang missing hoặc áp dụng chiến lược cleaning được document rõ.
- Không drop user chỉ vì thiếu gender.
- Kiểm tra category bất thường.
- Tạo các feature thời gian từ registration date chỉ sau khi xác định `cutoff_date`.

---

## 4. Transactions Dataset

### Observations

- Rows: **1,431,009**
- Unique users: **1,197,050**
- Mean transactions/user: **1.20**
- Median transactions/user: **1**
- Max transactions/user: **208**
- Missing values ở các cột đã kiểm tra: **0**
- Duplicate full rows: **0**
- `payment_plan_days <= 0`: **2,218 rows**
- `actual_amount_paid`: median **149**, max **2000**
- Auto renew:
  - `1`: **78.53%**
  - `0`: **21.47%**
- Cancel:
  - `1`: **2.46%**
  - `0`: **97.54%**
- Transaction date: **2015-01-01 → 2017-03-31**
- Membership expire date: **2016-04-19 → 2036-10-15**

### Findings

- Một user có thể có nhiều transaction records; bảng không thể join trực tiếp vào train.
- `payment_plan_days <= 0` cần điều tra.
- `membership_expire_date` có extreme future values cần kiểm tra.
- Auto-renew và cancellation là hai nhóm biến có tín hiệu churn mạnh trong EDA.

### Basic churn EDA

| Metric | Non-churn | Churn |
|---|---:|---:|
| Mean transaction count | 1.16 | 1.96 |
| Median transaction count | 1.00 | 1.00 |
| Mean paid amount | 129.00 | 367.03 |
| Mean auto-renew rate | 0.93 | 0.57 |
| Mean cancel rate | 0.01 | 0.24 |

### Interpretation

Trong dữ liệu quan sát:

- Churn users có `cancel_rate` cao hơn rõ rệt.
- Churn users có `auto_renew_rate` thấp hơn rõ rệt.
- Payment behavior khác biệt giữa hai nhóm.

Đây là **association**, chưa phải quan hệ nhân quả.

---

## 5. User Logs Dataset

### Observations

- Rows: **18,396,362**
- Unique users: **1,103,894**
- Date range: **2017-03-01 → 2017-03-31**
- Missing values ở các cột đã kiểm tra: **0**
- Negative listening values: **0 rows**
- Duplicate `msno + date`: **0 groups**
- Mean active days/user: **16.66**
- Median active days/user: **18**

### `total_secs`

| Statistic | Value |
|---|---:|
| P50 | 4,582.99 |
| P90 | 19,476.59 |
| P95 | 28,188.53 |
| P99 | 43,805.52 |
| Max | 9,194,058.52 |

### Finding

`total_secs` có một extreme maximum vượt rất xa P99 và cần được điều tra trước khi tạo listening features.

### Basic churn EDA

| Metric | Non-churn | Churn |
|---|---:|---:|
| Mean active days | 18.13 | 15.91 |
| Median active days | 19 | 16 |
| Mean total seconds | 141,113.08 | 126,291.23 |
| Median total seconds | 83,814.77 | 70,180.34 |
| Mean seconds/day | 6,473.17 | 6,456.91 |
| Median seconds/day | 4,708.15 | 4,717.18 |
| Mean unique songs | 24.29 | 24.71 |
| Median unique songs | 19.20 | 19.50 |

### Interpretation

- Churn users có **ít active days hơn**.
- Churn users có **tổng thời gian nghe thấp hơn** ở cả mean và median.
- `seconds/day` gần như tương đương giữa hai nhóm.
- `avg_unique_songs` cũng gần nhau.

Điều này gợi ý rằng **mức độ thường xuyên hoạt động** có thể hữu ích hơn cường độ nghe trong một ngày. Đây mới là giả thuyết Feature Engineering và cần được kiểm chứng ở QT3/QT4.

---

## 6. Member Profile vs Churn

### Gender

| Gender | Users | Churn rate |
|---|---:|---:|
| Female | 184,344 | ~0.13 |
| Male | 204,561 | ~0.13 |
| Missing | 582,055 | ~0.06 |

Do `gender` missing rất nhiều, không nên diễn giải khác biệt này theo hướng nhân quả.

### Registration channel

| `registered_via` | Users | Churn rate |
|---:|---:|---:|
| 4 | 52,744 | 0.23 |
| 3 | 106,459 | 0.17 |
| 9 | 235,689 | 0.13 |
| 13 | 3,391 | 0.10 |
| Missing profile | 109,993 | 0.05 |
| 7 | 462,684 | 0.04 |

Có khác biệt churn rate đáng kể giữa một số nhóm `registered_via`; biến này đáng được giữ lại để kiểm thử ở bước modeling.

---

## 7. Cross-table Coverage

| Source | Training users matched | Coverage |
|---|---:|---:|
| Members | 860,967 | **88.67%** |
| Transactions | 933,578 | **96.15%** |
| User logs | 754,551 | **77.71%** |

### Finding

Không phải mọi training user đều có dữ liệu từ tất cả nguồn.

### Implication for QT3

Không nên tự động dùng INNER JOIN vì có thể làm mất nhiều training samples.

Pipeline nên:

1. Aggregate source tables về `msno`.
2. Bắt đầu từ `train_v2`.
3. LEFT JOIN các feature tables.
4. Tạo chiến lược xử lý rõ ràng cho trường hợp user không có source data.

---

## 8. Main Data Quality Risks

| Risk | Mức độ | Hướng xử lý ở QT3 |
|---|---|---|
| Target imbalance | Cao | Không xử lý trong QT2; cân nhắc metric/weighting ở modeling |
| Invalid `bd` | Cao | Cleaning trước khi tạo age feature |
| Missing `gender` | Cao | Giữ missing như một trạng thái hoặc chiến lược imputation phù hợp |
| Missing source coverage | Cao | LEFT JOIN + missing indicators / imputation strategy |
| Extreme `total_secs` | Cao | Điều tra distribution và rule xử lý |
| `payment_plan_days <= 0` | Trung bình | Xác minh ý nghĩa trước khi cleaning |
| `membership_expire_date` rất xa | Trung bình | Kiểm tra anomaly/business meaning |
| `registered_via = -1` | Thấp/Trung bình | Xác minh category |
| Data leakage | Rất cao | Mọi time-based feature phải tuân thủ `cutoff_date` |

---

## 9. Constraints for QT3

Dataset cuối phải thỏa:

```text
1 msno = 1 row
```

Các bảng `transactions_v2.csv` và `user_logs_v2.csv` phải được aggregate trước khi merge.

Mọi feature có yếu tố thời gian phải được xây dựng từ dữ liệu xảy ra trước `cutoff_date`.

---

## 10. Recommended Next Steps

QT3 nên thực hiện theo thứ tự:

1. Xác định chính xác `cutoff_date` và observation window.
2. Viết Data Validation rules.
3. Clean `members`.
4. Clean transactions và kiểm tra các date/payment anomalies.
5. Clean user logs và xử lý extreme values.
6. Xây member features.
7. Aggregate transaction features theo `msno`.
8. Aggregate listening/activity features theo `msno`.
9. LEFT JOIN feature tables vào `train_v2`.
10. Kiểm tra lại `1 msno = 1 row`.
11. Lưu user-level dataset dưới dạng Parquet.

Expected output:

```text
data/processed/train_features.parquet
```
