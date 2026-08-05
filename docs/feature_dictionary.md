# Feature Dictionary

## KKBOX Customer Churn Prediction Platform

Tài liệu này mô tả feature set được tạo trong **QT3 — Data Processing & Feature Engineering** và được lưu tại:

```text
data/processed/train_features.parquet
```

Dataset cuối có:

- **970,960 rows**
- **970,960 unique users**
- **42 columns**
- **40 model candidate features**
- `msno` là identifier
- `is_churn` là target

Invariant chính:

```text
1 msno = 1 row
```

---

## 1. Identifier & Target

| Column | Nhóm | Kiểu logic | Ý nghĩa | Modeling note |
|---|---|---|---|---|
| `msno` | Identifier | string | Mã user đã ẩn danh | Chỉ dùng join/trace, không dùng làm model feature |
| `is_churn` | Target | binary | `0` = non-churn, `1` = churn | Target của bài toán |

---

## 2. Source Availability Features

Các feature này cho biết user có dữ liệu tương ứng trong source table hay không.

| Feature | Source | Ý nghĩa | Range / Missing |
|---|---|---|---|
| `has_member_data` | members | User có member profile | 0/1, không missing |
| `has_transaction_data` | transactions | User có transaction history | 0/1, không missing |
| `has_log_data` | user_logs | User có listening logs | 0/1, không missing |

Coverage sau Feature Engineering:

- Member coverage: **88.67%**
- Transaction coverage: **96.15%**
- Log coverage: **77.71%**

---

## 3. Member Features

| Feature | Source | Kiểu | Ý nghĩa | Cleaning / Missing note |
|---|---|---|---|---|
| `city` | members | categorical | Mã city/khu vực của user | Missing khi user không có member profile |
| `age` | members | numeric | Tuổi hợp lệ của user | Chỉ giữ giá trị `1–100`; invalid age chuyển thành missing |
| `gender` | members | categorical | Giới tính | Missing raw được chuyển thành `unknown`; vẫn NULL nếu user không có member profile |
| `registered_via` | members | categorical | Kênh đăng ký | Missing khi user không có member profile |
| `account_age_days` | members | numeric | Số ngày từ ngày đăng ký đến cutoff | Chỉ tính khi registration date hợp lệ và trước cutoff |

### Member validation results

- `age` min: **1**
- `age` median: **28**
- `age` mean: **29.904**
- `age` max: **100**
- Invalid age sau cleaning: **0**
- Missing `age`: **584,245 users (60.17%)**
- `account_age_days` median: **1,034 days**
- Negative account age: **0**

Gender distribution sau processing:

| Value | Users | Percentage |
|---|---:|---:|
| `unknown` | 472,062 | 48.62% |
| `male` | 204,561 | 21.07% |
| `female` | 184,344 | 18.99% |
| NULL | 109,993 | 11.33% |

NULL ở `gender` chủ yếu biểu diễn trường hợp user không có member profile; `unknown` biểu diễn member record tồn tại nhưng gender bị thiếu.

---

## 4. Transaction Features

Transaction data được aggregate:

```text
N transaction rows
        ↓
GROUP BY msno
        ↓
1 row / user
```

| Feature | Nhóm | Kiểu | Ý nghĩa | Missing / Note |
|---|---|---|---|---|
| `transaction_count` | Frequency | numeric | Số transaction của user trước cutoff | 0 nếu user không có transaction |
| `total_paid` | Payment | numeric | Tổng số tiền đã thanh toán | Missing nếu không có transaction |
| `avg_paid` | Payment | numeric | Số tiền thanh toán trung bình | Missing nếu không có transaction |
| `min_paid` | Payment | numeric | Số tiền thanh toán nhỏ nhất | Missing nếu không có transaction |
| `max_paid` | Payment | numeric | Số tiền thanh toán lớn nhất | Missing nếu không có transaction |
| `avg_plan_price` | Plan | numeric | Giá gói trung bình | Missing nếu không có transaction |
| `avg_plan_days` | Plan | numeric | Số ngày plan trung bình trên các plan hợp lệ | Non-positive plan days không tham gia average |
| `avg_discount` | Payment | numeric | Discount trung bình: `plan_list_price - actual_amount_paid` | Missing nếu không có transaction |
| `auto_renew_rate` | Renewal | numeric | Tỷ lệ transaction có auto-renew | Range `[0,1]` |
| `last_auto_renew` | Renewal | binary | Auto-renew status ở transaction gần nhất | Missing nếu không có transaction |
| `cancel_count` | Cancellation | numeric | Số transaction có `is_cancel = 1` | 0 nếu không có transaction |
| `cancel_rate` | Cancellation | numeric | Tỷ lệ transaction bị cancel | Range `[0,1]` |
| `last_is_cancel` | Cancellation | binary | Cancel status ở transaction gần nhất | Missing nếu không có transaction |
| `days_since_last_transaction` | Recency | numeric | Số ngày từ transaction gần nhất đến cutoff | Missing nếu không có transaction |

### Transaction validation results

- Transaction feature rows: **1,197,050**
- Unique users: **1,197,050**
- Median transaction count: **1**
- Mean transaction count: **1.195**
- Max transaction count: **208**
- Median `avg_paid`: **149**
- Mean `avg_paid`: **299.201**
- Mean `auto_renew_rate`: **0.770**
- Mean `cancel_rate`: **0.017**
- Median `days_since_last_transaction`: **17**
- Max `days_since_last_transaction`: **820**
- Invalid `auto_renew_rate`: **0**
- Invalid `cancel_rate`: **0**

Missing transaction-derived continuous/rate features trong final dataset:

- Phần lớn: **37,382 users (3.85%)**
- `avg_plan_days`: **37,383 users (3.85%)**

---

## 5. User Log Features

User logs được aggregate:

```text
Daily listening records
        ↓
GROUP BY msno
        ↓
1 row / user
```

| Feature | Nhóm | Kiểu | Ý nghĩa | Missing / Note |
|---|---|---|---|---|
| `active_days` | Frequency | numeric | Tổng số ngày user có activity | 0 nếu không có logs |
| `total_listening_secs` | Duration | numeric | Tổng thời gian nghe sau cleaning | 0 nếu không có logs |
| `avg_listening_secs_per_day` | Duration | numeric | Thời gian nghe trung bình mỗi active day | Missing nếu không có logs |
| `median_listening_secs_per_day` | Duration | numeric | Median thời gian nghe mỗi active day | Missing nếu không có logs |
| `avg_unique_songs` | Diversity | numeric | Số bài unique trung bình mỗi ngày | Missing nếu không có logs |
| `total_unique_song_events` | Diversity | numeric | Tổng `num_unq` qua các ngày | 0 nếu không có logs |
| `total_num_25` | Completion | numeric | Tổng lượt nghe thuộc nhóm `num_25` | 0 nếu không có logs |
| `total_num_50` | Completion | numeric | Tổng lượt nghe thuộc nhóm `num_50` | 0 nếu không có logs |
| `total_num_75` | Completion | numeric | Tổng lượt nghe thuộc nhóm `num_75` | 0 nếu không có logs |
| `total_num_985` | Completion | numeric | Tổng lượt nghe thuộc nhóm `num_985` | 0 nếu không có logs |
| `total_num_100` | Completion | numeric | Tổng lượt nghe hoàn thành | 0 nếu không có logs |
| `completion_rate` | Completion | numeric | `num_100 / tổng các completion buckets` | Range `[0,1]`, missing nếu denominator không tồn tại |
| `days_since_last_activity` | Recency | numeric | Số ngày từ activity gần nhất đến cutoff | Missing nếu không có logs |
| `active_days_7d` | Window | numeric | Số active days trong 7 ngày gần cutoff | Range `0–7` |
| `active_days_30d` | Window | numeric | Số active days trong 30 ngày gần cutoff | Range `0–30` |
| `listening_secs_7d` | Window | numeric | Listening seconds trong 7 ngày gần cutoff | 0 nếu không có logs |
| `listening_secs_30d` | Window | numeric | Listening seconds trong 30 ngày gần cutoff | 0 nếu không có logs |
| `extreme_listening_day_count` | Quality signal | numeric | Số ngày có listening duration vượt cleaning threshold | 0 nếu không có |

### User log validation results

- User-log feature rows: **1,103,894**
- Unique users: **1,103,894**
- Active days median: **18**
- Active days mean: **16.665**
- Active days max: **31**
- Median total listening seconds: **73,828.011**
- Mean total listening seconds: **131,559.486**
- Median activity recency: **1 day**
- Mean active days in last 7 days: **3.770**
- Mean active days in last 30 days: **16.143**
- Invalid `active_days_7d > 7`: **0**
- Invalid `active_days_30d > 30`: **0**
- Invalid `active_days_7d > active_days_30d`: **0**
- Invalid `listening_secs_7d > listening_secs_30d`: **0**
- Completion rate min: **0**
- Completion rate median: **0.731**
- Completion rate mean: **0.679**
- Completion rate max: **1**
- Invalid completion rate: **0**
- Users có ít nhất một extreme listening day: **2,638**
- Max extreme listening days/user: **31**

Missing các log-derived averages/recency features:

- **216,409 users (22.29%)**

Bao gồm:

- `avg_listening_secs_per_day`
- `median_listening_secs_per_day`
- `avg_unique_songs`
- `completion_rate`
- `days_since_last_activity`

---

## 6. Final Feature Groups

### Member features — 6

```text
has_member_data
city
age
gender
registered_via
account_age_days
```

### Transaction features — 15

```text
has_transaction_data
transaction_count
total_paid
avg_paid
min_paid
max_paid
avg_plan_price
avg_plan_days
avg_discount
auto_renew_rate
last_auto_renew
cancel_count
cancel_rate
last_is_cancel
days_since_last_transaction
```

### User log features — 19

```text
has_log_data
active_days
total_listening_secs
avg_listening_secs_per_day
median_listening_secs_per_day
avg_unique_songs
total_unique_song_events
total_num_25
total_num_50
total_num_75
total_num_985
total_num_100
completion_rate
days_since_last_activity
active_days_7d
active_days_30d
listening_secs_7d
listening_secs_30d
extreme_listening_day_count
```

Tổng cộng:

```text
40 model candidate features
```

Notebook xác nhận:

```text
Missing expected: set()
Unexpected: set()
```

---

## 7. Missing-value Strategy

QT3 không thực hiện global statistical imputation.

Nguyên tắc:

- Count feature có ý nghĩa “không có activity” có thể được fill `0`.
- Missing categorical/profile vẫn được giữ có chủ đích.
- Missing continuous values không được fill bằng median/mean trên toàn dataset.
- Statistical imputation sẽ được **fit trên training split ở QT4** để tránh leakage.

Các missing lớn nhất trong final feature dataset:

| Feature group / Feature | Missing | Percentage |
|---|---:|---:|
| `age` | 584,245 | 60.17% |
| Log averages / recency | 216,409 | 22.29% |
| `city`, `gender`, `registered_via` | 109,993 | 11.33% |
| `account_age_days` | 109,994 | 11.33% |
| Transaction averages / rates / recency | ~37,382 | 3.85% |

---

## 8. Leakage Constraint

Prediction cutoff:

```text
CUTOFF_DATE = 2017-04-01
```

Validation result:

- Latest transaction used: **2017-03-31**
- Transaction users at/after cutoff: **0**
- Latest listening activity used: **2017-03-31**
- Log users at/after cutoff: **0**

Do not use any feature data at or after cutoff.

---

## 9. QT4 Modeling Note

The following are **not** part of QT3:

- Train/validation/test split
- Statistical imputation fit
- Standardization/scaling
- One-hot encoding fit
- SMOTE/resampling
- Feature selection using target
- Model training

Các bước này phải được thực hiện sau split trong QT4 để hạn chế data leakage.
