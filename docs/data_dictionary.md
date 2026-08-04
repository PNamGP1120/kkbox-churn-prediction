# Data Dictionary

## KKBOX Customer Churn Prediction Platform

Tài liệu này mô tả 4 nguồn dữ liệu được sử dụng trong dự án và các vấn đề chất lượng dữ liệu đã quan sát trong `01_data_understanding.ipynb`.

---

## 1. `train_v2.csv`

**Vai trò:** Training labels  
**Grain:** 1 row / user  
**Số dòng:** 970,960  
**Số user duy nhất:** 970,960

| Column | Kiểu dữ liệu | Vai trò | Mô tả | Ghi chú |
|---|---|---|---|---|
| `msno` | string | Identifier | Mã định danh user đã được ẩn danh | Unique, không missing; không dùng làm model feature |
| `is_churn` | integer | Target | Nhãn churn của user | `0` = non-churn, `1` = churn |

### Target distribution

- Non-churn (`0`): 883,630 users — **91.01%**
- Churn (`1`): 87,330 users — **8.99%**

**Data quality note:** Target có class imbalance rõ ràng.

---

## 2. `members_v3.csv`

**Vai trò:** User profile  
**Grain:** 1 row / user  
**Số dòng:** 6,769,473  
**Số user duy nhất:** 6,769,473

| Column | Kiểu dữ liệu quan sát | Vai trò | Mô tả | Data quality / QT3 action |
|---|---|---|---|---|
| `msno` | string | Identifier | Mã định danh user | Unique, không missing |
| `city` | integer/category | Categorical | Mã khu vực/thành phố | Không missing trong dữ liệu đã kiểm tra |
| `bd` | integer | Numeric | Tuổi user | 67.15% nằm ngoài miền `0 < age <= 100`; min = -7168, max = 2016 |
| `gender` | string/category | Categorical | Giới tính | Missing 4,429,505 rows — **65.43%** |
| `registered_via` | integer/category | Categorical | Kênh/phương thức đăng ký | Có category đáng chú ý `-1` |
| `registration_init_time` | integer `YYYYMMDD` | Date | Ngày đăng ký tài khoản | Cần chuyển sang datetime ở QT3 |

### Registration date range

- Min: `2004-03-26`
- Max: `2017-04-29`
- Invalid parsed dates: `0`

---

## 3. `transactions_v2.csv`

**Vai trò:** Subscription / payment history  
**Grain:** N rows / user  
**Số dòng:** 1,431,009  
**Số user duy nhất:** 1,197,050

| Column | Kiểu dữ liệu logic | Vai trò | Mô tả | Data quality / QT3 action |
|---|---|---|---|---|
| `msno` | string | Identifier | Mã định danh user | Không missing |
| `payment_method_id` | integer/category | Categorical | Mã phương thức thanh toán | Không missing |
| `payment_plan_days` | integer | Numeric | Số ngày của gói đăng ký | Có **2,218** rows `<= 0`, cần điều tra |
| `plan_list_price` | numeric | Numeric | Giá niêm yết của gói | Min = 0, max = 2000 |
| `actual_amount_paid` | numeric | Numeric | Số tiền thực tế đã thanh toán | Min = 0, median = 149, max = 2000 |
| `is_auto_renew` | integer/binary | Categorical | Trạng thái tự động gia hạn | `1`: 78.53%, `0`: 21.47% |
| `transaction_date` | integer `YYYYMMDD` | Date | Ngày giao dịch | Cần chuyển sang datetime |
| `membership_expire_date` | integer `YYYYMMDD` | Date | Ngày hết hạn membership | Max = `2036-10-15`, cần kiểm tra anomaly |
| `is_cancel` | integer/binary | Categorical | Trạng thái hủy | `1`: 2.46%, `0`: 97.54% |

### Transaction statistics

- Min transactions/user: 1
- Mean transactions/user: 1.20
- Median transactions/user: 1
- Max transactions/user: 208
- Duplicate full rows: 0
- Missing values ở các cột đã kiểm tra: 0

### Date range

- Transaction date: `2015-01-01` → `2017-03-31`
- Membership expire date: `2016-04-19` → `2036-10-15`

**Important:** Bảng phải được aggregate về `msno` trước khi join vào training dataset.

---

## 4. `user_logs_v2.csv`

**Vai trò:** Daily listening behavior  
**Grain:** User-day level  
**Số dòng:** 18,396,362  
**Số user duy nhất:** 1,103,894

| Column | Kiểu dữ liệu logic | Vai trò | Mô tả | Data quality / QT3 action |
|---|---|---|---|---|
| `msno` | string | Identifier | Mã định danh user | Không missing |
| `date` | integer `YYYYMMDD` | Date | Ngày ghi nhận hoạt động | Cần chuyển sang datetime |
| `num_25` | integer | Numeric | Số lượt nghe ở nhóm hoàn thành khoảng 25% | Không phát hiện giá trị âm |
| `num_50` | integer | Numeric | Số lượt nghe ở nhóm hoàn thành khoảng 50% | Không phát hiện giá trị âm |
| `num_75` | integer | Numeric | Số lượt nghe ở nhóm hoàn thành khoảng 75% | Không phát hiện giá trị âm |
| `num_985` | integer | Numeric | Số lượt nghe ở nhóm hoàn thành khoảng 98.5% | Không phát hiện giá trị âm |
| `num_100` | integer | Numeric | Số lượt nghe hoàn thành bài | Không phát hiện giá trị âm |
| `num_unq` | integer | Numeric | Số bài hát unique đã nghe | Không phát hiện giá trị âm |
| `total_secs` | numeric | Numeric | Tổng số giây nghe trong record ngày | Có extreme outlier rất lớn |

### Listening statistics

| Column | Mean | Max |
|---|---:|---:|
| `num_25` | 6.19 | 5,639 |
| `num_50` | 1.51 | 912 |
| `num_75` | 0.94 | 508 |
| `num_985` | 1.08 | 1,561 |
| `num_100` | 30.28 | 41,107 |
| `num_unq` | 29.04 | 4,925 |

### `total_secs` distribution

- Min: 0.00
- P50: 4,582.99
- P90: 19,476.59
- P95: 28,188.53
- P99: 43,805.52
- Max: **9,194,058.52**

### Activity period

- Date range: `2017-03-01` → `2017-03-31`
- Mean active days/user: 16.66
- Median active days/user: 18
- Max active days/user: 31
- Duplicate `msno + date` groups: 0
- Missing values ở các cột đã kiểm tra: 0

**Important:** `total_secs` có extreme values cần được điều tra trước khi tạo listening features.

---

## 5. Cross-table Relationship

| Relationship | Cardinality | Training-user coverage |
|---|---|---:|
| `train` → `members` | 0..1 profile / user | **88.67%** |
| `train` → `transactions` | N transactions / user | **96.15%** |
| `train` → `user_logs` | N daily logs / user | **77.71%** |

Các bảng liên kết bằng:

```text
msno
```

QT3 nên aggregate `transactions` và `user_logs` về user-level trước, sau đó bắt đầu từ `train_v2` và sử dụng chiến lược join phù hợp để tránh làm mất training users.

---

## 6. Modeling Constraint

Dataset cuối cùng phải đảm bảo:

```text
1 msno = 1 row
```

`msno` chỉ dùng để join và trace prediction, không sử dụng làm model feature.

Mọi feature theo thời gian phải được xây dựng với `cutoff_date` phù hợp để tránh data leakage.
