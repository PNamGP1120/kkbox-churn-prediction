# Problem Definition

## KKBOX Customer Churn Prediction Platform

---

## 1. Business Problem

KKBOX là nền tảng nghe nhạc theo mô hình thuê bao. Một số người dùng tiếp tục gia hạn dịch vụ, trong khi một số khác ngừng sử dụng hoặc không tiếp tục gia hạn.

Mục tiêu của dự án là phát hiện sớm những người dùng có nguy cơ churn để KKBOX có thể chủ động triển khai các biện pháp giữ chân khách hàng như:

- Nhắc gia hạn dịch vụ.
- Gửi ưu đãi hoặc voucher.
- Chăm sóc khách hàng.
- Cá nhân hóa retention campaign.

Business objective:

> Xác định và xếp hạng những người dùng có nguy cơ churn cao trước khi họ thực sự rời bỏ dịch vụ.

---

## 2. Machine Learning Problem

Bài toán được mô hình hóa dưới dạng:

```text
Supervised Learning
        ↓
Binary Classification
```

Mô hình sử dụng thông tin hồ sơ người dùng, lịch sử thanh toán và hành vi nghe nhạc để ước lượng xác suất một user sẽ churn.

Target:

- `is_churn = 0`: Non-churn.
- `is_churn = 1`: Churn.

---

## 3. Prediction Unit

Mỗi prediction tương ứng với một user KKBOX.

User được định danh bởi:

```text
msno
```

Dataset cuối dùng để huấn luyện phải đảm bảo:

```text
1 msno = 1 row = 1 prediction
```

`msno` chỉ được dùng để:

- Join các bảng dữ liệu.
- Trace user.
- Lưu và truy xuất kết quả prediction.

`msno` không được sử dụng làm model feature.

---

## 4. Data Sources

Dự án sử dụng 4 nguồn dữ liệu chính.

### `train_v2.csv`

Vai trò:

- Chứa user identifier.
- Chứa target `is_churn`.
- Dùng làm tập gốc để xây training dataset.

### `members_v3.csv`

Chứa thông tin hồ sơ user như:

- `city`
- `bd`
- `gender`
- `registered_via`
- `registration_init_time`

### `transactions_v2.csv`

Chứa lịch sử subscription và thanh toán:

- `payment_method_id`
- `payment_plan_days`
- `plan_list_price`
- `actual_amount_paid`
- `is_auto_renew`
- `transaction_date`
- `membership_expire_date`
- `is_cancel`

Một user có thể có nhiều transaction records.

### `user_logs_v2.csv`

Chứa hành vi nghe nhạc theo ngày:

- `date`
- `num_25`
- `num_50`
- `num_75`
- `num_985`
- `num_100`
- `num_unq`
- `total_secs`

Một user có thể có nhiều activity records.

---

## 5. Prediction Time

Mô hình phải thực hiện prediction tại một thời điểm xác định:

```text
cutoff_date
```

Chỉ dữ liệu xảy ra trước thời điểm này mới được phép dùng để tạo feature:

```text
feature_date < cutoff_date
```

Dữ liệu xảy ra sau `cutoff_date` không được sử dụng nhằm tránh **Data Leakage**.

Logic tổng quát:

```text
Past Data             Prediction Time            Future
───────────────|──────────────────────────────>
               ^
          cutoff_date
```

---

## 6. Model Input

Raw data không được đưa trực tiếp vào model.

Sau Data Cleaning và Feature Engineering, mỗi user sẽ được biểu diễn bởi một tập feature ở user-level.

### Member Features

Ví dụ:

- `age`
- `city`
- `gender`
- `registered_via`
- `account_age_days`

### Transaction Features

Ví dụ:

- `transaction_count`
- `total_paid`
- `avg_paid`
- `auto_renew_rate`
- `cancel_rate`
- `days_since_last_transaction`

### User Activity Features

Ví dụ:

- `active_days`
- `total_listening_secs`
- `avg_listening_secs`
- `avg_unique_songs`
- `activity_last_7d`
- `activity_last_30d`
- `activity_change`

Dataset cuối phải có dạng:

```text
msno
member_features...
transaction_features...
activity_features...
is_churn
```

---

## 7. Model Output

Output chính của mô hình là:

```text
churn_probability ∈ [0, 1]
```

Ví dụ:

```text
churn_probability = 0.87
```

có nghĩa model ước lượng user có xác suất churn khoảng 87%.

Output hệ thống có thể gồm:

```json
{
  "msno": "abc123",
  "churn_probability": 0.87,
  "prediction": 1,
  "risk_level": "high"
}
```

Phiên bản nâng cao có thể bổ sung các yếu tố ảnh hưởng chính bằng SHAP.

---

## 8. Risk Classification

Probability có thể được chuyển thành risk level để phục vụ business action.

Ví dụ:

```text
Low Risk
Medium Risk
High Risk
```

Threshold cụ thể chưa được cố định ở QT1.

Threshold sẽ được lựa chọn ở giai đoạn Model Evaluation dựa trên:

- Precision.
- Recall.
- F1-score.
- Business cost.

---

## 9. Evaluation Metrics

Do target churn mất cân bằng, Accuracy không được sử dụng làm metric chính.

Các metric cần theo dõi:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Confusion Matrix.

Trong bài toán churn, Recall của class churn đặc biệt quan trọng vì:

```text
False Negative
=
User thực sự churn
nhưng model dự đoán non-churn
```

Điều này khiến doanh nghiệp bỏ lỡ cơ hội giữ chân khách hàng.

---

## 10. Error Cost

### False Positive

```text
Actual: Non-churn
Prediction: Churn
```

Hậu quả:

- Có thể gửi ưu đãi không cần thiết.
- Tăng chi phí retention campaign.

### False Negative

```text
Actual: Churn
Prediction: Non-churn
```

Hậu quả:

- Không phát hiện user có nguy cơ.
- Mất cơ hội can thiệp.
- Có thể mất khách hàng và doanh thu.

Trong project này, False Negative được xem là loại lỗi cần đặc biệt quan tâm.

---

## 11. Business Usage

Luồng sử dụng prediction:

```text
User Data
    ↓
Feature Pipeline
    ↓
Churn Model
    ↓
Churn Probability
    ↓
Risk Ranking
    ↓
Retention Campaign
```

Business có thể dùng output để:

- Xếp hạng user theo churn risk.
- Ưu tiên nhóm high-risk.
- Gửi renewal reminder.
- Gửi voucher hoặc discount.
- Thực hiện customer care.
- Theo dõi hiệu quả retention campaign.

---

## 12. Constraints

### Data Quality

Dữ liệu có thể chứa:

- Missing values.
- Invalid values.
- Outliers.
- Anomalous dates.
- Multiple records/user.

Các vấn đề này phải được xử lý trước modeling.

### Data Leakage

Không sử dụng dữ liệu xảy ra sau `cutoff_date`.

### Identifier Leakage

Không dùng `msno` làm feature.

### Large-scale Data

`transactions_v2.csv` và đặc biệt `user_logs_v2.csv` có kích thước lớn.

Pipeline nên ưu tiên các công cụ phù hợp như:

- DuckDB.
- Polars.
- Parquet.

### User-level Requirement

Trước modeling:

```text
1 msno = 1 row
```

---

## 13. Success Criteria

Dự án được xem là thành công khi:

1. Xây dựng được user-level dataset với `1 msno = 1 row`.
2. Không xảy ra data leakage.
3. Model vượt baseline trên các metric phù hợp.
4. Model trả về churn probability.
5. Có thể xác định và xếp hạng nhóm high-risk users.
6. Prediction có thể được giải thích ở mức feature.
7. Model có thể được đóng gói thành service để tích hợp vào ứng dụng khác.

---

## 14. End-to-End Goal

Luồng cuối cùng của dự án:

```text
KKBOX Raw Data
        ↓
Data Validation
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
User-level Dataset
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Churn Probability
        ↓
Risk Ranking
        ↓
FastAPI
        ↓
PostgreSQL
        ↓
Retention Decision
```

Mục tiêu cuối cùng là xây dựng một **end-to-end Machine Learning system** có khả năng xử lý dữ liệu KKBOX, dự đoán churn và cung cấp kết quả có thể tích hợp vào hệ thống thực tế.
