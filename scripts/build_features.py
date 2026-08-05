from kkbox_churn_prediction.config import (
    CUTOFF_DATE,
    INTERIM_DIR,
    PROCESSED_DIR,
    TRAIN_PATH,
    MEMBERS_PATH,
    TRANSACTIONS_PATH,
    USER_LOGS_PATH,
    MEMBER_FEATURES_PATH,
    TRANSACTION_FEATURES_PATH,
    USER_LOG_FEATURES_PATH,
    TRAIN_FEATURES_PATH,
)

from kkbox_churn_prediction.data.validation import (
    validate_all,
    validate_processed_dataset,
)

from kkbox_churn_prediction.features.members import (
    build_member_features,
)

from kkbox_churn_prediction.features.transactions import (
    build_transaction_features,
)

from kkbox_churn_prediction.features.user_logs import (
    build_user_log_features,
)

from kkbox_churn_prediction.features.build_dataset import (
    build_training_dataset,
)


def main() -> None:
    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("1. Validating raw data...")

    validate_all(
        TRAIN_PATH,
        MEMBERS_PATH,
        TRANSACTIONS_PATH,
        USER_LOGS_PATH,
    )

    print("2. Building member features...")

    build_member_features(
        MEMBERS_PATH,
        MEMBER_FEATURES_PATH,
        CUTOFF_DATE,
    )

    print("3. Building transaction features...")

    build_transaction_features(
        TRANSACTIONS_PATH,
        TRANSACTION_FEATURES_PATH,
        CUTOFF_DATE,
    )

    print("4. Building user log features...")

    build_user_log_features(
        USER_LOGS_PATH,
        USER_LOG_FEATURES_PATH,
        CUTOFF_DATE,
    )

    print("5. Building final training dataset...")

    build_training_dataset(
        TRAIN_PATH,
        MEMBER_FEATURES_PATH,
        TRANSACTION_FEATURES_PATH,
        USER_LOG_FEATURES_PATH,
        TRAIN_FEATURES_PATH,
    )

    print("6. Validating final dataset...")

    validate_processed_dataset(
        TRAIN_FEATURES_PATH,
        expected_rows=970_960,
    )

    print()
    print("Done.")
    print(
        f"Output: {TRAIN_FEATURES_PATH}"
    )


if __name__ == "__main__":
    main()