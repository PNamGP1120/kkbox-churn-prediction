import pandas as pd

from kkbox_churn_prediction.modeling.data import (
    split_dataset,
    split_xy,
)


def create_test_dataframe(
    n: int = 1000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": [
                f"user_{i}"
                for i in range(n)
            ],

            "feature": list(range(n)),

            "is_churn": (
                [0] * 900
                + [1] * 100
            ),
        }
    )


def test_split_preserves_all_rows():
    df = create_test_dataframe()

    train_df, val_df, test_df = (
        split_dataset(df)
    )

    assert (
        len(train_df)
        + len(val_df)
        + len(test_df)
        == len(df)
    )


def test_split_has_no_overlap():
    df = create_test_dataframe()

    train_df, val_df, test_df = (
        split_dataset(df)
    )

    train_ids = set(train_df["msno"])
    val_ids = set(val_df["msno"])
    test_ids = set(test_df["msno"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_preserves_class_ratio():
    df = create_test_dataframe()

    train_df, val_df, test_df = (
        split_dataset(df)
    )

    expected = 0.10

    assert abs(
        train_df["is_churn"].mean()
        - expected
    ) < 0.01

    assert abs(
        val_df["is_churn"].mean()
        - expected
    ) < 0.01

    assert abs(
        test_df["is_churn"].mean()
        - expected
    ) < 0.01


def test_identifier_and_target_removed():
    df = create_test_dataframe()

    train_df, _, _ = split_dataset(df)

    _, X, y = split_xy(train_df)

    assert "msno" not in X.columns
    assert "is_churn" not in X.columns

    assert len(X) == len(y)