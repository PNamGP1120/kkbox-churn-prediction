from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMN = "msno"
TARGET_COLUMN = "is_churn"

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "registered_via",
]


def load_modeling_data(
    path: Path,
) -> pd.DataFrame:
    """
    Load processed feature dataset for modeling.

    Categorical columns are converted to Python string objects.
    Missing categorical values are represented by np.nan so that
    scikit-learn SimpleImputer can handle them consistently.
    """

    df = pd.read_parquet(path)

    for column in CATEGORICAL_FEATURES:
        series = df[column].astype("string")

        df[column] = (
            series
            .astype(object)
            .where(
                series.notna(),
                np.nan,
            )
        )

    return df


def split_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
):
    """
    Split dataset into:

    - 70% train
    - 15% validation
    - 15% test

    Stratification preserves churn rate.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[TARGET_COLUMN],
    )

    return (
        train_df,
        val_df,
        test_df,
    )


def split_xy(
    df: pd.DataFrame,
):
    """
    Separate identifier, model features and target.
    """

    ids = df[ID_COLUMN].copy()

    X = df.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    )

    y = df[TARGET_COLUMN].copy()

    return (
        ids,
        X,
        y,
    )


def save_split_assignments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save msno -> split mapping so QT5 can reuse
    exactly the same test set.
    """

    train_ids = train_df[
        [ID_COLUMN]
    ].copy()

    train_ids["split"] = "train"

    val_ids = val_df[
        [ID_COLUMN]
    ].copy()

    val_ids["split"] = "validation"

    test_ids = test_df[
        [ID_COLUMN]
    ].copy()

    test_ids["split"] = "test"

    assignments = pd.concat(
        [
            train_ids,
            val_ids,
            test_ids,
        ],
        ignore_index=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_parquet(
        output_path,
        index=False,
    )