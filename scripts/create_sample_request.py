from __future__ import annotations

import json

import pandas as pd

from kkbox_churn_prediction.config import (
    MODEL_SPLITS_PATH,
    TRAIN_FEATURES_PATH,
)


def main() -> None:
    features = pd.read_parquet(
        TRAIN_FEATURES_PATH
    )

    splits = pd.read_parquet(
        MODEL_SPLITS_PATH
    )

    dataset = features.merge(
        splits,
        on="msno",
        how="inner",
        validate="one_to_one",
    )

    validation = dataset[
        dataset["split"]
        == "validation"
    ].copy()

    row = validation.iloc[0]

    msno = str(
        row["msno"]
    )

    excluded = {
        "msno",
        "is_churn",
        "split",
    }

    feature_values = {}

    for column in validation.columns:
        if column in excluded:
            continue

        value = row[column]

        if pd.isna(value):
            feature_values[
                column
            ] = None

        elif hasattr(
            value,
            "item",
        ):
            feature_values[
                column
            ] = value.item()

        else:
            feature_values[
                column
            ] = value

    payload = {
        "msno": msno,
        "features": feature_values,
    }

    with open(
        "sample_request.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Created sample_request.json"
    )

    print(
        "msno:",
        msno,
    )

    print(
        "actual is_churn:",
        int(
            row["is_churn"]
        ),
    )

    print(
        "feature count:",
        len(
            feature_values
        ),
    )


if __name__ == "__main__":
    main()