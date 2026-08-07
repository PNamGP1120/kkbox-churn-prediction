from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kkbox_churn_prediction.config import (
    EVALUATION_RESULTS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_NAMES_PATH,
    FINAL_MODEL_CONFIG_PATH,
    LIGHTGBM_MODEL_PATH,
    MODEL_SPLITS_PATH,
    PREPROCESSOR_PATH,
    TEST_PREDICTIONS_PATH,
    THRESHOLD_ANALYSIS_PATH,
    TRAIN_FEATURES_PATH,
    VALIDATION_PREDICTIONS_PATH,
)

from kkbox_churn_prediction.modeling.evaluation import (
    add_predictions_and_error_types,
    build_threshold_table,
    calculate_probability_metrics,
    calculate_threshold_metrics,
    find_best_f1_threshold,
    find_threshold_for_min_recall,
)


CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "registered_via",
]

EXCLUDED_COLUMNS = [
    "msno",
    "is_churn",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KKBOX churn model."
        )
    )

    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--validation-only",
        action="store_true",

        help=(
            "Evaluate Validation only. "
            "Test is not evaluated."
        ),
    )

    mode.add_argument(
        "--lock-config",
        action="store_true",

        help=(
            "Lock final model and threshold "
            "before Test evaluation."
        ),
    )

    mode.add_argument(
        "--final-test",
        action="store_true",

        help=(
            "Evaluate locked model on Test."
        ),
    )

    parser.add_argument(
        "--strategy",

        choices=[
            "max_f1",
            "recall_80",
            "threshold_0_5",
        ],

        default="max_f1",

        help=(
            "Threshold strategy used when "
            "--lock-config is selected."
        ),
    )

    return parser.parse_args()


def prepare_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the same categorical dtype
    preparation used during QT4 training.
    """

    X = X.copy()

    for column in (
        CATEGORICAL_FEATURES
    ):
        series = (
            X[column]
            .astype("string")
        )

        X[column] = (
            series
            .astype(object)
            .where(
                series.notna(),
                np.nan,
            )
        )

    return X


def save_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_json(
    path: Path,
) -> dict:
    with path.open(
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_joined_dataset(
) -> pd.DataFrame:
    """
    Load processed features and attach
    the fixed QT4 split assignment.

    DO NOT create a new split here.
    """

    df = pd.read_parquet(
        TRAIN_FEATURES_PATH
    )

    splits = pd.read_parquet(
        MODEL_SPLITS_PATH
    )

    joined = df.merge(
        splits,

        on="msno",

        how="inner",

        validate="one_to_one",
    )

    if len(joined) != len(df):
        raise ValueError(
            "Split mapping does not cover "
            "the whole processed dataset."
        )

    if not joined[
        "msno"
    ].is_unique:
        raise ValueError(
            "Duplicate msno after join."
        )

    return joined


def load_model_artifacts():
    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    model = joblib.load(
        LIGHTGBM_MODEL_PATH
    )

    return (
        preprocessor,
        model,
    )


def build_xy(
    split_df: pd.DataFrame,
):
    ids = split_df[
        "msno"
    ].copy()

    y = split_df[
        "is_churn"
    ].copy()

    X = split_df.drop(
        columns=EXCLUDED_COLUMNS
    )

    X = prepare_features(
        X
    )

    return (
        ids,
        X,
        y,
    )


# ============================================================
# VALIDATION PHASE
# ============================================================


def run_validation_only(
) -> None:
    print(
        "=" * 70
    )

    print(
        "KKBOX CHURN — QT5 VALIDATION"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "1. Loading dataset and "
        "fixed split assignments..."
    )

    joined = (
        load_joined_dataset()
    )

    validation_df = joined[
        joined["split"]
        == "validation"
    ].copy()

    print(
        "Validation rows:",
        f"{len(validation_df):,}",
    )

    print()
    print(
        "2. Loading QT4 model artifacts..."
    )

    (
        preprocessor,
        model,
    ) = load_model_artifacts()

    (
        val_ids,
        X_val,
        y_val,
    ) = build_xy(
        validation_df
    )

    print()
    print(
        "3. Transforming Validation..."
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    print(
        "Processed Validation shape:",
        X_val_processed.shape,
    )

    print()
    print(
        "4. Generating probabilities..."
    )

    val_probabilities = (
        model.predict_proba(
            X_val_processed
        )[:, 1]
    )

    probability_metrics = (
        calculate_probability_metrics(
            y_val,
            val_probabilities,
        )
    )

    print()
    print(
        "Validation probability metrics:"
    )

    print(
        json.dumps(
            probability_metrics,
            indent=2,
        )
    )

    # QT4 LightGBM PR-AUC ≈ 0.9345.
    # If reconstruction is wrong,
    # stop before Test.
    if probability_metrics[
        "pr_auc"
    ] < 0.90:
        raise ValueError(
            "Validation PR-AUC is "
            "unexpectedly low. "
            "Stop before Test."
        )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    print()
    print(
        "5. Building threshold table..."
    )

    threshold_table = (
        build_threshold_table(
            y_val,
            val_probabilities,
        )
    )

    THRESHOLD_ANALYSIS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_table.to_parquet(
        THRESHOLD_ANALYSIS_PATH,
        index=False,
    )

    threshold_05 = (
        calculate_threshold_metrics(
            y_val,
            val_probabilities,
            threshold=0.5,
        )
    )

    best_f1 = (
        find_best_f1_threshold(
            threshold_table
        )
    )

    best_f1_metrics = (
        calculate_threshold_metrics(
            y_val,
            val_probabilities,

            threshold=(
                best_f1[
                    "threshold"
                ]
            ),
        )
    )

    recall_80 = (
        find_threshold_for_min_recall(
            threshold_table,
            min_recall=0.80,
        )
    )

    recall_80_metrics = (
        calculate_threshold_metrics(
            y_val,
            val_probabilities,

            threshold=(
                recall_80[
                    "threshold"
                ]
            ),
        )
    )

    print()
    print(
        "Threshold = 0.50:"
    )

    print(
        json.dumps(
            threshold_05,
            indent=2,
        )
    )

    print()
    print(
        "Best F1 threshold:"
    )

    print(
        json.dumps(
            best_f1_metrics,
            indent=2,
        )
    )

    print()
    print(
        "Recall >= 0.80 scenario:"
    )

    print(
        json.dumps(
            recall_80_metrics,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    selected_threshold = float(
        best_f1[
            "threshold"
        ]
    )

    validation_predictions = (
        pd.DataFrame(
            {
                "msno": (
                    val_ids.to_numpy()
                ),

                "is_churn": (
                    y_val.to_numpy()
                ),

                "churn_probability": (
                    val_probabilities
                ),
            }
        )
    )

    validation_predictions = (
        add_predictions_and_error_types(
            validation_predictions,
            threshold=(
                selected_threshold
            ),
        )
    )

    VALIDATION_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_predictions.to_parquet(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print()
    print(
        "6. Saving feature importance..."
    )

    with FEATURE_NAMES_PATH.open(
        encoding="utf-8",
    ) as file:
        feature_names = (
            json.load(file)
        )

    importance = (
        model.booster_
        .feature_importance(
            importance_type="gain"
        )
    )

    if len(
        feature_names
    ) != len(
        importance
    ):
        raise ValueError(
            "Feature-name count does "
            "not match model importance."
        )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": (
                    feature_names
                ),

                "gain_importance": (
                    importance
                ),
            }
        )
        .sort_values(
            "gain_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance.to_parquet(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    results = {
        "selected_model_candidate": (
            "lightgbm"
        ),

        "test_evaluated": False,

        "validation": {
            "probability_metrics": (
                probability_metrics
            ),

            "threshold_0_5": (
                threshold_05
            ),

            "max_f1": (
                best_f1_metrics
            ),

            "recall_80": (
                recall_80_metrics
            ),
        },
    }

    save_json(
        EVALUATION_RESULTS_PATH,
        results,
    )

    # Suggested config only.
    # It is NOT locked yet.
    suggested_config = {
        "model": "lightgbm",

        "model_path": str(
            LIGHTGBM_MODEL_PATH
        ),

        "threshold_strategy": (
            "max_validation_f1"
        ),

        "threshold": (
            selected_threshold
        ),

        "calibration_strategy": (
            "none"
        ),

        "primary_metric": (
            "pr_auc"
        ),

        "locked": False,
    }

    save_json(
        FINAL_MODEL_CONFIG_PATH,
        suggested_config,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "VALIDATION PHASE COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Test has NOT been evaluated."
    )

    print(
        "Suggested threshold:",
        f"{selected_threshold:.6f}",
    )


# ============================================================
# LOCK MODEL / THRESHOLD
# ============================================================


def run_lock_config(
    strategy: str,
) -> None:
    if not (
        EVALUATION_RESULTS_PATH.exists()
    ):
        raise FileNotFoundError(
            "Run --validation-only first."
        )

    results = load_json(
        EVALUATION_RESULTS_PATH
    )

    validation = results[
        "validation"
    ]

    if strategy == "max_f1":
        metrics = validation[
            "max_f1"
        ]

        strategy_name = (
            "max_validation_f1"
        )

    elif strategy == "recall_80":
        metrics = validation[
            "recall_80"
        ]

        strategy_name = (
            "validation_recall_at_least_0_80"
        )

    elif strategy == "threshold_0_5":
        metrics = validation[
            "threshold_0_5"
        ]

        strategy_name = (
            "fixed_0_5"
        )

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )

    config = {
        "model": "lightgbm",

        "model_path": str(
            LIGHTGBM_MODEL_PATH
        ),

        "threshold_strategy": (
            strategy_name
        ),

        "threshold": float(
            metrics[
                "threshold"
            ]
        ),

        "calibration_strategy": (
            "none"
        ),

        "primary_metric": (
            "pr_auc"
        ),

        "locked": True,
    }

    save_json(
        FINAL_MODEL_CONFIG_PATH,
        config,
    )

    print(
        "=" * 70
    )

    print(
        "FINAL MODEL CONFIG LOCKED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        json.dumps(
            config,
            indent=2,
        )
    )

    print()
    print(
        "Do not change model or threshold "
        "after Final Test."
    )


# ============================================================
# FINAL TEST
# ============================================================


def run_final_test(
) -> None:
    if not (
        FINAL_MODEL_CONFIG_PATH.exists()
    ):
        raise FileNotFoundError(
            "Final model config not found."
        )

    config = load_json(
        FINAL_MODEL_CONFIG_PATH
    )

    if not config.get(
        "locked",
        False,
    ):
        raise RuntimeError(
            "Model config is not locked."
        )

    if not (
        EVALUATION_RESULTS_PATH.exists()
    ):
        raise FileNotFoundError(
            "Validation results missing."
        )

    existing_results = load_json(
        EVALUATION_RESULTS_PATH
    )

    if existing_results.get(
        "test_evaluated",
        False,
    ):
        raise RuntimeError(
            "Test has already been evaluated. "
            "Refusing to evaluate it again."
        )

    print(
        "=" * 70
    )

    print(
        "KKBOX CHURN — FINAL TEST"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "WARNING: Opening locked Test set."
    )

    joined = (
        load_joined_dataset()
    )

    test_df = joined[
        joined["split"]
        == "test"
    ].copy()

    print(
        "Test rows:",
        f"{len(test_df):,}",
    )

    (
        preprocessor,
        model,
    ) = load_model_artifacts()

    (
        test_ids,
        X_test,
        y_test,
    ) = build_xy(
        test_df
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    test_probabilities = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    probability_metrics = (
        calculate_probability_metrics(
            y_test,
            test_probabilities,
        )
    )

    selected_threshold = float(
        config["threshold"]
    )

    threshold_metrics = (
        calculate_threshold_metrics(
            y_test,
            test_probabilities,
            threshold=(
                selected_threshold
            ),
        )
    )

    test_predictions = pd.DataFrame(
        {
            "msno": (
                test_ids.to_numpy()
            ),

            "is_churn": (
                y_test.to_numpy()
            ),

            "churn_probability": (
                test_probabilities
            ),
        }
    )

    test_predictions = (
        add_predictions_and_error_types(
            test_predictions,

            threshold=(
                selected_threshold
            ),
        )
    )

    TEST_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_predictions.to_parquet(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    existing_results[
        "selected_model"
    ] = config[
        "model"
    ]

    existing_results[
        "selected_threshold"
    ] = selected_threshold

    existing_results[
        "threshold_strategy"
    ] = config[
        "threshold_strategy"
    ]

    existing_results[
        "test_evaluated"
    ] = True

    existing_results[
        "test"
    ] = {
        "probability_metrics": (
            probability_metrics
        ),

        "selected_threshold_metrics": (
            threshold_metrics
        ),
    }

    save_json(
        EVALUATION_RESULTS_PATH,
        existing_results,
    )

    print()
    print(
        "Final Test probability metrics:"
    )

    print(
        json.dumps(
            probability_metrics,
            indent=2,
        )
    )

    print()
    print(
        "Final Test threshold metrics:"
    )

    print(
        json.dumps(
            threshold_metrics,
            indent=2,
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL TEST COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Do NOT tune model or threshold "
        "using Test results."
    )


def main() -> None:
    args = parse_args()

    if args.validation_only:
        run_validation_only()

    elif args.lock_config:
        run_lock_config(
            args.strategy
        )

    elif args.final_test:
        run_final_test()


if __name__ == "__main__":
    main()