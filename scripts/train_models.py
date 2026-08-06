from __future__ import annotations

import json
from typing import Any

import joblib
import lightgbm as lgb

from kkbox_churn_prediction.config import (
    DUMMY_MODEL_PATH,
    FEATURE_NAMES_PATH,
    LIGHTGBM_MODEL_PATH,
    LOGISTIC_MODEL_PATH,
    MODELS_DIR,
    MODEL_SPLITS_PATH,
    PREPROCESSOR_PATH,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE,
    TRAIN_FEATURES_PATH,
    TRAINING_METADATA_PATH,
    TRAINING_RESULTS_PATH,
)

from kkbox_churn_prediction.modeling.data import (
    load_modeling_data,
    save_split_assignments,
    split_dataset,
    split_xy,
)

from kkbox_churn_prediction.modeling.metrics import (
    evaluate_probability_metrics,
)

from kkbox_churn_prediction.modeling.preprocessing import (
    build_preprocessor,
)

from kkbox_churn_prediction.modeling.train import (
    build_dummy_classifier,
    build_lightgbm,
    build_logistic_regression,
    build_random_forest,
)


def print_split_summary(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) -> None:
    """Print information about train/validation/test splits."""

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(f"Train rows:      {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")

    print()

    print("Churn rates:")

    print(
        f"Train:      "
        f"{y_train.mean():.4f} "
        f"({y_train.mean() * 100:.2f}%)"
    )

    print(
        f"Validation: "
        f"{y_val.mean():.4f} "
        f"({y_val.mean() * 100:.2f}%)"
    )

    print(
        f"Test:       "
        f"{y_test.mean():.4f} "
        f"({y_test.mean() * 100:.2f}%)"
    )


def print_model_result(
    model_name: str,
    train_scores: dict[str, float],
    validation_scores: dict[str, float],
) -> None:
    """Print train and validation metrics for one model."""

    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    print(
        f"Train PR-AUC:      "
        f"{train_scores['pr_auc']:.4f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{validation_scores['pr_auc']:.4f}"
    )

    print(
        f"Train ROC-AUC:     "
        f"{train_scores['roc_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC:"
        f" {validation_scores['roc_auc']:.4f}"
    )

    pr_gap = (
        train_scores["pr_auc"]
        - validation_scores["pr_auc"]
    )

    roc_gap = (
        train_scores["roc_auc"]
        - validation_scores["roc_auc"]
    )

    print()
    print(f"PR-AUC gap:  {pr_gap:.4f}")
    print(f"ROC-AUC gap: {roc_gap:.4f}")


def train_and_evaluate_model(
    *,
    model_name: str,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_path,
) -> dict[str, Any]:
    """
    Train a standard sklearn-compatible classifier,
    evaluate it on train and validation sets,
    and save the trained model.
    """

    print()
    print(f"Training {model_name}...")

    model.fit(
        X_train,
        y_train,
    )

    train_scores = evaluate_probability_metrics(
        model,
        X_train,
        y_train,
    )

    validation_scores = evaluate_probability_metrics(
        model,
        X_val,
        y_val,
    )

    joblib.dump(
        model,
        model_path,
    )

    print_model_result(
        model_name,
        train_scores,
        validation_scores,
    )

    return {
        "train": train_scores,
        "validation": validation_scores,
    }


def main() -> None:
    # ---------------------------------------------------------
    # 0. Prepare output directories
    # ---------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_SPLITS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("KKBOX CHURN — MODEL TRAINING")
    print("=" * 70)

    print()
    print(f"Dataset: {TRAIN_FEATURES_PATH}")
    print(f"Random state: {RANDOM_STATE}")

    # ---------------------------------------------------------
    # 1. Load processed feature dataset
    # ---------------------------------------------------------

    print()
    print("1. Loading processed dataset...")

    df = load_modeling_data(
        TRAIN_FEATURES_PATH
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "Total users:",
        f"{len(df):,}",
    )

    print(
        "Overall churn rate:",
        f"{df['is_churn'].mean():.4f}",
    )

    # ---------------------------------------------------------
    # 2. Create Train / Validation / Test split
    # ---------------------------------------------------------

    print()
    print(
        "2. Creating "
        "train / validation / test split..."
    )

    (
        train_df,
        val_df,
        test_df,
    ) = split_dataset(
        df,
        RANDOM_STATE,
    )

    # Save split membership by msno.
    #
    # QT5 must reuse exactly the same test set.
    save_split_assignments(
        train_df,
        val_df,
        test_df,
        MODEL_SPLITS_PATH,
    )

    (
        train_ids,
        X_train,
        y_train,
    ) = split_xy(train_df)

    (
        val_ids,
        X_val,
        y_val,
    ) = split_xy(val_df)

    (
        test_ids,
        X_test,
        y_test,
    ) = split_xy(test_df)

    print_split_summary(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    # Defensive checks.
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    assert "msno" not in X_train.columns
    assert "is_churn" not in X_train.columns

    # ---------------------------------------------------------
    # 3. Build and fit preprocessing pipeline
    # ---------------------------------------------------------

    print()
    print(
        "3. Building preprocessing pipeline..."
    )

    preprocessor = build_preprocessor(
        X_train.columns.tolist()
    )

    print(
        "Fitting preprocessing on TRAIN only..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        "Transforming validation data..."
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    # IMPORTANT:
    # Test data is deliberately NOT transformed
    # for evaluation in QT4.
    #
    # QT5 will load the saved preprocessor
    # and transform the locked test set.

    print()
    print(
        "Processed train shape:",
        X_train_processed.shape,
    )

    print(
        "Processed validation shape:",
        X_val_processed.shape,
    )

    # ---------------------------------------------------------
    # 4. Save preprocessing pipeline
    # ---------------------------------------------------------

    print()
    print("4. Saving preprocessing pipeline...")

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    with open(
        FEATURE_NAMES_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_names.tolist(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Raw feature count:",
        len(X_train.columns),
    )

    print(
        "Processed feature count:",
        len(feature_names),
    )

    # ---------------------------------------------------------
    # 5. Prepare result dictionary
    # ---------------------------------------------------------

    results: dict[str, Any] = {}

    # ---------------------------------------------------------
    # 6. Dummy baseline
    # ---------------------------------------------------------

    dummy_model = build_dummy_classifier()

    results["dummy"] = (
        train_and_evaluate_model(
            model_name="DummyClassifier",
            model=dummy_model,
            X_train=X_train_processed,
            y_train=y_train,
            X_val=X_val_processed,
            y_val=y_val,
            model_path=DUMMY_MODEL_PATH,
        )
    )

    # ---------------------------------------------------------
    # 7. Logistic Regression
    # ---------------------------------------------------------

    logistic_model = (
        build_logistic_regression(
            RANDOM_STATE
        )
    )

    results["logistic_regression"] = (
        train_and_evaluate_model(
            model_name="Logistic Regression",
            model=logistic_model,
            X_train=X_train_processed,
            y_train=y_train,
            X_val=X_val_processed,
            y_val=y_val,
            model_path=LOGISTIC_MODEL_PATH,
        )
    )

    # ---------------------------------------------------------
    # 8. Random Forest
    # ---------------------------------------------------------

    random_forest_model = (
        build_random_forest(
            RANDOM_STATE
        )
    )

    results["random_forest"] = (
        train_and_evaluate_model(
            model_name="Random Forest",
            model=random_forest_model,
            X_train=X_train_processed,
            y_train=y_train,
            X_val=X_val_processed,
            y_val=y_val,
            model_path=RANDOM_FOREST_MODEL_PATH,
        )
    )

    # ---------------------------------------------------------
    # 9. LightGBM
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Training LightGBM...")
    print("=" * 70)

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print(
        "Negative samples:",
        f"{negative_count:,}",
    )

    print(
        "Positive samples:",
        f"{positive_count:,}",
    )

    print(
        "scale_pos_weight:",
        f"{scale_pos_weight:.4f}",
    )

    lightgbm_model = build_lightgbm(
        RANDOM_STATE
    )

    lightgbm_model.set_params(
        scale_pos_weight=scale_pos_weight
    )

    lightgbm_model.fit(
        X_train_processed,
        y_train,

        eval_X=X_val_processed,
        eval_y=y_val,

        eval_metric="average_precision",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50
            ),
            lgb.log_evaluation(
                period=50
            ),
        ],
    )

    lightgbm_train_scores = (
        evaluate_probability_metrics(
            lightgbm_model,
            X_train_processed,
            y_train,
        )
    )

    lightgbm_validation_scores = (
        evaluate_probability_metrics(
            lightgbm_model,
            X_val_processed,
            y_val,
        )
    )

    best_iteration = (
        lightgbm_model.best_iteration_
    )

    results["lightgbm"] = {
        "train": lightgbm_train_scores,
        "validation": (
            lightgbm_validation_scores
        ),
        "best_iteration": (
            int(best_iteration)
            if best_iteration is not None
            else None
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
    }

    joblib.dump(
        lightgbm_model,
        LIGHTGBM_MODEL_PATH,
    )

    print_model_result(
        "LightGBM",
        lightgbm_train_scores,
        lightgbm_validation_scores,
    )

    print()
    print(
        "Best LightGBM iteration:",
        best_iteration,
    )

    # ---------------------------------------------------------
    # 10. Save validation results
    # ---------------------------------------------------------

    print()
    print("10. Saving validation results...")

    with open(
        TRAINING_RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # 11. Save training metadata
    # ---------------------------------------------------------

    metadata = {
        "random_state": RANDOM_STATE,

        "source_dataset": str(
            TRAIN_FEATURES_PATH
        ),

        "total_rows": int(
            len(df)
        ),

        "train_rows": int(
            len(X_train)
        ),

        "validation_rows": int(
            len(X_val)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "train_churn_rate": float(
            y_train.mean()
        ),

        "validation_churn_rate": float(
            y_val.mean()
        ),

        "test_churn_rate": float(
            y_test.mean()
        ),

        "raw_feature_count": int(
            len(X_train.columns)
        ),

        "processed_feature_count": int(
            len(feature_names)
        ),

        "negative_train_samples": (
            negative_count
        ),

        "positive_train_samples": (
            positive_count
        ),

        "scale_pos_weight": float(
            scale_pos_weight
        ),

        "test_evaluated": False,

        "primary_validation_metric": (
            "pr_auc"
        ),

        "secondary_validation_metric": (
            "roc_auc"
        ),
    }

    with open(
        TRAINING_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # 12. Print final comparison
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 70)

    sorted_results = sorted(
        results.items(),
        key=lambda item: (
            item[1]["validation"]["pr_auc"]
        ),
        reverse=True,
    )

    print()
    print(
        f"{'Model':<25}"
        f"{'PR-AUC':>12}"
        f"{'ROC-AUC':>12}"
    )

    print("-" * 49)

    for model_name, model_result in (
        sorted_results
    ):
        validation = (
            model_result["validation"]
        )

        print(
            f"{model_name:<25}"
            f"{validation['pr_auc']:>12.4f}"
            f"{validation['roc_auc']:>12.4f}"
        )

    best_model_name = (
        sorted_results[0][0]
    )

    best_model_score = (
        sorted_results[0][1]
        ["validation"]
        ["pr_auc"]
    )

    print()
    print(
        "Best validation model by PR-AUC:",
        best_model_name,
    )

    print(
        "Best validation PR-AUC:",
        f"{best_model_score:.4f}",
    )

    # ---------------------------------------------------------
    # 13. Final artifact summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Preprocessor:       "
        f"{PREPROCESSOR_PATH}"
    )

    print(
        f"Feature names:      "
        f"{FEATURE_NAMES_PATH}"
    )

    print(
        f"Dummy model:        "
        f"{DUMMY_MODEL_PATH}"
    )

    print(
        f"Logistic model:     "
        f"{LOGISTIC_MODEL_PATH}"
    )

    print(
        f"Random Forest:      "
        f"{RANDOM_FOREST_MODEL_PATH}"
    )

    print(
        f"LightGBM:           "
        f"{LIGHTGBM_MODEL_PATH}"
    )

    print(
        f"Validation results: "
        f"{TRAINING_RESULTS_PATH}"
    )

    print(
        f"Training metadata:  "
        f"{TRAINING_METADATA_PATH}"
    )

    print(
        f"Split assignments:  "
        f"{MODEL_SPLITS_PATH}"
    )

    print()
    print(
        "IMPORTANT: Test set has not been "
        "evaluated."
    )

    print(
        "Final test evaluation belongs to QT5."
    )


if __name__ == "__main__":
    main()