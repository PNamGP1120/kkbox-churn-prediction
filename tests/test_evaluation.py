import numpy as np
import pandas as pd
import pytest

from kkbox_churn_prediction.modeling.evaluation import (
    add_predictions_and_error_types,
    build_threshold_table,
    calculate_probability_metrics,
    calculate_threshold_metrics,
    find_best_f1_threshold,
    find_threshold_for_min_recall,
)


def test_probability_metrics_perfect_ranking():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    result = (
        calculate_probability_metrics(
            y_true,
            probabilities,
        )
    )

    assert result[
        "pr_auc"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "roc_auc"
    ] == pytest.approx(
        1.0
    )

    assert (
        0
        <= result["brier_score"]
        <= 1
    )


def test_threshold_metrics_perfect():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.4, 0.6, 0.9]
    )

    result = (
        calculate_threshold_metrics(
            y_true,
            probabilities,
            threshold=0.5,
        )
    )

    assert result["tn"] == 2
    assert result["fp"] == 0

    assert result["fn"] == 0
    assert result["tp"] == 2

    assert result[
        "precision"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "recall"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "f1"
    ] == pytest.approx(
        1.0
    )


def test_threshold_table_not_empty():
    y_true = np.array(
        [0, 0, 0, 1, 1]
    )

    probabilities = np.array(
        [
            0.1,
            0.2,
            0.4,
            0.6,
            0.9,
        ]
    )

    table = (
        build_threshold_table(
            y_true,
            probabilities,
        )
    )

    assert not table.empty

    assert {
        "threshold",
        "precision",
        "recall",
        "f1",
    }.issubset(
        table.columns
    )

    assert table[
        "threshold"
    ].between(
        0,
        1,
    ).all()


def test_best_f1_threshold():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [
            0.1,
            0.2,
            0.7,
            0.9,
        ]
    )

    table = (
        build_threshold_table(
            y_true,
            probabilities,
        )
    )

    result = (
        find_best_f1_threshold(
            table
        )
    )

    assert (
        0
        <= result["threshold"]
        <= 1
    )

    assert (
        0
        <= result["precision"]
        <= 1
    )

    assert (
        0
        <= result["recall"]
        <= 1
    )

    assert (
        0
        <= result["f1"]
        <= 1
    )


def test_recall_constraint():
    y_true = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.15,
            0.45,
            0.55,
            0.75,
            0.95,
        ]
    )

    table = (
        build_threshold_table(
            y_true,
            probabilities,
        )
    )

    result = (
        find_threshold_for_min_recall(
            table,
            min_recall=0.80,
        )
    )

    assert (
        result["recall"]
        >= 0.80
    )


def test_invalid_recall_constraint():
    table = pd.DataFrame(
        {
            "threshold": [0.5],
            "precision": [0.8],
            "recall": [0.7],
            "f1": [0.75],
        }
    )

    with pytest.raises(
        ValueError
    ):
        find_threshold_for_min_recall(
            table,
            min_recall=1.1,
        )


def test_error_types():
    df = pd.DataFrame(
        {
            "is_churn": [
                0,
                0,
                1,
                1,
            ],

            "churn_probability": [
                0.1,
                0.8,
                0.2,
                0.9,
            ],
        }
    )

    result = (
        add_predictions_and_error_types(
            df,
            threshold=0.5,
        )
    )

    assert result[
        "prediction"
    ].tolist() == [
        0,
        1,
        0,
        1,
    ]

    assert result[
        "error_type"
    ].tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]