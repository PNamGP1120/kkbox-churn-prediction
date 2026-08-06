from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def evaluate_probability_metrics(
    model,
    X,
    y,
) -> dict[str, float]:
    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    return {
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
    }