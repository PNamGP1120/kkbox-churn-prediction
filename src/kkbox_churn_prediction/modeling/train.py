from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_dummy_classifier():
    return DummyClassifier(
        strategy="prior",
    )


def build_logistic_regression(
    random_state: int = 42,
):
    return LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=300,
        random_state=random_state,
    )


def build_random_forest(
    random_state: int = 42,
):
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def build_lightgbm(
    random_state: int = 42,
):
    return LGBMClassifier(
        objective="binary",

        n_estimators=1000,
        learning_rate=0.05,

        num_leaves=31,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=0.1,

        metric="average_precision",
        first_metric_only=True,

        n_jobs=-1,
        random_state=random_state,

        verbosity=-1,
    )