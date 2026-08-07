from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from kkbox_churn_prediction.config import (
    FINAL_MODEL_CONFIG_PATH,
    LIGHTGBM_MODEL_PATH,
    PREPROCESSOR_PATH,
)


CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "registered_via",
]


class ModelArtifactError(
    RuntimeError
):
    """
    Raised when frozen model artifacts
    are missing or inconsistent.
    """


class FeatureContractError(
    ValueError
):
    """
    Raised when request feature names
    do not match the trained model contract.
    """


class FeatureValueError(
    ValueError
):
    """
    Raised when a provided feature value
    cannot be converted to the expected type.
    """


class ChurnPredictionService:
    """
    Frozen LightGBM inference service.

    The service loads:
    - QT4 preprocessor
    - QT4 LightGBM model
    - QT5 locked threshold configuration

    No model training or tuning happens here.
    """

    def __init__(
        self,
        preprocessor_path: Path = (
            PREPROCESSOR_PATH
        ),
        model_path: Path = (
            LIGHTGBM_MODEL_PATH
        ),
        config_path: Path = (
            FINAL_MODEL_CONFIG_PATH
        ),
    ) -> None:
        self.preprocessor_path = (
            Path(
                preprocessor_path
            )
        )

        self.model_path = Path(
            model_path
        )

        self.config_path = Path(
            config_path
        )

        self._validate_artifact_paths()

        self.config = (
            self._load_config()
        )

        self._validate_config()

        self.preprocessor = (
            joblib.load(
                self.preprocessor_path
            )
        )

        self.model = joblib.load(
            self.model_path
        )

        self.expected_features = (
            self._load_expected_features()
        )

        self.numeric_features = [
            feature
            for feature
            in self.expected_features
            if feature
            not in CATEGORICAL_FEATURES
        ]

        self.threshold = float(
            self.config["threshold"]
        )

        self.model_name = str(
            self.config["model"]
        )

        self.threshold_strategy = str(
            self.config[
                "threshold_strategy"
            ]
        )

    def _validate_artifact_paths(
        self,
    ) -> None:
        required_paths = {
            "preprocessor":
                self.preprocessor_path,

            "model":
                self.model_path,

            "final config":
                self.config_path,
        }

        missing = [
            f"{name}: {path}"
            for name, path
            in required_paths.items()
            if not path.exists()
        ]

        if missing:
            raise ModelArtifactError(
                "Missing deployment artifacts: "
                + "; ".join(
                    missing
                )
            )

    def _load_config(
        self,
    ) -> dict[str, Any]:
        with self.config_path.open(
            encoding="utf-8",
        ) as file:
            return json.load(
                file
            )

    def _validate_config(
        self,
    ) -> None:
        required_keys = {
            "model",
            "threshold_strategy",
            "threshold",
            "calibration_strategy",
            "primary_metric",
            "locked",
        }

        missing_keys = (
            required_keys
            - self.config.keys()
        )

        if missing_keys:
            raise ModelArtifactError(
                "Final model config is "
                "missing keys: "
                + ", ".join(
                    sorted(
                        missing_keys
                    )
                )
            )

        if (
            self.config["locked"]
            is not True
        ):
            raise ModelArtifactError(
                "Final model configuration "
                "is not locked."
            )

        if (
            self.config["model"]
            != "lightgbm"
        ):
            raise ModelArtifactError(
                "Deployment currently expects "
                "the locked LightGBM model."
            )

        threshold = float(
            self.config[
                "threshold"
            ]
        )

        if not (
            0.0
            <= threshold
            <= 1.0
        ):
            raise ModelArtifactError(
                "Threshold must be "
                "between 0 and 1."
            )

    def _load_expected_features(
        self,
    ) -> list[str]:
        feature_names = getattr(
            self.preprocessor,
            "feature_names_in_",
            None,
        )

        if feature_names is None:
            raise ModelArtifactError(
                "Frozen preprocessor does "
                "not contain feature_names_in_."
            )

        features = [
            str(feature)
            for feature
            in feature_names.tolist()
        ]

        if len(features) != 40:
            raise ModelArtifactError(
                "Expected 40 raw model "
                "features, but frozen "
                f"preprocessor contains "
                f"{len(features)}."
            )

        missing_categorical = (
            set(
                CATEGORICAL_FEATURES
            )
            - set(features)
        )

        if missing_categorical:
            raise ModelArtifactError(
                "Frozen feature contract "
                "is missing categorical "
                "features: "
                + ", ".join(
                    sorted(
                        missing_categorical
                    )
                )
            )

        return features

    def _validate_feature_keys(
        self,
        features: Mapping[
            str,
            object,
        ],
    ) -> None:
        provided = set(
            features.keys()
        )

        expected = set(
            self.expected_features
        )

        missing = sorted(
            expected - provided
        )

        unexpected = sorted(
            provided - expected
        )

        messages: list[str] = []

        if missing:
            messages.append(
                "Missing features: "
                + ", ".join(
                    missing
                )
            )

        if unexpected:
            messages.append(
                "Unexpected features: "
                + ", ".join(
                    unexpected
                )
            )

        if messages:
            raise FeatureContractError(
                "; ".join(
                    messages
                )
            )

    def _prepare_frame(
        self,
        records: Sequence[
            Mapping[str, object]
        ],
    ) -> pd.DataFrame:
        if not records:
            raise FeatureContractError(
                "At least one prediction "
                "record is required."
            )

        for record in records:
            self._validate_feature_keys(
                record
            )

        # Explicit columns guarantee exactly
        # the same feature order used in QT4.
        frame = pd.DataFrame(
            records,
            columns=(
                self.expected_features
            ),
        )

        # Match QT4 categorical preparation.
        for column in (
            CATEGORICAL_FEATURES
        ):
            series = (
                frame[column]
                .astype("string")
            )

            frame[column] = (
                series
                .astype(object)
                .where(
                    series.notna(),
                    np.nan,
                )
            )

        # Ensure numeric features cannot silently
        # contain arbitrary string values.
        for column in (
            self.numeric_features
        ):
            try:
                frame[column] = (
                    pd.to_numeric(
                        frame[column],
                        errors="raise",
                    )
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise FeatureValueError(
                    "Invalid numeric value "
                    f"for feature '{column}'."
                ) from exc

        return frame

    def _predict_frame(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        processed = (
            self.preprocessor
            .transform(
                frame
            )
        )

        probabilities = (
            self.model
            .predict_proba(
                processed
            )
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[1]
            < 2
        ):
            raise RuntimeError(
                "Model predict_proba() "
                "returned an invalid shape."
            )

        scores = np.asarray(
            probabilities[:, 1],
            dtype=float,
        )

        if not np.isfinite(
            scores
        ).all():
            raise RuntimeError(
                "Model returned non-finite "
                "prediction scores."
            )

        return scores

    def predict(
        self,
        *,
        msno: str | None,
        features: Mapping[
            str,
            object,
        ],
    ) -> dict[str, Any]:
        results = self.predict_batch(
            records=[
                (
                    msno,
                    features,
                )
            ]
        )

        return results[0]

    def predict_batch(
        self,
        *,
        records: Sequence[
            tuple[
                str | None,
                Mapping[
                    str,
                    object,
                ],
            ]
        ],
    ) -> list[
        dict[str, Any]
    ]:
        if not records:
            raise FeatureContractError(
                "At least one prediction "
                "record is required."
            )

        ids = [
            msno
            for msno, _
            in records
        ]

        feature_records = [
            features
            for _, features
            in records
        ]

        frame = self._prepare_frame(
            feature_records
        )

        scores = self._predict_frame(
            frame
        )

        results: list[
            dict[str, Any]
        ] = []

        for msno, score in zip(
            ids,
            scores,
            strict=True,
        ):
            numeric_score = float(
                score
            )

            prediction = int(
                numeric_score
                >= self.threshold
            )

            results.append(
                {
                    "msno": msno,

                    "model":
                        self.model_name,

                    "churn_score":
                        numeric_score,

                    "threshold":
                        self.threshold,

                    "prediction":
                        prediction,

                    "threshold_strategy":
                        self.threshold_strategy,
                }
            )

        return results

    def metadata(
        self,
    ) -> dict[str, Any]:
        return {
            "model":
                self.model_name,

            "threshold":
                self.threshold,

            "threshold_strategy":
                self.threshold_strategy,

            "calibration_strategy":
                self.config[
                    "calibration_strategy"
                ],

            "primary_metric":
                self.config[
                    "primary_metric"
                ],

            "raw_feature_count":
                len(
                    self.expected_features
                ),

            "locked":
                bool(
                    self.config[
                        "locked"
                    ]
                ),
        }