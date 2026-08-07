from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


FeatureValue = (
    int
    | float
    | str
    | None
)


class PredictionRequest(BaseModel):
    """
    One churn prediction request.

    The API currently expects the 40 engineered
    features produced by QT3.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    msno: str | None = Field(
        default=None,
        min_length=1,
    )

    features: dict[
        str,
        FeatureValue,
    ] = Field(
        min_length=1
    )


class PredictionResponse(BaseModel):
    msno: str | None

    model: str

    churn_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    prediction: Literal[0, 1]

    threshold_strategy: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    records: list[
        PredictionRequest
    ] = Field(
        min_length=1,
        max_length=1000,
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[
        PredictionResponse
    ]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str


class ModelMetadataResponse(BaseModel):
    model: str

    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    threshold_strategy: str

    calibration_strategy: str

    primary_metric: str

    raw_feature_count: int

    locked: bool