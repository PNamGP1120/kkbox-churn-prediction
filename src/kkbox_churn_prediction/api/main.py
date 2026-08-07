from __future__ import annotations

import logging
from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)

from kkbox_churn_prediction.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
)

from kkbox_churn_prediction.api.service import (
    ChurnPredictionService,
    FeatureContractError,
    FeatureValueError,
)


logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(
    __name__
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    logger.info(
        "Loading frozen model artifacts..."
    )

    service = (
        ChurnPredictionService()
    )

    app.state.prediction_service = (
        service
    )

    logger.info(
        "Model ready: %s, threshold=%.6f",
        service.model_name,
        service.threshold,
    )

    yield

    logger.info(
        "Shutting down prediction service."
    )

    app.state.prediction_service = (
        None
    )


app = FastAPI(
    title=(
        "KKBOX Customer Churn "
        "Prediction API"
    ),

    description=(
        "Inference API for the frozen "
        "LightGBM churn model selected "
        "in QT5."
    ),

    version="1.0.0",

    lifespan=lifespan,
)


def get_service(
    request: Request,
) -> ChurnPredictionService:
    service = getattr(
        request.app.state,
        "prediction_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),

            detail=(
                "Prediction service "
                "is not ready."
            ),
        )

    return service


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> dict[
    str,
    str,
]:
    """
    Liveness endpoint.

    Confirms that the FastAPI process
    is alive.
    """

    return {
        "status": "ok"
    }


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
)
def ready(
    service: ChurnPredictionService = (
        Depends(
            get_service
        )
    ),
) -> dict[
    str,
    str,
]:
    """
    Readiness endpoint.

    Returns ready only after frozen
    model artifacts have loaded.
    """

    _ = service

    return {
        "status": "ready"
    }


@app.get(
    "/metadata",
    response_model=(
        ModelMetadataResponse
    ),
    tags=["Model"],
)
def metadata(
    service: ChurnPredictionService = (
        Depends(
            get_service
        )
    ),
) -> dict:
    """
    Return safe deployment metadata.

    Local artifact filesystem paths
    are intentionally not exposed.
    """

    return service.metadata()


@app.post(
    "/predict",
    response_model=(
        PredictionResponse
    ),
    tags=["Prediction"],
)
def predict(
    payload: PredictionRequest,

    service: ChurnPredictionService = (
        Depends(
            get_service
        )
    ),
) -> dict:
    """
    Predict churn for one user.
    """

    try:
        return service.predict(
            msno=payload.msno,
            features=(
                payload.features
            ),
        )

    except (
        FeatureContractError,
        FeatureValueError,
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected inference failure."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Prediction failed."
            ),
        ) from exc


@app.post(
    "/predict/batch",
    response_model=(
        BatchPredictionResponse
    ),
    tags=["Prediction"],
)
def predict_batch(
    payload: BatchPredictionRequest,

    service: ChurnPredictionService = (
        Depends(
            get_service
        )
    ),
) -> dict:
    """
    Vectorized batch inference.

    Maximum batch size is currently
    limited by the Pydantic schema.
    """

    records = [
        (
            record.msno,
            record.features,
        )
        for record
        in payload.records
    ]

    try:
        predictions = (
            service.predict_batch(
                records=records
            )
        )

        return {
            "predictions":
                predictions
        }

    except (
        FeatureContractError,
        FeatureValueError,
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected batch "
            "inference failure."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Batch prediction failed."
            ),
        ) from exc