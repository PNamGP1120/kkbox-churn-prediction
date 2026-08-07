FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest \
    /uv \
    /uvx \
    /bin/

WORKDIR /app


# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1

ENV PYTHONPATH=/app/src


# ------------------------------------------------------------
# LightGBM runtime dependency
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y \
        --no-install-recommends \
        libgomp1 \
    && rm -rf \
        /var/lib/apt/lists/*


# ------------------------------------------------------------
# Install Python dependencies first
# for better Docker layer caching
# ------------------------------------------------------------

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project


# ------------------------------------------------------------
# Application source
# ------------------------------------------------------------

COPY src ./src


# ------------------------------------------------------------
# Frozen ML deployment artifacts
# ------------------------------------------------------------

COPY models/preprocessor.joblib \
    ./models/preprocessor.joblib

COPY models/lightgbm.joblib \
    ./models/lightgbm.joblib

COPY models/final_model_config.json \
    ./models/final_model_config.json


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

EXPOSE 8000


# ------------------------------------------------------------
# Container health check
# ------------------------------------------------------------

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD [ \
        "/app/.venv/bin/python", \
        "-c", \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" \
    ]

CMD ["/app/.venv/bin/uvicorn", "kkbox_churn_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]