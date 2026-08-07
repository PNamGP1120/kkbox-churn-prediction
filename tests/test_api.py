import pytest

from fastapi.testclient import (
    TestClient,
)

from kkbox_churn_prediction.api.main import (
    app,
)

from kkbox_churn_prediction.api.service import (
    CATEGORICAL_FEATURES,
)


@pytest.fixture(
    scope="module"
)
def client():
    with TestClient(
        app
    ) as test_client:
        yield test_client


def make_valid_payload():
    service = (
        app.state
        .prediction_service
    )

    features = {}

    for feature in (
        service.expected_features
    ):
        if (
            feature
            in CATEGORICAL_FEATURES
        ):
            features[
                feature
            ] = None

        else:
            features[
                feature
            ] = 0.0

    return {
        "msno": "api-test-user",
        "features": features,
    }


def test_health(
    client,
):
    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    assert response.json() == {
        "status": "ok"
    }


def test_ready(
    client,
):
    response = client.get(
        "/ready"
    )

    assert (
        response.status_code
        == 200
    )

    assert response.json() == {
        "status": "ready"
    }


def test_metadata(
    client,
):
    response = client.get(
        "/metadata"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert data[
        "model"
    ] == "lightgbm"

    assert data[
        "locked"
    ] is True

    assert data[
        "raw_feature_count"
    ] == 40

    assert (
        0
        <= data[
            "threshold"
        ]
        <= 1
    )


def test_predict_valid_request(
    client,
):
    payload = (
        make_valid_payload()
    )

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert data[
        "msno"
    ] == "api-test-user"

    assert data[
        "model"
    ] == "lightgbm"

    assert (
        0
        <= data[
            "churn_score"
        ]
        <= 1
    )

    assert data[
        "prediction"
    ] in {
        0,
        1,
    }


def test_predict_missing_feature(
    client,
):
    payload = (
        make_valid_payload()
    )

    service = (
        app.state
        .prediction_service
    )

    feature = (
        service
        .expected_features[0]
    )

    payload[
        "features"
    ].pop(
        feature
    )

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 400
    )

    assert (
        "Missing features"
        in response.json()[
            "detail"
        ]
    )


def test_predict_unexpected_feature(
    client,
):
    payload = (
        make_valid_payload()
    )

    payload[
        "features"
    ][
        "fake_feature"
    ] = 100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 400
    )

    assert (
        "Unexpected features"
        in response.json()[
            "detail"
        ]
    )


def test_predict_invalid_numeric(
    client,
):
    payload = (
        make_valid_payload()
    )

    service = (
        app.state
        .prediction_service
    )

    numeric_feature = next(
        feature
        for feature
        in service.expected_features
        if feature
        not in CATEGORICAL_FEATURES
    )

    payload[
        "features"
    ][
        numeric_feature
    ] = "invalid-number"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 400
    )


def test_batch_prediction(
    client,
):
    record_1 = (
        make_valid_payload()
    )

    record_2 = (
        make_valid_payload()
    )

    record_1[
        "msno"
    ] = "batch-user-1"

    record_2[
        "msno"
    ] = "batch-user-2"

    response = client.post(
        "/predict/batch",
        json={
            "records": [
                record_1,
                record_2,
            ]
        },
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert len(
        data[
            "predictions"
        ]
    ) == 2

    assert (
        data[
            "predictions"
        ][0][
            "msno"
        ]
        == "batch-user-1"
    )

    assert (
        data[
            "predictions"
        ][1][
            "msno"
        ]
        == "batch-user-2"
    )