import pytest

from kkbox_churn_prediction.api.service import (
    CATEGORICAL_FEATURES,
    ChurnPredictionService,
    FeatureContractError,
    FeatureValueError,
)


@pytest.fixture(
    scope="module"
)
def service():
    return (
        ChurnPredictionService()
    )


def make_valid_features(
    service: ChurnPredictionService,
) -> dict:
    """
    Create a syntactically valid
    40-feature vector.

    Numeric features use 0.0.
    Categoricals use None so the frozen
    categorical imputer handles them.
    """

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

    return features


def test_service_loads_locked_model(
    service,
):
    assert (
        service.config[
            "locked"
        ]
        is True
    )

    assert (
        service.model_name
        == "lightgbm"
    )

    assert len(
        service.expected_features
    ) == 40

    assert (
        0.0
        <= service.threshold
        <= 1.0
    )


def test_prediction_is_deterministic(
    service,
):
    features = (
        make_valid_features(
            service
        )
    )

    first = service.predict(
        msno="test-user",
        features=features,
    )

    second = service.predict(
        msno="test-user",
        features=features,
    )

    assert first[
        "prediction"
    ] == second[
        "prediction"
    ]

    assert first[
        "churn_score"
    ] == pytest.approx(
        second[
            "churn_score"
        ]
    )


def test_prediction_score_and_class(
    service,
):
    features = (
        make_valid_features(
            service
        )
    )

    result = service.predict(
        msno="test-user",
        features=features,
    )

    assert (
        0.0
        <= result[
            "churn_score"
        ]
        <= 1.0
    )

    assert result[
        "prediction"
    ] in {
        0,
        1,
    }

    expected_prediction = int(
        result[
            "churn_score"
        ]
        >= service.threshold
    )

    assert (
        result[
            "prediction"
        ]
        == expected_prediction
    )


def test_missing_feature_rejected(
    service,
):
    features = (
        make_valid_features(
            service
        )
    )

    removed_feature = (
        service
        .expected_features[0]
    )

    features.pop(
        removed_feature
    )

    with pytest.raises(
        FeatureContractError
    ):
        service.predict(
            msno="test-user",
            features=features,
        )


def test_unexpected_feature_rejected(
    service,
):
    features = (
        make_valid_features(
            service
        )
    )

    features[
        "fake_feature"
    ] = 123

    with pytest.raises(
        FeatureContractError
    ):
        service.predict(
            msno="test-user",
            features=features,
        )


def test_invalid_numeric_value_rejected(
    service,
):
    features = (
        make_valid_features(
            service
        )
    )

    numeric_feature = next(
        feature
        for feature
        in service.expected_features
        if feature
        not in CATEGORICAL_FEATURES
    )

    features[
        numeric_feature
    ] = "not-a-number"

    with pytest.raises(
        FeatureValueError
    ):
        service.predict(
            msno="test-user",
            features=features,
        )


def test_batch_prediction(
    service,
):
    features_1 = (
        make_valid_features(
            service
        )
    )

    features_2 = (
        make_valid_features(
            service
        )
    )

    results = (
        service.predict_batch(
            records=[
                (
                    "user-1",
                    features_1,
                ),
                (
                    "user-2",
                    features_2,
                ),
            ]
        )
    )

    assert len(
        results
    ) == 2

    assert results[
        0
    ][
        "msno"
    ] == "user-1"

    assert results[
        1
    ][
        "msno"
    ] == "user-2"