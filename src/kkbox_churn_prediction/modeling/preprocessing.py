from sklearn.compose import (
    ColumnTransformer,
)

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "registered_via",
]


def get_numeric_features(
    feature_columns: list[str],
) -> list[str]:
    return [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]


def build_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    numeric_features = (
        get_numeric_features(
            feature_columns
        )
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",

                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",

                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                ),
            ),

            (
                "encoder",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],

        sparse_threshold=1.0,
    )