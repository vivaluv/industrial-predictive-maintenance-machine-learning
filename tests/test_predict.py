import pytest

from src.config import DECISION_THRESHOLD

from src.data_preprocessing import (
    load_data,
    prepare_model_data,
    split_data,
)

from src.train_model import (
    build_tuned_xgboost_pipeline,
    train_model,
)

from src.predict import (
    predict_failure,
    predict_probability,
    predict_risk_level,
    predict_with_explanation,
)


@pytest.fixture(scope="module")
def trained_model_and_data():
    """
    Train the tuned XGBoost model once and reuse it
    across prediction tests.
    """

    df = load_data()

    X, y = prepare_model_data(df)

    X_train, X_test, y_train, _ = split_data(
        X,
        y,
    )

    model = build_tuned_xgboost_pipeline()

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    return trained_model, X_test


def test_predict_failure(
    trained_model_and_data,
):
    """
    Test binary class predictions.
    """

    model, X_test = (
        trained_model_and_data
    )

    X_sample = X_test.head(10)

    predictions = predict_failure(
        model,
        X_sample,
    )

    assert len(predictions) == 10

    assert set(
        predictions
    ).issubset(
        {0, 1}
    )


def test_predict_probability(
    trained_model_and_data,
):
    """
    Test machine failure probability predictions.
    """

    model, X_test = (
        trained_model_and_data
    )

    X_sample = X_test.head(10)

    probabilities = predict_probability(
        model,
        X_sample,
    )

    assert len(probabilities) == 10

    for probability in probabilities:

        assert 0 <= probability <= 1


def test_predict_risk_level(
    trained_model_and_data,
):
    """
    Test threshold-based operational decisions.
    """

    model, X_test = (
        trained_model_and_data
    )

    X_sample = X_test.head(10)

    predictions = predict_risk_level(
        model,
        X_sample,
        threshold=DECISION_THRESHOLD,
    )

    assert len(predictions) == 10

    assert set(
        predictions
    ).issubset(
        {0, 1}
    )


def test_predict_with_explanation(
    trained_model_and_data,
):
    """
    Test prediction output with local SHAP
    explanations.
    """

    model, X_test = (
        trained_model_and_data
    )

    X_sample = X_test.head(3)

    results = predict_with_explanation(
        model,
        X_sample,
        threshold=DECISION_THRESHOLD,
        top_n=3,
    )

    expected_columns = {
        "predicted_class",
        "failure_probability",
        "decision_threshold",
        "decision_prediction",
        "top_contributors",
    }

    assert len(results) == 3

    assert expected_columns.issubset(
        results.columns
    )

    assert (
        results[
            "decision_threshold"
        ]
        == DECISION_THRESHOLD
    ).all()

    assert (
        results[
            "failure_probability"
        ]
        .between(
            0,
            1
        )
        .all()
    )

    assert set(
        results[
            "predicted_class"
        ]
    ).issubset(
        {0, 1}
    )

    assert set(
        results[
            "decision_prediction"
        ]
    ).issubset(
        {0, 1}
    )

    for contributors in results[
        "top_contributors"
    ]:

        assert len(contributors) == 3