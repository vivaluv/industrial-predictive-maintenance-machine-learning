from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_preprocessing import (
    load_data,
    prepare_model_data,
    split_data,
)

from src.train_model import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    build_tuned_xgboost_pipeline,
    train_model,
)


def test_build_logistic_regression():
    """
    Test construction of the Logistic Regression model.
    """

    model = build_logistic_regression()

    assert model is not None
    assert isinstance(
        model,
        LogisticRegression,
    )


def test_build_random_forest():
    """
    Test construction of the Random Forest model.
    """

    model = build_random_forest()

    assert model is not None
    assert isinstance(
        model,
        RandomForestClassifier,
    )


def test_build_xgboost():
    """
    Test construction of the XGBoost model.
    """

    model = build_xgboost()

    assert model is not None
    assert isinstance(
        model,
        XGBClassifier,
    )


def test_build_tuned_xgboost_pipeline():
    """
    Test construction of the tuned XGBoost pipeline.
    """

    model = build_tuned_xgboost_pipeline()

    assert model is not None
    assert isinstance(
        model,
        Pipeline,
    )


def test_train_tuned_xgboost():
    """
    Test training and prediction using the
    tuned XGBoost pipeline.
    """

    df = load_data()

    X, y = prepare_model_data(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        _,
    ) = split_data(
        X,
        y,
    )

    model = build_tuned_xgboost_pipeline()

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    predictions = trained_model.predict(
        X_test
    )

    probabilities = trained_model.predict_proba(
        X_test
    )

    assert len(
        predictions
    ) == len(
        X_test
    )

    assert probabilities.shape == (
        len(X_test),
        2,
    )

    assert set(
        predictions
    ).issubset(
        {0, 1}
    )