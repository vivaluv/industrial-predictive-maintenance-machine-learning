import pytest

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
)

from src.data_preprocessing import (
    load_data,
    prepare_model_data,
    split_data,
)

from src.train_model import (
    build_tuned_xgboost_pipeline,
    train_model,
)

from src.utils import (
    save_model,
    load_model,
    save_json,
    load_json,
)


@pytest.fixture(scope="module")
def trained_model():
    """
    Build and train the tuned XGBoost pipeline
    once for persistence tests.
    """

    df = load_data()

    X, y = prepare_model_data(df)

    X_train, _, y_train, _ = split_data(
        X,
        y,
    )

    model = build_tuned_xgboost_pipeline()

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    return trained_model


def test_save_and_load_model(
    trained_model,
):
    """
    Test model persistence using joblib.
    """

    model_path = (
        MODELS_DIR
        / "test_pipeline.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        save_model(
            trained_model,
            model_path,
        )

        assert model_path.exists()

        loaded_model = load_model(
            model_path,
        )

        assert loaded_model is not None

        assert hasattr(
            loaded_model,
            "predict",
        )

        assert hasattr(
            loaded_model,
            "predict_proba",
        )

    finally:

        if model_path.exists():

            model_path.unlink()


def test_save_and_load_json():
    """
    Test JSON result persistence.
    """

    results = {
        "accuracy": 0.968,
        "roc_auc": 0.970,
        "model": "Tuned XGBoost",
    }

    json_path = (
        REPORTS_DIR
        / "test_results.json"
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        save_json(
            results,
            json_path,
        )

        assert json_path.exists()

        loaded_results = load_json(
            json_path,
        )

        assert loaded_results == results

    finally:

        if json_path.exists():

            json_path.unlink()