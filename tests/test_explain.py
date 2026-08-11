import pytest

from src.data_preprocessing import (
    load_data,
    prepare_model_data,
    split_data,
)

from src.train_model import (
    build_tuned_xgboost_pipeline,
    train_model,
)

from src.explain import (
    get_tree_model,
    create_explainer,
    compute_shap_values,
    get_global_feature_importance,
    explain_single_prediction,
    get_top_contributors,
)


@pytest.fixture(scope="module")
def trained_model_and_sample():
    """
    Train the tuned XGBoost model once and
    create SHAP explanations for testing.
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

    X_sample = X_test.head(20)

    explainer = create_explainer(
        trained_model,
    )

    shap_values = compute_shap_values(
        explainer,
        X_sample,
    )

    return (
        trained_model,
        X_sample,
        explainer,
        shap_values,
    )


def test_get_tree_model(
    trained_model_and_sample,
):
    """
    Test extraction of the tree estimator.
    """

    model, _, _, _ = trained_model_and_sample

    tree_model = get_tree_model(
        model,
    )

    assert tree_model is not None


def test_create_explainer(
    trained_model_and_sample,
):
    """
    Test SHAP explainer creation.
    """

    _, _, explainer, _ = (
        trained_model_and_sample
    )

    assert explainer is not None


def test_compute_shap_values(
    trained_model_and_sample,
):
    """
    Test SHAP value computation.
    """

    _, X_sample, _, shap_values = (
        trained_model_and_sample
    )

    assert len(shap_values) == len(
        X_sample
    )


def test_global_feature_importance(
    trained_model_and_sample,
):
    """
    Test global SHAP importance calculation.
    """

    (
        _,
        X_sample,
        _,
        shap_values,
    ) = trained_model_and_sample

    importance = get_global_feature_importance(
        shap_values,
        X_sample,
    )

    assert len(importance) == len(
        X_sample.columns
    )

    assert (
        "feature"
        in importance.columns
    )

    assert (
        "mean_abs_shap"
        in importance.columns
    )


def test_single_prediction_explanation(
    trained_model_and_sample,
):
    """
    Test local SHAP explanation.
    """

    (
        _,
        X_sample,
        _,
        shap_values,
    ) = trained_model_and_sample

    explanation = explain_single_prediction(
        shap_values,
        X_sample,
        index=0,
    )

    assert isinstance(
        explanation,
        list,
    )

    assert len(explanation) == len(
        X_sample.columns
    )


def test_top_contributors(
    trained_model_and_sample,
):
    """
    Test retrieval of the leading SHAP
    contributors.
    """

    (
        _,
        X_sample,
        _,
        shap_values,
    ) = trained_model_and_sample

    contributors = get_top_contributors(
        shap_values,
        X_sample,
        index=0,
        top_n=3,
    )

    assert len(
        contributors
    ) == 3

    for feature in contributors:

        assert "feature" in feature
        assert "feature_value" in feature
        assert "shap_value" in feature