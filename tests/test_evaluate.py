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

from src.evaluate import (
    evaluate_classifier,
    get_confusion_matrix,
    get_classification_report,
    evaluate_calibration,
    find_optimal_threshold,
    evaluate_threshold,
)


@pytest.fixture(scope="module")
def trained_model_and_data():
    """
    Train the tuned XGBoost model once and reuse it
    across all evaluation tests.
    """

    df = load_data()

    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    model = build_tuned_xgboost_pipeline()

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    return trained_model, X_test, y_test


def test_evaluate_classifier(
    trained_model_and_data,
):
    """
    Test the standard classification metrics.
    """

    model, X_test, y_test = (
        trained_model_and_data
    )

    metrics = evaluate_classifier(
        model,
        X_test,
        y_test,
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    }

    assert expected_keys.issubset(
        metrics.keys()
    )

    for key in expected_keys:

        assert 0 <= metrics[key] <= 1


def test_confusion_matrix(
    trained_model_and_data,
):
    """
    Test confusion matrix structure and sample count.
    """

    model, X_test, y_test = (
        trained_model_and_data
    )

    matrix = get_confusion_matrix(
        model,
        X_test,
        y_test,
    )

    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert len(matrix[1]) == 2

    total_predictions = sum(
        sum(row)
        for row in matrix
    )

    assert total_predictions == len(
        y_test
    )


def test_classification_report(
    trained_model_and_data,
):
    """
    Test classification report contents.
    """

    model, X_test, y_test = (
        trained_model_and_data
    )

    report = get_classification_report(
        model,
        X_test,
        y_test,
    )

    assert "0" in report
    assert "1" in report
    assert "accuracy" in report
    assert "macro avg" in report
    assert "weighted avg" in report


def test_calibration(
    trained_model_and_data,
):
    """
    Test calibration evaluation output.
    """

    model, X_test, y_test = (
        trained_model_and_data
    )

    results = evaluate_calibration(
        model,
        X_test,
        y_test,
    )

    expected_keys = {
        "model_brier_score",
        "baseline_brier_score",
    }

    assert expected_keys.issubset(
        results.keys()
    )

    assert (
        0
        <= results["model_brier_score"]
        <= 1
    )

    assert (
        0
        <= results["baseline_brier_score"]
        <= 1
    )


def test_threshold_optimisation(
    trained_model_and_data,
):
    """
    Test threshold optimisation and threshold-based
    model evaluation.
    """

    model, X_test, y_test = (
        trained_model_and_data
    )

    threshold_results = (
        find_optimal_threshold(
            model,
            X_test,
            y_test,
        )
    )

    assert (
        "optimal_threshold"
        in threshold_results
    )

    threshold = threshold_results[
        "optimal_threshold"
    ]

    assert 0 <= threshold <= 1

    evaluation = evaluate_threshold(
        model,
        X_test,
        y_test,
        threshold,
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert expected_keys.issubset(
        evaluation.keys()
    )

    for key in expected_keys:

        assert 0 <= evaluation[key] <= 1