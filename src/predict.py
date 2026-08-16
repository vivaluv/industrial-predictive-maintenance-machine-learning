import pandas as pd

from src.config import DECISION_THRESHOLD
from src.evaluate import predict_with_threshold
from src.explain import (
    create_explainer,
    compute_shap_values,
    get_top_contributors,
)


def _validate_input(X):
    """
    Validate prediction input data.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.

    ValueError
        If X is empty.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Input feature matrix is empty."
        )


def _validate_threshold(threshold):
    """
    Validate a probability decision threshold.

    Parameters
    ----------
    threshold : float
        Probability threshold between 0 and 1.

    Raises
    ------
    ValueError
        If threshold is outside the valid range.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )


def predict_failure(
    model,
    X,
):
    """
    Predict machine failure using the model's
    default classification decision.

    Parameters
    ----------
    model
        Trained classification model.

    X : pandas.DataFrame
        Feature matrix.

    Returns
    -------
    numpy.ndarray
        Predicted binary class labels.
    """

    _validate_input(X)

    return model.predict(X)


def predict_probability(
    model,
    X,
):
    """
    Predict machine failure probabilities.

    Parameters
    ----------
    model
        Trained classification model.

    X : pandas.DataFrame
        Feature matrix.

    Returns
    -------
    numpy.ndarray
        Probability of machine failure.
    """

    _validate_input(X)

    probabilities = model.predict_proba(X)

    return probabilities[:, 1]


def predict_risk_level(
    model,
    X,
    threshold=DECISION_THRESHOLD,
):
    """
    Convert failure probabilities into
    operational maintenance decisions.

    Parameters
    ----------
    model
        Trained classification model.

    X : pandas.DataFrame
        Feature matrix.

    threshold : float, default=DECISION_THRESHOLD
        Operational probability threshold.

    Returns
    -------
    numpy.ndarray
        Binary operational decisions.
    """

    _validate_input(X)
    _validate_threshold(threshold)

    predictions, _ = predict_with_threshold(
        model=model,
        X=X,
        threshold=threshold,
    )

    return predictions


def predict_with_explanation(
    model,
    X,
    threshold=DECISION_THRESHOLD,
    top_n=5,
):
    """
    Predict machine failure together with
    probability, operational decision,
    and local SHAP explanations.

    Parameters
    ----------
    model
        Trained classification model.

    X : pandas.DataFrame
        Feature matrix.

    threshold : float, default=DECISION_THRESHOLD
        Operational probability threshold.

    top_n : int, default=5
        Number of leading SHAP contributors
        returned for each prediction.

    Returns
    -------
    pandas.DataFrame
        Prediction results containing class,
        probability, decision threshold,
        operational decision, and SHAP
        contributors.
    """

    _validate_input(X)
    _validate_threshold(threshold)

    if not isinstance(top_n, int):
        raise TypeError(
            "top_n must be an integer."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than 0."
        )

    # ============================
    # Predictions
    # ============================

    predicted_classes = predict_failure(
        model=model,
        X=X,
    )

    probabilities = predict_probability(
        model=model,
        X=X,
    )

    decision_predictions = predict_risk_level(
        model=model,
        X=X,
        threshold=threshold,
    )

    # ============================
    # SHAP Explanation
    # ============================

    explainer = create_explainer(
        model
    )

    shap_values = compute_shap_values(
        explainer=explainer,
        X=X,
    )

    # ============================
    # Build Results
    # ============================

    results = []

    for index in range(len(X)):

        explanation = get_top_contributors(
            shap_values=shap_values,
            X=X,
            index=index,
            top_n=top_n,
        )

        results.append(
            {
                "predicted_class": int(
                    predicted_classes[index]
                ),
                "failure_probability": float(
                    probabilities[index]
                ),
                "decision_threshold": float(
                    threshold
                ),
                "decision_prediction": int(
                    decision_predictions[index]
                ),
                "top_contributors": explanation,
            }
        )

    return pd.DataFrame(results)