import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    precision_recall_curve,
)

from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
)

from sklearn.calibration import calibration_curve

from statsmodels.stats.contingency_tables import mcnemar

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
)


def evaluate_classifier(
    model,
    X_test,
    y_test,
):
    """
    Evaluate a trained binary classification model.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 score,
        ROC-AUC and Average Precision.
    """

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                y_prob,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test,
                y_prob,
            )
        ),
    }

    return metrics


def get_confusion_matrix(
    model,
    X_test,
    y_test,
):
    """
    Return the confusion matrix as a standard
    Python list.
    """

    y_pred = model.predict(
        X_test
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    return matrix.tolist()


def get_classification_report(
    model,
    X_test,
    y_test,
):
    """
    Return the classification report as a dictionary.
    """

    y_pred = model.predict(
        X_test
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return report


def predict_with_threshold(
    model,
    X,
    threshold=0.50,
):
    """
    Generate binary predictions using a custom
    probability threshold.

    Parameters
    ----------
    model
        Trained binary classification model.

    X
        Feature matrix used for prediction.

    threshold : float, default=0.50
        Probability threshold used to convert
        probabilities into binary predictions.

    Returns
    -------
    tuple
        Binary predictions and failure probabilities.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_prob = model.predict_proba(
        X
    )[:, 1]

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    return y_pred, y_prob


def cross_validate_model(
    model,
    X,
    y,
):
    """
    Perform stratified cross-validation.

    SMOTE should already be included inside the
    supplied model pipeline.

    Returns
    -------
    dict
        Mean cross-validation performance.
    """

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    results = {
        metric: float(
            scores[
                f"test_{metric}"
            ].mean()
        )
        for metric in scoring
    }

    return results


def evaluate_calibration(
    model,
    X_test,
    y_test,
    n_bins=10,
):
    """
    Evaluate probability calibration for a
    binary classification model.

    Returns
    -------
    dict
        Model Brier score, baseline Brier score,
        and calibration-curve values.
    """

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    model_brier_score = brier_score_loss(
        y_test,
        y_prob,
    )

    baseline_probability = float(
        y_test.mean()
    )

    baseline_probabilities = (
        np.full(
            shape=len(y_test),
            fill_value=baseline_probability,
            dtype=float,
        )
    )

    baseline_brier_score = brier_score_loss(
        y_test,
        baseline_probabilities,
    )

    (
        fraction_of_positives,
        mean_predicted_probability,
    ) = calibration_curve(
        y_test,
        y_prob,
        n_bins=n_bins,
        strategy="uniform",
    )

    results = {
        "model_brier_score": float(
            model_brier_score
        ),
        "baseline_brier_score": float(
            baseline_brier_score
        ),
        "fraction_of_positives": (
            fraction_of_positives.tolist()
        ),
        "mean_predicted_probability": (
            mean_predicted_probability.tolist()
        ),
    }

    return results


def find_optimal_threshold(
    model,
    X_test,
    y_test,
):
    """
    Find the probability threshold that maximises
    the F1 score on the supplied evaluation data.

    Returns
    -------
    dict
        Optimal threshold and associated metrics.
    """

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    (
        precision,
        recall,
        thresholds,
    ) = precision_recall_curve(
        y_test,
        y_prob,
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate an optimal threshold."
        )

    precision_for_thresholds = (
        precision[:-1]
    )

    recall_for_thresholds = (
        recall[:-1]
    )

    f1_scores = (
        2
        * precision_for_thresholds
        * recall_for_thresholds
        / (
            precision_for_thresholds
            + recall_for_thresholds
            + 1e-12
        )
    )

    best_index = int(
        np.argmax(
            f1_scores
        )
    )

    best_threshold = float(
        thresholds[
            best_index
        ]
    )

    results = {
        "optimal_threshold": best_threshold,
        "precision": float(
            precision_for_thresholds[
                best_index
            ]
        ),
        "recall": float(
            recall_for_thresholds[
                best_index
            ]
        ),
        "f1_score": float(
            f1_scores[
                best_index
            ]
        ),
    }

    return results


def evaluate_threshold(
    model,
    X_test,
    y_test,
    threshold,
):
    """
    Evaluate classification performance at a
    specified probability threshold.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 score,
        and confusion matrix at the threshold.
    """

    y_pred, _ = predict_with_threshold(
        model=model,
        X=X_test,
        threshold=threshold,
    )

    results = {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_test,
                y_pred,
            ).tolist()
        ),
    }

    return results


def mcnemar_test(
    model_a,
    model_b,
    X_test,
    y_test,
    exact=False,
    correction=True,
):
    """
    Perform McNemar's test to compare two classifiers
    evaluated on the same test observations.

    Returns
    -------
    dict
        Contingency table, test statistic,
        p-value, and significance decision.
    """

    pred_a = model_a.predict(
        X_test
    )

    pred_b = model_b.predict(
        X_test
    )

    y_true = np.asarray(
        y_test
    )

    correct_a = (
        pred_a == y_true
    )

    correct_b = (
        pred_b == y_true
    )

    table = np.array(
        [
            [
                np.sum(
                    correct_a
                    & correct_b
                ),
                np.sum(
                    correct_a
                    & ~correct_b
                ),
            ],
            [
                np.sum(
                    ~correct_a
                    & correct_b
                ),
                np.sum(
                    ~correct_a
                    & ~correct_b
                ),
            ],
        ]
    )

    result = mcnemar(
        table,
        exact=exact,
        correction=correction,
    )

    p_value = float(
        result.pvalue
    )

    statistic = float(
        result.statistic
    )

    return {
        "contingency_table": (
            table.tolist()
        ),
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_0_05": bool(
            p_value < 0.05
        ),
    }