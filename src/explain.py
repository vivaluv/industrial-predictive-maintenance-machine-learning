import numpy as np
import pandas as pd
import shap


def _validate_feature_matrix(X):
    """
    Validate a feature matrix used for SHAP analysis.

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


def _validate_index(index, length):
    """
    Validate a positional observation index.
    """

    if not isinstance(index, int):
        raise TypeError(
            "index must be an integer."
        )

    if index < 0 or index >= length:
        raise IndexError(
            f"index must be between 0 and "
            f"{length - 1}."
        )


def get_tree_model(model):
    """
    Extract the underlying tree-based estimator.

    Supports:
    - imbalanced-learn pipelines
    - scikit-learn pipelines
    - directly supplied tree-based models

    Parameters
    ----------
    model
        Trained pipeline or tree-based estimator.

    Returns
    -------
    estimator
        Underlying trained tree estimator.
    """

    if model is None:
        raise ValueError(
            "model cannot be None."
        )

    if hasattr(model, "named_steps"):

        if "model" not in model.named_steps:
            raise ValueError(
                "Pipeline does not contain a "
                "'model' step."
            )

        return model.named_steps[
            "model"
        ]

    return model


def create_explainer(model):
    """
    Create a SHAP TreeExplainer for a trained
    tree-based model.

    Parameters
    ----------
    model
        Trained pipeline or tree-based estimator.

    Returns
    -------
    shap.TreeExplainer
        SHAP explainer for the underlying tree model.
    """

    tree_model = get_tree_model(
        model
    )

    return shap.TreeExplainer(
        tree_model
    )


def compute_shap_values(
    explainer,
    X,
):
    """
    Compute SHAP explanations for a feature matrix.

    Parameters
    ----------
    explainer
        Fitted SHAP explainer.

    X : pandas.DataFrame
        Feature matrix to explain.

    Returns
    -------
    shap.Explanation
        SHAP values for all supplied observations.
    """

    if explainer is None:
        raise ValueError(
            "explainer cannot be None."
        )

    _validate_feature_matrix(
        X
    )

    shap_values = explainer(
        X
    )

    return shap_values


def get_global_feature_importance(
    shap_values,
    X,
):
    """
    Calculate global feature importance using
    mean absolute SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        Computed SHAP explanations.

    X : pandas.DataFrame
        Feature matrix corresponding to the
        SHAP explanations.

    Returns
    -------
    pandas.DataFrame
        Features ranked by mean absolute
        SHAP contribution.
    """

    _validate_feature_matrix(
        X
    )

    if shap_values is None:
        raise ValueError(
            "shap_values cannot be None."
        )

    values = np.asarray(
        shap_values.values
    )

    if values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional SHAP values "
            "for binary classification."
        )

    if values.shape[0] != len(X):
        raise ValueError(
            "Number of SHAP observations does not "
            "match the number of rows in X."
        )

    if values.shape[1] != len(X.columns):
        raise ValueError(
            "Number of SHAP features does not "
            "match the columns in X."
        )

    importance_values = np.abs(
        values
    ).mean(
        axis=0
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": (
                importance_values
            ),
        }
    )

    importance = (
        importance
        .sort_values(
            by="mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return importance


def explain_single_prediction(
    shap_values,
    X,
    index,
):
    """
    Explain one individual model prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        Computed SHAP explanations.

    X : pandas.DataFrame
        Feature matrix corresponding to the
        SHAP explanations.

    index : int
        Positional index of the observation
        to explain.

    Returns
    -------
    list of dict
        Feature values and SHAP contributions
        ranked by absolute contribution.
    """

    _validate_feature_matrix(
        X
    )

    if shap_values is None:
        raise ValueError(
            "shap_values cannot be None."
        )

    _validate_index(
        index,
        len(X),
    )

    values = np.asarray(
        shap_values.values
    )

    if values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional SHAP values "
            "for binary classification."
        )

    if values.shape[0] != len(X):
        raise ValueError(
            "Number of SHAP observations does not "
            "match the number of rows in X."
        )

    row = X.iloc[
        index
    ]

    explanation_values = values[
        index
    ]

    explanation = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_value": (
                row.values
            ),
            "shap_value": (
                explanation_values
            ),
        }
    )

    explanation[
        "absolute_shap"
    ] = (
        explanation[
            "shap_value"
        ].abs()
    )

    explanation = (
        explanation
        .sort_values(
            by="absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    records = []

    for record in explanation.to_dict(
        orient="records"
    ):

        records.append(
            {
                "feature": str(
                    record[
                        "feature"
                    ]
                ),
                "feature_value": float(
                    record[
                        "feature_value"
                    ]
                ),
                "shap_value": float(
                    record[
                        "shap_value"
                    ]
                ),
                "absolute_shap": float(
                    record[
                        "absolute_shap"
                    ]
                ),
            }
        )

    return records


def get_top_contributors(
    shap_values,
    X,
    index,
    top_n=5,
):
    """
    Return the most influential features for
    an individual prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        Computed SHAP explanations.

    X : pandas.DataFrame
        Feature matrix corresponding to the
        SHAP explanations.

    index : int
        Positional index of the observation.

    top_n : int, default=5
        Number of leading contributors to return.

    Returns
    -------
    list of dict
        Top features ranked by absolute
        SHAP contribution.
    """

    if not isinstance(top_n, int):
        raise TypeError(
            "top_n must be an integer."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than 0."
        )

    explanation = explain_single_prediction(
        shap_values=shap_values,
        X=X,
        index=index,
    )

    return explanation[
        :top_n
    ]


def plot_shap_summary(
    shap_values,
    show=True,
):
    """
    Display the global SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        Computed SHAP explanations.

    show : bool, default=True
        Whether to display the plot immediately.
    """

    if shap_values is None:
        raise ValueError(
            "shap_values cannot be None."
        )

    shap.plots.beeswarm(
        shap_values,
        show=show,
    )


def plot_shap_bar(
    shap_values,
    show=True,
):
    """
    Display the global SHAP feature-importance
    bar plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        Computed SHAP explanations.

    show : bool, default=True
        Whether to display the plot immediately.
    """

    if shap_values is None:
        raise ValueError(
            "shap_values cannot be None."
        )

    shap.plots.bar(
        shap_values,
        show=show,
    )


def plot_shap_waterfall(
    shap_values,
    index,
    show=True,
):
    """
    Display a SHAP waterfall plot for one
    observation.

    Parameters
    ----------
    shap_values : shap.Explanation
        Computed SHAP explanations.

    index : int
        Positional index of the observation.

    show : bool, default=True
        Whether to display the plot immediately.
    """

    if shap_values is None:
        raise ValueError(
            "shap_values cannot be None."
        )

    _validate_index(
        index,
        len(shap_values),
    )

    shap.plots.waterfall(
        shap_values[
            index
        ],
        show=show,
    )