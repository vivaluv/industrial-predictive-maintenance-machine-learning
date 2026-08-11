"""
Industrial Predictive Maintenance Machine Learning

A modular package for industrial equipment fault prediction,
model evaluation, explainable AI, and deployment.
"""

__version__ = "1.0.0"
__author__ = "Vivian Chiamaka Ijomah"


# ============================
# Data preprocessing
# ============================

from .data_preprocessing import (
    load_data,
    prepare_model_data,
    split_data,
)


# ============================
# Feature engineering
# ============================

from .feature_engineering import (
    apply_smote,
    scale_numeric_features,
)


# ============================
# Model training
# ============================

from .train_model import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    build_tuned_xgboost_pipeline,
    train_model,
)


# ============================
# Model evaluation
# ============================

from .evaluate import (
    evaluate_classifier,
    get_confusion_matrix,
    get_classification_report,
    predict_with_threshold,
    cross_validate_model,
    evaluate_calibration,
    evaluate_threshold,
    find_optimal_threshold,
    mcnemar_test,
)


# ============================
# Explainable AI
# ============================

from .explain import (
    create_explainer,
    compute_shap_values,
    get_global_feature_importance,
    explain_single_prediction,
    get_top_contributors,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
)


# ============================
# Public package API
# ============================

__all__ = [
    # Data preprocessing
    "load_data",
    "prepare_model_data",
    "split_data",

    # Feature engineering
    "apply_smote",
    "scale_numeric_features",

    # Model training
    "build_logistic_regression",
    "build_random_forest",
    "build_xgboost",
    "build_tuned_xgboost_pipeline",
    "train_model",

    # Model evaluation
    "evaluate_classifier",
    "get_confusion_matrix",
    "get_classification_report",
    "predict_with_threshold",
    "cross_validate_model",
    "evaluate_calibration",
    "evaluate_threshold",
    "find_optimal_threshold",
    "mcnemar_test",

    # Explainable AI
    "create_explainer",
    "compute_shap_values",
    "get_global_feature_importance",
    "explain_single_prediction",
    "get_top_contributors",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_waterfall",
]