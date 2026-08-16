# Model, Explainability and Operational Decision Logic

## Model Overview

The final predictive maintenance system uses a tuned XGBoost classifier trained on the AI4I 2020 Predictive Maintenance Dataset.

Logistic Regression, Random Forest and XGBoost were evaluated during model development. The final production pipeline combines SMOTE with a tuned XGBoost classifier to address class imbalance during training.

## Model Inputs

The production model uses seven features:

| Feature | Description |
| --- | --- |
| Air_temperature_K | Air temperature in Kelvin |
| Process_temperature_K | Process temperature in Kelvin |
| Rotational_speed_rpm | Rotational speed in revolutions per minute |
| Torque_Nm | Torque in Newton-metres |
| Tool_wear_min | Tool wear in minutes |
| Type_M | Encoded medium machine type |
| Type_H | Encoded high machine type |

Machine type L is represented by Type_M = 0 and Type_H = 0.

## Production Pipeline

The final modelling pipeline applies SMOTE during training and uses a tuned XGBoost classifier for machine failure prediction.

## Predictive Performance

The final XGBoost pipeline was evaluated on a held-out test set of 2,000 observations, including 1,932 non-failure cases and 68 failure cases.

| Metric | Result |
| --- | ---: |
| Accuracy | 96.80% |
| ROC-AUC | 0.9700 |
| PR-AUC | 0.7715 |
| Failure Precision | 51.89% |
| Failure Recall | 80.88% |
| Failure F1-score | 63.22% |

The model identified 55 of the 68 failure cases in the held-out test set.

## Operational Decision Threshold

The application separates predictive probability from the operational maintenance decision.

A decision threshold of 0.9252 is used for maintenance prioritisation.

| Metric | Result |
| --- | ---: |
| Accuracy | 98.35% |
| Failure Precision | 84.31% |
| Failure Recall | 63.24% |
| Failure F1-score | 72.27% |

At this threshold, false maintenance alerts are reduced and failure precision increases, while failure recall decreases. The threshold therefore represents an operational trade-off rather than a universally superior classification setting.

## Explainable AI

SHAP is used to provide local explanations for individual XGBoost predictions.

Positive SHAP values increase predicted failure risk, while negative values decrease predicted failure risk. The application identifies the most influential features contributing to each prediction, providing greater transparency around the model output.

## Validation

The model and application were validated across the machine learning and application workflow, including model loading, prediction, SHAP explanation, decision-threshold logic, FastAPI inference, Supabase persistence, prediction history and Streamlit functionality.

The automated test suite currently passes 31 tests.

## Limitations

The system is developed and evaluated using the AI4I 2020 benchmark dataset rather than live industrial sensor data. The reported results therefore demonstrate the predictive maintenance methodology and application architecture under benchmark conditions.

Deployment in a real industrial environment would require equipment-specific validation, calibration, data-drift monitoring and appropriate operational and safety assessment.
