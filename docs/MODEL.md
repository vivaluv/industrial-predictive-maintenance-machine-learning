# Model, Explainability and Operational Decision Logic

## Model Overview

The predictive maintenance system uses a tuned XGBoost classifier trained on the AI4I 2020 Predictive Maintenance Dataset.

Logistic Regression, Random Forest and XGBoost were evaluated during model development. XGBoost was selected for the production system based on its predictive performance and suitability for the imbalanced binary classification problem.

SMOTE is applied only to the training data to improve representation of the minority failure class. It is not used during inference.

## Model Inputs

The production model uses seven input features:

| Feature | Description |
| --- | --- |
| `Air_temperature_K` | Air temperature in Kelvin |
| `Process_temperature_K` | Process temperature in Kelvin |
| `Rotational_speed_rpm` | Rotational speed in revolutions per minute |
| `Torque_Nm` | Torque in Newton-metres |
| `Tool_wear_min` | Accumulated tool wear in minutes |
| `Type_M` | Encoded medium machine type |
| `Type_H` | Encoded high machine type |

Machine type is one-hot encoded with Low as the reference category:

```text
Low:     Type_M = 0, Type_H = 0
Medium:  Type_M = 1, Type_H = 0
High:    Type_M = 0, Type_H = 1
```

## Production Pipeline

During model training, the pipeline applies SMOTE to the training set before fitting the tuned XGBoost classifier.

During inference, validated input features are passed directly to the trained production model to generate a machine failure probability and model classification.

The predicted probability is then evaluated against the operational decision threshold used by the application.

## Predictive Performance

The production model was evaluated on a held-out test set of 2,000 observations:

```text
Non-failure cases: 1,932
Failure cases:        68
```

Performance using the model's default classification behaviour:

| Metric | Result |
| --- | ---: |
| Accuracy | 96.80% |
| ROC-AUC | 0.9700 |
| PR-AUC | 0.7715 |
| Failure Precision | 51.89% |
| Failure Recall | 80.88% |
| Failure F1 | 63.22% |

The model identified 55 of the 68 failure cases in the held-out test set.

Because failure events are relatively rare, model assessment considers precision, recall, F1, ROC-AUC and PR-AUC rather than relying on accuracy alone.

## Operational Decision Threshold

The application separates the model's predicted probability from the operational maintenance decision.

An operational threshold of:

```text
0.9252
```

is applied to the predicted failure probability.

At this threshold:

| Metric | Result |
| --- | ---: |
| Accuracy | 98.35% |
| Failure Precision | 84.31% |
| Failure Recall | 63.24% |
| Failure F1 | 72.27% |

A probability greater than or equal to `0.9252` results in:

```text
Maintenance Required
```

A probability below the threshold results in:

```text
Normal Operation
```

The higher threshold substantially increases failure precision and reduces false maintenance alerts, while accepting lower failure recall. It therefore represents an operational trade-off rather than a universally superior classification setting.

## Explainable AI

SHAP provides local explanations for individual XGBoost predictions.

Positive SHAP contributions increase the model output towards failure risk, while negative contributions reduce it. Features are ranked by the absolute magnitude of their SHAP contribution.

For each prediction, the application returns the three highest-ranked local contributors together with:

- feature name
- input feature value
- signed SHAP value
- absolute SHAP magnitude

These explanations provide additional context for the prediction but do not replace engineering judgement or operational validation.

## Validation

The system has been validated across the machine-learning and application workflow, including:

- model loading and inference
- prediction probability generation
- operational threshold logic
- SHAP explanation generation
- FastAPI request validation and inference
- Supabase prediction persistence and history retrieval
- Streamlit application functionality

The automated test suite currently passes:

```text
31 tests
```

See [TESTING.md](TESTING.md) for detailed test and validation information.

## Limitations

The system is developed and evaluated using the AI4I 2020 benchmark dataset rather than live industrial sensor telemetry.

The reported results therefore demonstrate the predictive-maintenance methodology and application architecture under benchmark conditions. They should not be interpreted as evidence of equivalent performance on unseen industrial equipment.

Real-world deployment would require equipment-specific validation, probability calibration where appropriate, data and model-drift monitoring, maintenance-policy evaluation, and operational and safety assessment.

## Related Documentation

- [API Reference](API.md)
- [Setup](SETUP.md)
- [Testing](TESTING.md)
- [Data Dictionary](../DATA_DICTIONARY.md)
