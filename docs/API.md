# API Reference

## Overview

The FastAPI service exposes the predictive maintenance system through endpoints for API status, machine failure prediction and prediction history.

The API supports the Streamlit dashboard and can also be used by external clients that need access to model inference.

## Base URL

For local development:

```text
http://127.0.0.1:8000

```
## API Home

```http
GET /
```

Returns a confirmation that the predictive maintenance API is running.

Example response:

```json
{
  "message": "Industrial Predictive Maintenance API is running."
}
```

## Health Check

```http
GET /health
```

Returns the current API health status.

Example response:

```json
{
  "status": "healthy"
}
```
## Machine Failure Prediction

```http
POST /predict
```

Predicts the probability of machine failure using the trained XGBoost model and returns both the model prediction and the threshold-based maintenance decision.

### Request Fields

| Field | Type | Description |
| --- | --- | --- |
| `Air_temperature_K` | float | Air temperature in Kelvin |
| `Process_temperature_K` | float | Process temperature in Kelvin |
| `Rotational_speed_rpm` | int | Machine rotational speed in RPM |
| `Torque_Nm` | float | Machine torque in Newton metres |
| `Tool_wear_min` | int | Accumulated tool wear in minutes |
| `Type_M` | int | Medium-quality machine indicator, 0 or 1 |
| `Type_H` | int | High-quality machine indicator, 0 or 1 |

### Machine Type Encoding

Machine type is represented using one-hot encoding, with Low as the reference category:

```text
Low:     Type_M = 0, Type_H = 0
Medium:  Type_M = 1, Type_H = 0
High:    Type_M = 0, Type_H = 1
```

`Type_M` and `Type_H` cannot both equal `1`.

### Example Request

```json
{
  "Air_temperature_K": 300.2,
  "Process_temperature_K": 309.4,
  "Rotational_speed_rpm": 1500,
  "Torque_Nm": 42.5,
  "Tool_wear_min": 120,
  "Type_M": 1,
  "Type_H": 0
}
```
### Response Fields

| Field | Description |
| --- | --- |
| `predicted_class` | Model classification: `0 = No Failure`, `1 = Failure` |
| `failure_probability` | Predicted probability of machine failure |
| `decision_threshold` | Operational threshold used for the maintenance decision |
| `decision_prediction` | Threshold-based decision: `0 = Normal Operation`, `1 = Maintenance Required` |
| `top_contributors` | Most influential local SHAP contributors |

Each item in `top_contributors` contains:

- `feature`
- `feature_value`
- `shap_value`
- `absolute_shap`

### Example Response

```json
{
  "predicted_class": 1,
  "failure_probability": 0.9988,
  "decision_threshold": 0.9252,
  "decision_prediction": 1,
  "top_contributors": [
    {
      "feature": "Torque_Nm",
      "feature_value": 65.7,
      "shap_value": 6.637,
      "absolute_shap": 6.637
    }
  ]
}
```
## Prediction History

```http
GET /history
```

Retrieves recent machine failure predictions stored in Supabase.

The endpoint accepts an optional `limit` query parameter.

Default:

```text
20
```

Example:

```http
GET /history?limit=20
```

If `limit` is less than or equal to `0`, the API returns HTTP `400`.

### Example Response Structure

```json
{
  "count": 2,
  "predictions": [
    {},
    {}
  ]
}
```
## Operational Decision Logic

The model produces a machine failure probability.

The application compares this probability with the configured operational threshold:

```text
0.9252
```

A probability at or above the threshold results in:

```text
Maintenance Required
```

Otherwise, the operational decision is:

```text
Normal Operation
```

The threshold is an operational decision rule and is separate from the model's default class prediction.

## Error Handling

Prediction failures and history retrieval failures return HTTP `500`.

Invalid history limits return HTTP `400`.

## Related Documentation

- [Model and Decision Logic](MODEL.md)
- [Setup](SETUP.md)
- [Testing](TESTING.md)