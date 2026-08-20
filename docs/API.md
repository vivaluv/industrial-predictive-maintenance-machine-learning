# API Reference

## Overview

The FastAPI service exposes the predictive maintenance system through endpoints for API status, machine failure prediction and prediction history.

It supports the Streamlit dashboard and can also be used by external clients that require programmatic access to model inference.

## Base URL

For local development:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API Home

```http
GET /
```

Confirms that the predictive maintenance API is running.

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

Predicts machine failure risk using the production XGBoost model.

The endpoint returns the model prediction, failure probability, operational maintenance decision and the three most influential local SHAP contributors.

### Request Fields

| Field | Type | Description |
| --- | --- | --- |
| `Air_temperature_K` | float | Air temperature in Kelvin |
| `Process_temperature_K` | float | Process temperature in Kelvin |
| `Rotational_speed_rpm` | int | Machine rotational speed in RPM |
| `Torque_Nm` | float | Machine torque in Newton metres |
| `Tool_wear_min` | int | Accumulated tool wear in minutes |
| `Type_M` | int | Medium-quality machine indicator: `0` or `1` |
| `Type_H` | int | High-quality machine indicator: `0` or `1` |

### Request Validation

The API validates incoming prediction data before model inference.

- `Rotational_speed_rpm`, `Torque_Nm` and `Tool_wear_min` must not be negative.
- `Type_M` and `Type_H` must each be either `0` or `1`.
- `Type_M` and `Type_H` cannot both equal `1`.
- Unexpected request fields are rejected.

Invalid prediction requests return HTTP `422`.

### Machine Type Encoding

Machine type is represented using one-hot encoding, with Low as the reference category:

```text
Low:     Type_M = 0, Type_H = 0
Medium:  Type_M = 1, Type_H = 0
High:    Type_M = 0, Type_H = 1
```

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
| `decision_threshold` | Operational threshold applied to the failure probability |
| `decision_prediction` | Maintenance decision: `0 = Normal Operation`, `1 = Maintenance Required` |
| `top_contributors` | Three most influential local SHAP contributors |

Each item in `top_contributors` contains:

- `feature` — feature associated with the contribution
- `feature_value` — input value supplied to the model
- `shap_value` — signed SHAP contribution
- `absolute_shap` — absolute contribution used for ranking

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

The example above shows the response structure. The production endpoint returns the three highest-ranked SHAP contributors.

## Prediction History

```http
GET /history
```

Retrieves recent prediction records stored in Supabase.

The endpoint accepts an optional `limit` query parameter.

Default:

```text
20
```

Example:

```http
GET /history?limit=20
```

The response contains:

- `count` — number of prediction records returned
- `predictions` — array of stored prediction records

If `limit` is less than or equal to `0`, the API returns HTTP `400`.

## Operational Decision Logic

The model produces a machine failure probability.

The application compares that probability with the configured operational threshold:

```text
0.9252
```

A probability greater than or equal to the threshold produces:

```text
Maintenance Required
```

A probability below the threshold produces:

```text
Normal Operation
```

The operational threshold is a decision rule applied after probability estimation and is separate from the model's default class prediction.

## Error Handling

| HTTP Status | Meaning |
| --- | --- |
| `400` | Invalid history limit |
| `422` | Prediction request failed input validation |
| `500` | Prediction or history retrieval failed during processing |

## Related Documentation

- [Model and Decision Logic](MODEL.md)
- [Setup](SETUP.md)
- [Testing](TESTING.md)
