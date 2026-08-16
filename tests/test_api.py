from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from src.config import DECISION_THRESHOLD


client = TestClient(app)


def test_health_endpoint():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


@patch(
    "app.routes.save_prediction"
)
def test_predict_normal_case(
    mock_save_prediction,
):
    mock_save_prediction.return_value = []

    payload = {
        "Air_temperature_K": 300.2,
        "Process_temperature_K": 309.4,
        "Rotational_speed_rpm": 1500,
        "Torque_Nm": 42.5,
        "Tool_wear_min": 120,
        "Type_M": 1,
        "Type_H": 0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "predicted_class"
    ] == 0

    assert data[
        "decision_prediction"
    ] == 0

    assert data[
        "decision_threshold"
    ] == DECISION_THRESHOLD

    assert (
        data[
            "failure_probability"
        ]
        < DECISION_THRESHOLD
    )

    assert isinstance(
        data[
            "top_contributors"
        ],
        list,
    )

    assert len(
        data[
            "top_contributors"
        ]
    ) > 0

    mock_save_prediction.assert_called_once()


@patch(
    "app.routes.save_prediction"
)
def test_predict_high_risk_case(
    mock_save_prediction,
):
    mock_save_prediction.return_value = []

    payload = {
        "Air_temperature_K": 298.9,
        "Process_temperature_K": 309.0,
        "Rotational_speed_rpm": 1410,
        "Torque_Nm": 65.7,
        "Tool_wear_min": 191,
        "Type_M": 0,
        "Type_H": 0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "predicted_class"
    ] == 1

    assert data[
        "decision_prediction"
    ] == 1

    assert data[
        "decision_threshold"
    ] == DECISION_THRESHOLD

    assert (
        data[
            "failure_probability"
        ]
        >= DECISION_THRESHOLD
    )

    assert isinstance(
        data[
            "top_contributors"
        ],
        list,
    )

    assert len(
        data[
            "top_contributors"
        ]
    ) > 0

    mock_save_prediction.assert_called_once()


@patch(
    "app.routes.get_prediction_history"
)
def test_history_endpoint(
    mock_get_prediction_history,
):
    mock_get_prediction_history.return_value = [
        {
            "id": 1,
            "machine_type": "L",
            "predicted_class": "Failure",
            "failure_probability": 0.99,
            "decision_threshold":
                DECISION_THRESHOLD,
            "decision":
                "Maintenance Required",
        }
    ]

    response = client.get(
        "/history",
        params={
            "limit": 5
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "count"
    ] == 1

    assert len(
        data[
            "predictions"
        ]
    ) == 1

    assert data[
        "predictions"
    ][0][
        "predicted_class"
    ] == "Failure"

    mock_get_prediction_history.assert_called_once_with(
        limit=5
    )


def test_history_invalid_limit():
    response = client.get(
        "/history",
        params={
            "limit": 0
        },
    )

    assert response.status_code == 400