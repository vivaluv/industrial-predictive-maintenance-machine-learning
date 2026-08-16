import json
import logging
import os

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()

logger = logging.getLogger(__name__)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set "
        "in the .env file."
    )


supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


def _make_json_safe(value):
    """
    Convert NumPy/Pandas values into
    JSON-compatible Python values.
    """

    if hasattr(value, "item"):
        return value.item()

    return str(value)


def save_prediction(
    request_data: dict,
    prediction: dict,
):
    """
    Save a machine failure prediction
    to the Supabase predictions table.
    """

    predicted_class = int(
        prediction.get(
            "predicted_class",
            0,
        )
    )

    decision_prediction = int(
        prediction.get(
            "decision_prediction",
            0,
        )
    )

    predicted_class_text = (
        "Failure"
        if predicted_class == 1
        else "No Failure"
    )

    decision_text = (
        "Maintenance Required"
        if decision_prediction == 1
        else "Normal Operation"
    )

    top_contributors = prediction.get(
        "top_contributors",
        [],
    )

    shap_values = json.loads(
        json.dumps(
            top_contributors,
            default=_make_json_safe,
        )
    )

    record = {
        "machine_type": request_data.get(
            "machine_type"
        ),
        "air_temperature_k": float(
            request_data.get(
                "air_temperature_k"
            )
        ),
        "process_temperature_k": float(
            request_data.get(
                "process_temperature_k"
            )
        ),
        "rotational_speed_rpm": int(
            request_data.get(
                "rotational_speed_rpm"
            )
        ),
        "torque_nm": float(
            request_data.get(
                "torque_nm"
            )
        ),
        "tool_wear_min": int(
            request_data.get(
                "tool_wear_min"
            )
        ),
        "predicted_class": predicted_class_text,
        "failure_probability": float(
            prediction.get(
                "failure_probability",
                0.0,
            )
        ),
        "decision_threshold": float(
            prediction.get(
                "decision_threshold",
                0.0,
            )
        ),
        "decision": decision_text,
        "shap_values": shap_values,
    }

    try:
        response = (
            supabase
            .table("predictions")
            .insert(record)
            .execute()
        )

        logger.info(
            "Prediction saved successfully "
            "to Supabase."
        )

        return response.data

    except Exception:
        logger.exception(
            "Failed to save prediction "
            "to Supabase."
        )
        raise


def get_prediction_history(
    limit: int = 20,
):
    """
    Retrieve recent prediction records
    from the Supabase predictions table.
    """

    if not isinstance(limit, int):
        raise TypeError(
            "limit must be an integer."
        )

    if limit <= 0:
        raise ValueError(
            "limit must be greater than 0."
        )

    try:
        response = (
            supabase
            .table("predictions")
            .select("*")
            .order(
                "created_at",
                desc=True,
            )
            .limit(limit)
            .execute()
        )

        logger.info(
            "Prediction history retrieved "
            "successfully from Supabase."
        )

        return response.data or []

    except Exception:
        logger.exception(
            "Failed to retrieve prediction "
            "history from Supabase."
        )
        raise