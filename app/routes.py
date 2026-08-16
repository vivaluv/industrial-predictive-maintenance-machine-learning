import logging

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.model_loader import model
from app.supabase_client import (
    get_prediction_history,
    save_prediction,
)
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)

from src.config import (
    DECISION_THRESHOLD,
    MODEL_FEATURES,
)

from src.predict import (
    predict_with_explanation,
)

logger = logging.getLogger(
    __name__
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Machine Failure",
    description=(
        "Predict machine failure using the trained "
        "XGBoost model and return a threshold-based "
        "maintenance decision with local SHAP "
        "explanations."
    ),
)
def predict_failure(
    request: PredictionRequest,
):
    """
    Predict machine failure and return an
    explainable operational prediction.

    The request is converted into the exact
    feature structure expected by the trained
    machine learning pipeline.
    """

    try:

        request_data = (
            request.model_dump()
        )

        input_data = pd.DataFrame(
            [request_data],
            columns=MODEL_FEATURES,
        )

        results = predict_with_explanation(
            model=model,
            X=input_data,
            threshold=DECISION_THRESHOLD,
            top_n=3,
        )

        prediction = (
            results
            .iloc[0]
            .to_dict()
        )

        database_data = {
            "machine_type": (
                "M"
                if request_data["Type_M"] == 1
                else (
                    "H"
                    if request_data["Type_H"] == 1
                    else "L"
                )
            ),
            "air_temperature_k": request_data[
                "Air_temperature_K"
            ],
            "process_temperature_k": request_data[
                "Process_temperature_K"
            ],
            "rotational_speed_rpm": request_data[
                "Rotational_speed_rpm"
            ],
            "torque_nm": request_data[
                "Torque_Nm"
            ],
            "tool_wear_min": request_data[
                "Tool_wear_min"
            ],
        }

        save_prediction(
            request_data=database_data,
            prediction=prediction,
        )

        return prediction

    except Exception as error:

        logger.exception(
            "Machine failure prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Machine failure prediction "
                "could not be completed."
            ),
        ) from error


@router.get(
    "/history",
    summary="Get Prediction History",
    description=(
        "Retrieve recent machine failure predictions "
        "stored in Supabase."
    ),
)
def prediction_history(
    limit: int = 20,
):
    """
    Return recent prediction records.
    """

    try:

        if limit <= 0:
            raise HTTPException(
                status_code=400,
                detail="limit must be greater than 0.",
            )

        history = get_prediction_history(
            limit=limit,
        )

        return {
            "count": len(history),
            "predictions": history,
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "Prediction history retrieval failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction history could not "
                "be retrieved."
            ),
        ) from error