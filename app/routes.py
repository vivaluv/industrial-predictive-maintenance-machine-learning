import logging

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.model_loader import model
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