from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run application startup and shutdown tasks.
    """

    print(
        "Industrial Predictive Maintenance API started."
    )

    yield

    print(
        "Industrial Predictive Maintenance API stopped."
    )


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "REST API for machine failure prediction "
        "using a tuned XGBoost model with SHAP "
        "explainability."
    ),
    version="1.0.0",
    contact={
        "name": "Vivian Chiamaka Ijomah",
    },
    lifespan=lifespan,
)


@app.get(
    "/",
    tags=["System"],
    summary="API Home",
)
def root():
    """
    Root endpoint.
    """

    return {
        "message": (
            "Industrial Predictive Maintenance "
            "API is running."
        )
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
)
def health_check():
    """
    Check API health status.
    """

    return {
        "status": "healthy"
    }


app.include_router(
    router,
    tags=["Prediction"],
)