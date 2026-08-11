from fastapi import FastAPI

from app.routes import router


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "REST API for machine failure prediction "
        "using a tuned XGBoost model with SHAP explainability."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Industrial Predictive Maintenance API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


app.include_router(router)