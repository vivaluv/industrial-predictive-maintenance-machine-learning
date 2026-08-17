import os
from pathlib import Path

# ==============================
# Project Paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "ai4i2020_processed.csv"
)


# ==============================
# Dataset Configuration
# ==============================

TARGET_COLUMN = "Machine failure"

IDENTIFIER_COLUMNS = [
    "UDI",
    "Product ID"
]

FAILURE_MECHANISM_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


# ==============================
# Model Features
# ==============================

NUMERIC_FEATURES = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min"
]

MODEL_FEATURES = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_M",
    "Type_H"
]


# ==============================
# Machine Learning Configuration
# ==============================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ==============================
# Decision Threshold
# ==============================

DECISION_THRESHOLD = 0.9252

# ==============================
# Final Model Configuration
# ==============================

MODEL_FILENAME = "xgboost_pipeline.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# ==========================
# API Configuration
# ==========================

API_HOST = "127.0.0.1"
API_PORT = 8000

API_URL = os.getenv(
    "API_URL",
    f"http://{API_HOST}:{API_PORT}/predict",
)
