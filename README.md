# Industrial Predictive Maintenance Using Explainable Machine Learning

An end-to-end industrial AI decision-support system for predicting machine failure risk, explaining model decisions and supporting maintenance prioritisation.

## Overview

Built using the AI4I 2020 Predictive Maintenance Dataset, the system combines a tuned XGBoost model with SHAP explainability and an operational decision threshold.

FastAPI provides model inference, Streamlit delivers the user interface, and Supabase stores prediction history.

## Live Application

[Open the deployed application](https://vivaluv-predictive-maintenance.streamlit.app)

> The current application uses benchmark operating data entered through the dashboard rather than continuous live industrial telemetry.

## System Architecture

```text
Machine Operating Data
        |
        v
Streamlit Dashboard
        |
        v
FastAPI Validation
        |
        v
Tuned XGBoost Model
        |
        +--> Failure Probability
        |
        +--> SHAP Explanation
        |
        v
Operational Threshold
        |
        v
Maintenance Decision
        |
        v
Supabase Prediction History
```

## Model Performance

On a held-out test set of 2,000 observations, the production model achieved:

| Metric | Result |
| --- | ---: |
| Accuracy | 96.80% |
| ROC-AUC | 0.9700 |
| Failure Recall | 80.88% |
| Failure F1 | 63.22% |

The operational threshold is `0.9252`, increasing failure precision to **84.31%** for maintenance prioritisation.

See [Model Documentation](docs/MODEL.md) for the full evaluation, threshold analysis and explainability details.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.13.9 |
| Machine Learning | XGBoost, scikit-learn, imbalanced-learn |
| Explainability | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Altair |
| Persistence | Supabase |
| Testing | pytest |
| CI | GitHub Actions |

## Project Structure

```text
industrial-predictive-maintenance-machine-learning/
|-- app/                  # FastAPI application
|-- data/                 # Project data
|-- docs/                 # Technical documentation
|-- models/               # Production model
|-- src/                  # ML and prediction logic
|-- tests/                # Automated tests
|-- DATA_DICTIONARY.md
|-- requirements.txt
\-- streamlit_app.py
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/vivaluv/industrial-predictive-maintenance-machine-learning.git
cd industrial-predictive-maintenance-machine-learning
```

Use Python 3.13.9, create a virtual environment and install the pinned dependencies:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Supabase credentials to `.env`, then start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

See the [Setup Guide](docs/SETUP.md) for full environment and compatibility instructions.


## Documentation

| Document | Purpose |
| --- | --- |
| [API Reference](docs/API.md) | Endpoints, validation and responses |
| [Model Documentation](docs/MODEL.md) | Model performance, threshold logic and SHAP |
| [Setup Guide](docs/SETUP.md) | Installation and environment configuration |
| [Testing Guide](docs/TESTING.md) | Automated testing and validation |
| [Data Dictionary](DATA_DICTIONARY.md) | Dataset and feature definitions |

## Limitations

The current system is validated using the AI4I 2020 benchmark dataset. Real-world industrial deployment would require equipment-specific validation and operational testing.

## License

See [LICENSE](LICENSE) for software licensing information.
