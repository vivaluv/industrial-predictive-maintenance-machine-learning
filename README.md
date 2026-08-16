# Industrial Predictive Maintenance Using Explainable Machine Learning

An end-to-end machine failure prediction system combining a tuned XGBoost model, SHAP explainability, FastAPI, Streamlit and Supabase to support risk-aware maintenance decisions.

## Project Overview

This project demonstrates how machine learning can be integrated into an operational predictive maintenance workflow rather than used only as a standalone prediction model.

Using the AI4I 2020 Predictive Maintenance Dataset, the system predicts machine failure risk, explains individual predictions with SHAP, applies an operational maintenance threshold, exposes predictions through a REST API and presents results through an interactive dashboard.

The project focuses on three questions:

1. Can machine failure risk be predicted accurately from operating conditions?
2. Can individual predictions be explained clearly enough to support maintenance decisions?
3. Can the model be integrated into a usable decision-support system?

## System Architecture

```text
Machine Operating Data
        │
        ▼
Data Validation
        │
        ▼
XGBoost + SMOTE Pipeline
        │
        ▼
Failure Probability
        │
        ├──────────────► SHAP Explanation
        │
        ▼
Operational Threshold
        │
        ▼
Maintenance Decision
        │
        ├──────────────► FastAPI
        │
        ├──────────────► Supabase History
        │
        └──────────────► Streamlit Dashboard
```

## Key Features

- Machine failure probability prediction using tuned XGBoost
- Class imbalance handling with SMOTE
- Decision-specific maintenance threshold
- Global and local SHAP explainability
- FastAPI prediction service
- Streamlit operational dashboard
- Supabase prediction history
- Maintenance alert and prioritisation workflow
- Automated testing across the ML and API layers

## Model Inputs

| Feature | Description |
| --- | --- |
| `Air_temperature_K` | Air temperature in Kelvin |
| `Process_temperature_K` | Process temperature in Kelvin |
| `Rotational_speed_rpm` | Rotational speed in RPM |
| `Torque_Nm` | Machine torque in Newton metres |
| `Tool_wear_min` | Accumulated tool wear in minutes |
| `Type_M` | Medium machine type indicator |
| `Type_H` | High machine type indicator |

Low machine type is represented by `Type_M = 0` and `Type_H = 0`.

## Model Performance

The production XGBoost pipeline was evaluated on a held-out test set of 2,000 observations.

| Metric | Result |
| --- | ---: |
| Accuracy | 96.80% |
| ROC-AUC | 0.9700 |
| PR-AUC | 0.7715 |
| Failure Precision | 51.89% |
| Failure Recall | 80.88% |
| Failure F1 | 63.22% |

Because machine failures are relatively rare, evaluation considers recall, precision, F1, ROC-AUC and PR-AUC rather than relying on accuracy alone.

### Operational Decision Threshold

The deployed decision layer uses a probability threshold of `0.9252`.

At this threshold:

| Metric | Result |
| --- | ---: |
| Accuracy | 98.35% |
| Failure Precision | 84.31% |
| Failure Recall | 63.24% |
| Failure F1 | 72.27% |

This threshold increases the precision of maintenance alerts while accepting lower failure recall. It is therefore treated as an operational trade-off rather than a universally superior model setting.

Further model details are available in [docs/MODEL.md](docs/MODEL.md).

## Explainability and Decision Support

SHAP is used to show which operating conditions contribute most strongly to each machine failure prediction.

The application separates:

```text
Model prediction
        ↓
Failure probability
        ↓
Operational threshold
        ↓
Maintenance decision
```

This allows the system to provide both predictive information and an interpretable operational recommendation.

## System Demo

### Streamlit Dashboard

![Streamlit predictive maintenance dashboard](images/streamlit_dashboard.png)

### SHAP Explainability

![SHAP feature contribution summary](images/shap_summary_plot.png)

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Machine Learning | XGBoost, scikit-learn, imbalanced-learn |
| Explainability | SHAP |
| Data Processing | pandas, NumPy |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Altair |
| Persistence | Supabase |
| Testing | pytest |
| Model Persistence | joblib |

## Project Structure

```text
industrial-predictive-maintenance-machine-learning/
├── app/                  # FastAPI application and database integration
├── data/                 # Project data
├── docs/                 # Technical documentation
├── images/               # Selected public project visuals
├── models/               # Production model
├── src/                  # Machine learning and prediction logic
├── tests/                # Automated test suite
├── DATA_DICTIONARY.md
├── LICENSE
├── README.md
├── requirements.txt
├── .env.example
└── streamlit_app.py
```

Detailed exploratory analysis and development artefacts are intentionally excluded from the public repository to keep the project focused on the reproducible production workflow.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/vivaluv/industrial-predictive-maintenance-machine-learning.git
cd industrial-predictive-maintenance-machine-learning
```

Create and activate the validated environment:

```powershell
python -m venv venv_compat
.\venv_compat\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file from `.env.example` and provide your Supabase credentials where required.

Start the FastAPI service:

```powershell
.\venv_compat\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the Streamlit dashboard in a second terminal:

```powershell
.\venv_compat\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Complete environment instructions are available in [docs/SETUP.md](docs/SETUP.md).

## Testing

The validated automated test suite currently contains:

```text
31 passed
```

Run the tests with:

```powershell
.\venv_compat\Scripts\python.exe -m pytest .\tests -q
```

See [docs/TESTING.md](docs/TESTING.md) for testing details.

## Limitations and Future Work

The current system is based on the AI4I 2020 benchmark dataset rather than live industrial telemetry. Real-world deployment would require validation against equipment-specific operating conditions, failure costs and maintenance policies.

Future development could include model drift monitoring, real-time sensor integration, cost-sensitive threshold optimisation, continuous model monitoring and validation using operational industrial data.

## Documentation

| Document | Purpose |
| --- | --- |
| [API Reference](docs/API.md) | API endpoints, request schemas and responses |
| [Model Documentation](docs/MODEL.md) | Model performance, threshold logic and explainability |
| [Setup Guide](docs/SETUP.md) | Environment and application setup |
| [Testing Guide](docs/TESTING.md) | Automated testing and validation |
| [Data Dictionary](DATA_DICTIONARY.md) | Dataset feature definitions |

## License

See [LICENSE](LICENSE) for licensing information.