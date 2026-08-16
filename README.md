# Industrial Predictive Maintenance Using Explainable Machine Learning

An end-to-end industrial AI decision-support system that predicts machine failure risk, explains model decisions and translates predictions into actionable maintenance priorities.

## Project Overview

This project demonstrates an end-to-end predictive maintenance system using the AI4I 2020 Predictive Maintenance Dataset.

A tuned XGBoost model predicts machine failure risk, SHAP explains the factors influencing each prediction, and an operational threshold converts risk scores into maintenance decisions. FastAPI exposes the model, Supabase stores prediction history, and Streamlit provides the user interface.

## System Architecture

```text
Machine Operating Data
        |
        v
Data Validation
        |
        v
XGBoost + SMOTE Pipeline
        |
        v
Failure Probability
        |
        +--> SHAP Explanation
        |
        v
Operational Threshold
        |
        v
Maintenance Decision
        |
        +--> FastAPI
        +--> Supabase History
        +--> Streamlit Dashboard
```

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

The application uses an operational probability threshold of `0.9252` to convert model risk scores into maintenance decisions.

At this threshold, failure precision increases to `84.31%`, reducing false maintenance alerts while accepting lower recall.

See [docs/MODEL.md](docs/MODEL.md) for the full threshold evaluation and model validation.


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
|-- app/                  # FastAPI application and database integration
|-- data/                 # Project data
|-- docs/                 # Technical documentation
|-- models/               # Production model
|-- src/                  # Machine learning and prediction logic
|-- tests/                # Automated test suite
|-- DATA_DICTIONARY.md
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- .env.example
\-- streamlit_app.py
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/vivaluv/industrial-predictive-maintenance-machine-learning.git
cd industrial-predictive-maintenance-machine-learning
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Add your Supabase credentials to `.env` where required.

Start the FastAPI service:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the Streamlit dashboard:

```powershell
python -m streamlit run .\streamlit_app.py
```

Detailed setup and compatibility information is available in [docs/SETUP.md](docs/SETUP.md).

## Testing

The project includes automated tests across the machine learning and API layers.

Run the full test suite with:

```powershell
python -m pytest .\tests -q
```

Current validated result:

```text
31 passed
```

See [docs/TESTING.md](docs/TESTING.md) for testing and validation details.

## Limitations and Future Work

The current system is based on the AI4I 2020 benchmark dataset rather than live industrial telemetry. Real-world deployment would require validation against equipment-specific operating conditions, failure costs and maintenance policies.

Future development could include real-time sensor integration, model drift monitoring, cost-sensitive threshold optimisation and validation using operational industrial data.

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
