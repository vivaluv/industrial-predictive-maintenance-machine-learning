# Setup and Environment

## Requirements

The project was validated with:

```text
Python 3.13.9
```

The production model was created and tested with the following core environment:

```text
NumPy 2.3.5
pandas 2.3.3
scikit-learn 1.7.2
imbalanced-learn 0.14.0
XGBoost 3.2.0
SHAP 0.52.0
```

Exact package versions are provided in `requirements.txt`.

## Clone the Repository

```bash
git clone https://github.com/vivaluv/industrial-predictive-maintenance-machine-learning.git
cd industrial-predictive-maintenance-machine-learning
```

## Create a Virtual Environment

Create a dedicated virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

Install the pinned project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Variables

Create a local environment file from the provided example:

```powershell
Copy-Item .env.example .env
```

Then update `.env` with your Supabase project credentials:

```text
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key
```

Do not commit the real `.env` file or expose production credentials in the repository.

## Run the FastAPI Service

Start the API locally:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Run the Streamlit Dashboard

With the FastAPI service running, open a second terminal in the project directory, activate the same virtual environment, and start the dashboard:

```powershell
python -m streamlit run .\streamlit_app.py
```

Streamlit will display the local dashboard URL in the terminal.

## Compatibility Note

The production model was validated using Python 3.13.9 and the pinned package versions in `requirements.txt`.

Using significantly different versions of scikit-learn or XGBoost may cause model-loading compatibility warnings or inconsistent behaviour. For reproducible results, use the documented Python version and pinned dependencies.

## Related Documentation

- [API Reference](API.md)
- [Model and Decision Logic](MODEL.md)
- [Testing](TESTING.md)