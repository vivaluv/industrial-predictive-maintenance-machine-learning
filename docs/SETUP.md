# Setup and Environment

## Requirements

The project has been validated with:

```text
Python 3.13.9
```

Core production dependencies include:

```text
NumPy 2.3.5
pandas 2.3.3
scikit-learn 1.7.2
imbalanced-learn 0.14.0
XGBoost 3.2.0
SHAP 0.52.0
```

All project dependencies and versions are pinned in `requirements.txt`.

## Clone the Repository

```bash
git clone https://github.com/vivaluv/industrial-predictive-maintenance-machine-learning.git
cd industrial-predictive-maintenance-machine-learning
```

## Create the Virtual Environment

Confirm that the Python interpreter you intend to use is Python 3.13.9:

```powershell
python --version
```

If `python` resolves to Python 3.13.9, create the virtual environment with:

```powershell
python -m venv .venv
```

If another Python version is returned, invoke your installed Python 3.13.9 interpreter explicitly when creating `.venv`.

Do not recreate an existing project environment using a different Python version.

## Windows PowerShell

Virtual-environment activation is optional.

If PowerShell permits local script execution:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by the Windows execution policy, changing the system policy is not required. Use the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe --version
```

The expected result is:

```text
Python 3.13.9
```

## Install Dependencies

Install the pinned dependencies using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify dependency consistency:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

A healthy environment should report:

```text
No broken requirements found.
```

## Environment Variables

Create a local environment file from the provided template:

```powershell
Copy-Item .env.example .env
```

Add your Supabase credentials to `.env`:

```text
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key
```

The real `.env` file is excluded from version control and must not be committed or shared publicly.

## Run the FastAPI Service

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Local API:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

## Run the Streamlit Dashboard

With FastAPI running, open a second terminal in the project directory and start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Streamlit will display the local dashboard address in the terminal.

## Validate the Installation

Run the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Current validated result:

```text
31 passed
```

See [TESTING.md](TESTING.md) for detailed validation coverage and known third-party dependency warnings.

## Compatibility

The production environment is validated with Python 3.13.9 and the pinned dependencies in `requirements.txt`.

Using materially different versions of scikit-learn, XGBoost or related dependencies may produce model-loading compatibility warnings or inconsistent inference behaviour.

For reproducible results, use the documented Python version and pinned environment.

## Related Documentation

- [API Reference](API.md)
- [Model and Decision Logic](MODEL.md)
- [Testing](TESTING.md)
