# Testing

## Overview

The project includes automated tests covering the machine learning pipeline, prediction logic, explainability utilities and FastAPI endpoints.

The current validated test suite contains:

```text
31 tests
```

## Run the Test Suite

From the project root, run:

```powershell
.\venv_compat\Scripts\python.exe -m pytest .\tests -v
```

For a shorter output:

```powershell
.\venv_compat\Scripts\python.exe -m pytest .\tests -q
```
## Test Coverage

The automated tests cover the main components of the predictive maintenance system, including:

- data preprocessing and validation
- model evaluation
- prediction logic
- SHAP explainability utilities
- training and utility functions
- FastAPI health checks
- normal-risk and high-risk prediction requests
- prediction history retrieval
- invalid history-limit handling

The API tests mock Supabase interactions so the test suite does not depend on a live database connection.
```
## Validated Result

The full automated test suite currently passes:

```text
31 passed
```

The test run also reports four dependency-related warnings from FastAPI/Starlette and SHAP. These are non-fatal deprecation warnings and do not indicate project test failures.

## Additional Validation

The Streamlit application has also been checked with Python compilation:

```powershell
.\venv_compat\Scripts\python.exe -m py_compile .\streamlit_app.py
```

Before committing repository changes, formatting can be checked with:

```powershell
git diff --check
```

## Related Documentation

- [API Reference](API.md)
- [Setup](SETUP.md)
- [Model and Decision Logic](MODEL.md)
```