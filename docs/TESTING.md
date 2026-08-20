# Testing

## Overview

The project includes automated tests across the machine-learning and API layers, covering data preprocessing, prediction logic, explainability, validation and API behaviour.

The current test suite contains:

```text
31 automated tests
```

## Run the Test Suite

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

For detailed test output:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## Test Coverage

The automated test suite covers:

- data preprocessing and validation
- model evaluation and prediction logic
- SHAP explainability utilities
- training and utility functions
- FastAPI health checks
- normal-risk and high-risk prediction requests
- request validation
- prediction history retrieval
- invalid history-limit handling

Supabase interactions are mocked during API testing, allowing the automated suite to run without a live database connection.

## Validated Result

The current validated result is:

```text
31 passed
```

The test run also reports four non-fatal third-party dependency warnings associated with FastAPI/Starlette and SHAP/Matplotlib. These warnings do not represent project test failures.

## Additional Validation

Check the Streamlit application for Python syntax and compilation errors with:

```powershell
.\.venv\Scripts\python.exe -m py_compile streamlit_app.py
```

Verify dependency consistency with:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Check repository formatting before committing changes with:

```powershell
git diff --check
```

## Continuous Integration

GitHub Actions runs the automated test suite using Python 3.13.9.

A separate production verification workflow checks the health of the deployed Streamlit application.

## Related Documentation

- [API Reference](API.md)
- [Setup](SETUP.md)
- [Model and Decision Logic](MODEL.md)
