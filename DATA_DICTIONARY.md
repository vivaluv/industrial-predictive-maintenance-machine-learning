{
  "name": "Industrial Predictive Maintenance",
  "image": "mcr.microsoft.com/devcontainers/python:1-3.13-bookworm",

  "customizations": {
    "codespaces": {
      "openFiles": [
        "README.md",
        "streamlit_app.py"
      ]
    },
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  },

  "postCreateCommand": "python -m pip install --upgrade pip && python -m pip install -r requirements.txt",

  "portsAttributes": {
    "8000": {
      "label": "FastAPI",
      "onAutoForward": "notify"
    },
    "8501": {
      "label": "Streamlit Dashboard",
      "onAutoForward": "openPreview"
    }
  },

  "forwardPorts": [
    8000,
    8501
  ]
}
