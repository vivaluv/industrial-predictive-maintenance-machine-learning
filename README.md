# Industrial Predictive Maintenance Using Explainable Machine Learning

## Project Overview

Industrial equipment failures can result in unplanned downtime, increased maintenance costs, production delays, and reduced operational efficiency. This project develops an explainable machine learning system for predicting equipment failure using the **AI4I 2020 Predictive Maintenance Dataset**.

Multiple classification algorithms, including **Logistic Regression, Random Forest, and XGBoost**, were developed and evaluated. The final system uses a tuned **XGBoost** pipeline for machine failure prediction and integrates **SHAP (SHapley Additive exPlanations)** to provide transparent global and local explanations of model predictions.

The project demonstrates an end-to-end machine learning solution, from data preprocessing and model development to explainability and application development through a **FastAPI REST API** and an interactive **Streamlit dashboard**.

## Key Features

* Predicts machine failure from operational machine conditions.
* Compares Logistic Regression, Random Forest, and XGBoost models.
* Handles class imbalance using SMOTE within the training pipeline.
* Uses stratified cross-validation and hyperparameter optimisation.
* Evaluates the final model using multiple classification and ranking metrics.
* Includes probability calibration and operational decision-threshold optimisation.
* Uses SHAP for global feature importance and individual prediction explanations.
* Includes statistical model validation and feature-group ablation analysis.
* Provides a FastAPI REST API for machine failure prediction.
* Provides an interactive Streamlit dashboard for prediction and explainability.
* Includes automated tests for core machine learning and application functionality.

## Tech Stack

* **Programming Language:** Python
* **Data Processing:** pandas, NumPy
* **Machine Learning:** scikit-learn, XGBoost
* **Class Imbalance:** imbalanced-learn (SMOTE)
* **Explainable AI:** SHAP
* **API:** FastAPI
* **Dashboard:** Streamlit
* **Testing:** pytest
* **Version Control:** Git and GitHub

## Project Structure
The repository is organised to support a complete end-to-end machine learning workflow, from data preparation and model development to explainability, testing, and application development.

```text
industrial-predictive-maintenance-machine-learning/
│
├── app/                         # FastAPI REST API
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Processed dataset used for modelling
├── images/                      # EDA, evaluation, SHAP and application figures
├── models/                      # Saved trained machine learning pipeline
├── notebooks/                   # Jupyter notebooks documenting the analysis
├── reports/                     # Evaluation results and generated reports
├── src/                         # Reusable machine learning source code
├── tests/                       # Automated tests
├── streamlit_app.py             # Interactive Streamlit dashboard
├── .gitignore
├── DATA_DICTIONARY.md
├── LICENSE
├── README.md
└── requirements.txt
```