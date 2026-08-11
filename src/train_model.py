from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def build_logistic_regression():
    """
    Build Logistic Regression baseline model.
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    return model


def build_random_forest():
    """
    Build Random Forest baseline model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return model


def build_xgboost():
    """
    Build baseline XGBoost model.
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    return model


def build_tuned_xgboost_pipeline():
    """
    Build the final tuned XGBoost training pipeline.

    SMOTE is included inside the pipeline so that it is
    applied only during model fitting.
    """

    pipeline = Pipeline([
        (
            "smote",
            SMOTE(random_state=RANDOM_STATE)
        ),
        (
            "model",
            XGBClassifier(
                subsample=0.9,
                n_estimators=100,
                min_child_weight=1,
                max_depth=8,
                learning_rate=0.1,
                gamma=0,
                colsample_bytree=1.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            )
        )
    ])

    return pipeline


def train_model(model, X_train, y_train):
    """
    Train a supplied machine learning model.
    """

    model.fit(
        X_train,
        y_train
    )

    return model
