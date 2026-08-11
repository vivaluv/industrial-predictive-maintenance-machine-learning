from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, NUMERIC_FEATURES


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to the training data only.

    Returns
    -------
    X_resampled, y_resampled
        Balanced training features and target.
    """

    smote = SMOTE(random_state=RANDOM_STATE)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


def scale_numeric_features(X_train, X_test):
    """
    Standardise numerical features for models that require scaling,
    such as Logistic Regression.

    The scaler is fitted only on the training data.
    """

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(
        X_train[NUMERIC_FEATURES]
    )

    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(
        X_test[NUMERIC_FEATURES]
    )

    return X_train_scaled, X_test_scaled, scaler