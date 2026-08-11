import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    IDENTIFIER_COLUMNS,
    FAILURE_MECHANISM_COLUMNS,
    MODEL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_data(path=PROCESSED_DATA_PATH):
    """
    Load the processed AI4I predictive maintenance dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the processed dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    df = pd.read_csv(path)

    return df


def prepare_model_data(df):
    """
    Prepare the dataset for machine learning.

    Steps
    -----
    1. Copy the input dataframe.
    2. Remove identifier columns if present.
    3. One-hot encode machine Type if necessary.
    4. Separate the target variable.
    5. Remove failure-mechanism columns.
    6. Convert feature names to XGBoost-safe names.
    7. Validate and enforce the final model feature set.
    """

    df_model = df.copy()

    # ============================
    # Remove identifiers
    # ============================

    columns_to_drop = [
        col
        for col in IDENTIFIER_COLUMNS
        if col in df_model.columns
    ]

    if columns_to_drop:
        df_model = df_model.drop(
            columns=columns_to_drop
        )


    # ============================
    # Encode machine type
    # ============================

    if "Type" in df_model.columns:

        df_model["Type"] = pd.Categorical(
            df_model["Type"],
            categories=[
                "L",
                "M",
                "H"
            ],
            ordered=False
        )

        df_model = pd.get_dummies(
            df_model,
            columns=["Type"],
            drop_first=True,
            dtype=int
        )


    # ============================
    # Validate target
    # ============================

    if TARGET_COLUMN not in df_model.columns:

        raise ValueError(
            f"Target column "
            f"'{TARGET_COLUMN}' "
            f"was not found in the dataset."
        )


    # ============================
    # Separate target
    # ============================

    y = df_model[
        TARGET_COLUMN
    ].copy()


    # ============================
    # Remove leakage columns
    # ============================

    excluded_columns = [
        TARGET_COLUMN,
        *FAILURE_MECHANISM_COLUMNS
    ]

    X = df_model.drop(
        columns=[
            col
            for col in excluded_columns
            if col in df_model.columns
        ]
    )


    # ============================
    # XGBoost-safe column names
    # ============================

    X.columns = (
        X.columns
        .str.replace(
            "[",
            "",
            regex=False
        )
        .str.replace(
            "]",
            "",
            regex=False
        )
        .str.replace(
            "<",
            "",
            regex=False
        )
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )


    # ============================
    # Validate model features
    # ============================

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in X.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required model features: "
            f"{missing_features}"
        )


    # ============================
    # Enforce feature order
    # ============================

    X = X[
        MODEL_FEATURES
    ].copy()


    return X, y


def split_data(
    X,
    y
):
    """
    Create a stratified train/test split.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )