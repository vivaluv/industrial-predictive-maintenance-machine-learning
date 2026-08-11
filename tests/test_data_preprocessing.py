from src.config import MODEL_FEATURES

from src.data_preprocessing import (
    load_data,
    prepare_model_data,
    split_data,
)


def test_load_data():
    """
    Test that the processed dataset loads correctly.
    """

    df = load_data()

    assert df is not None
    assert not df.empty
    assert df.shape[0] == 10000


def test_prepare_model_data():
    """
    Test preparation of predictors and target.
    """

    df = load_data()

    X, y = prepare_model_data(df)

    assert X is not None
    assert y is not None

    assert X.shape[0] == 10000
    assert y.shape[0] == 10000

    assert X.shape[1] == len(
        MODEL_FEATURES
    )

    assert (
        X.columns.tolist()
        == MODEL_FEATURES
    )

    assert set(
        y.unique()
    ) == {0, 1}


def test_target_distribution():
    """
    Test the known target distribution.
    """

    df = load_data()

    _, y = prepare_model_data(df)

    counts = y.value_counts()

    assert counts.loc[0] == 9661
    assert counts.loc[1] == 339


def test_split_data():
    """
    Test the stratified train/test split.
    """

    df = load_data()

    X, y = prepare_model_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    assert X_train.shape == (
        8000,
        len(MODEL_FEATURES),
    )

    assert X_test.shape == (
        2000,
        len(MODEL_FEATURES),
    )

    assert y_train.shape[0] == 8000
    assert y_test.shape[0] == 2000

    assert (
        X_train.columns.tolist()
        == MODEL_FEATURES
    )

    assert (
        X_test.columns.tolist()
        == MODEL_FEATURES
    )

    assert (
        y_train
        .value_counts()
        .loc[0]
        == 7729
    )

    assert (
        y_train
        .value_counts()
        .loc[1]
        == 271
    )

    assert (
        y_test
        .value_counts()
        .loc[0]
        == 1932
    )

    assert (
        y_test
        .value_counts()
        .loc[1]
        == 68
    )