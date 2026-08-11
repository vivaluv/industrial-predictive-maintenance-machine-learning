import json
from pathlib import Path
from typing import Any, Union

import joblib


def save_model(
    model: Any,
    path: Union[str, Path],
) -> Path:
    """
    Save a trained model or pipeline.

    Parameters
    ----------
    model
        Trained model or pipeline.

    path
        Destination file path.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved model.
    """

    if model is None:
        raise ValueError(
            "model cannot be None."
        )

    path = Path(path)

    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )

    return path.resolve()


def load_model(
    path: Union[str, Path],
) -> Any:
    """
    Load a saved model or pipeline.

    Parameters
    ----------
    path
        Model file path.

    Returns
    -------
    object
        Loaded model.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    model = joblib.load(
        path
    )

    return model


def save_json(
    data: Any,
    path: Union[str, Path],
) -> Path:
    """
    Save JSON-compatible data.

    Parameters
    ----------
    data
        JSON-compatible object.

    path
        Destination JSON file.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved JSON file.
    """

    if data is None:
        raise ValueError(
            "data cannot be None."
        )

    path = Path(path)

    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
        )

    return path.resolve()


def load_json(
    path: Union[str, Path],
) -> Any:
    """
    Load JSON data.

    Parameters
    ----------
    path
        JSON file path.

    Returns
    -------
    dict | list
        Loaded JSON object.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    return data