from src.config import MODEL_PATH
from src.utils import load_model


def get_model() -> object:
    """
    Load the trained machine learning pipeline.

    Returns
    -------
    object
        Trained prediction pipeline.

    Raises
    ------
    FileNotFoundError
        If the trained model file cannot be loaded.
    """

    model = load_model(
        MODEL_PATH
    )

    return model


model = get_model()