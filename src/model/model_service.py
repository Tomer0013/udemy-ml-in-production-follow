"""
This module provides functionality for managing an ML model.

It contains the ModelService class, which handles loading and using
a pre-trained ML model. This class offers methods to load a model
from a file, building it if it doesn't exist, and to make predictions
using the loaded model.
"""

import pickle
from pathlib import Path

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelService:
    """
    A service class for managing the ML model.

    This class provides functionalities to load an ML model from
    a specified path, build it if it doesn't exist, and make
    predictions using the loaded model.

    Attributes:
        model: ML model managed by the service. Initially set to None.

    Methods:
        __init__: Constructor that initializes the ModelService.
        load_model: Loads the model from the file or builds if it doesn't exist.
        predict: Makes a prediction using the loaded model.
    """

    def __init__(self) -> None:
        """Initializes the ModelService with no model loaded."""
        self.model = None

    def load_model(self) -> None:
        """Loads the model from the file or builds if it doesn't exist."""
        model_path = Path(model_settings.model_path) / model_settings.model_name
        logger.info(f"Checking the existence of model config file at {model_path}")
        if not model_path.exists():
            logger.warning(
                f"Model at {model_path} was not found -> Building {model_settings.model_name}"
            )
            build_model()

        logger.info(
            f"Model {model_settings.model_name} exists! -> Loading model configuration file"
        )
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, input_parameters: list) -> list:
        """
        Makes a prediction using the loaded model.

        Takes input parameters and passes it to the model, which was loaded using a pickle file.

        Args:
            input_parameters (list): The input data for making a prediction.

        Returns:
            list: The prediction result from the model.
        """
        logger.info("Making a prediction")

        return self.model.predict([input_parameters])  # type: ignore
