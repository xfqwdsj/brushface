import typing

import numpy as np
from keras.layers import Activation, Convolution2D, Flatten
from keras.models import Model

from brushface.internal import Logger
from brushface.internal.file import weights
from brushface.models.abstract import Analyzer
from .vgg_face import VggFaceClient

logger = Logger(__name__)


class ApparentAgeClient(Analyzer):
    """
    Age model class
    """

    def __init__(self):
        self.model = load_model()
        self.name = "Age"

    def predict(self, img: np.ndarray) -> np.float64:
        age_predictions = self.model.predict(img, verbose=0)[0, :]
        return find_apparent_age(age_predictions)


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
) -> Model:
    """
    Construct age model, download its weights and load
    Returns:
        model (Model)
    """

    model = VggFaceClient.base_model()

    # --------------------------

    classes = 101

    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
        model.layers[-4].output
    )
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # load weights

    age_model.load_weights(weights(url, logger_delegate=logger))

    return age_model


def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (?)
    Returns:
        apparent_age (float)
    """

    output_indexes = np.array(list(range(0, 101)))
    apparent_age = typing.cast(np.float64, np.sum(age_predictions * output_indexes))
    return apparent_age
