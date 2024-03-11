import numpy as np
from keras.layers import Activation, Convolution2D, Flatten
from keras.models import Model

from brushface.internal import Logger
from brushface.internal.file import weights
from brushface.models.abstract import Analyzer
from .vgg_face import VggFaceClient

logger = Logger(__name__)


class GenderClient(Analyzer):
    """
    Gender model class
    """

    # Labels for the genders that can be detected by the model.
    labels = ["Woman", "Man"]

    def __init__(self):
        self.model = load_model()
        self.name = "Gender"

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img, verbose=0)[0, :]


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
) -> Model:
    """
    Construct gender model, download its weights and load
    Returns:
        model (Model)
    """

    model = VggFaceClient.base_model()

    classes = 2
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
        model.layers[-4].output
    )
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    gender_model.load_weights(weights(url, logger_delegate=logger))

    return gender_model
