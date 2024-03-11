import numpy as np
from keras.layers import Activation, Convolution2D, Flatten
from keras.models import Model

from brushface.internal import Logger
from brushface.internal.file import weights
from brushface.models.abstract import Analyzer
from .vgg_face import VggFaceClient

logger = Logger(__name__)


class RaceClient(Analyzer):
    """
    Race model class
    """

    # Labels for the ethnic phenotypes that can be detected by the model.
    labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

    def __init__(self):
        self.model = load_model()
        self.name = "Race"

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img, verbose=0)[0, :]


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
) -> Model:
    """
    Construct race model, download its weights and load
    """

    model = VggFaceClient.base_model()

    classes = 6
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
        model.layers[-4].output
    )
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    race_model = Model(inputs=model.input, outputs=base_model_output)

    race_model.load_weights(weights(url, logger_delegate=logger))

    return race_model
