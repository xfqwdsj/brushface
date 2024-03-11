from keras.layers import (
    Activation,
    Add,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Model

from brushface.internal import Logger
from brushface.internal.file import weights
from brushface.models.abstract import KerasRecognizer, Threshold

logger = Logger(__name__)


class DeepIdClient(KerasRecognizer):
    """
    DeepId model class
    """

    name = "DeepId"
    input_shape = (47, 55)
    output_shape = 160

    def __init__(
        self,
        threshold: Threshold = Threshold(cosine=0.015, euclidean=45, euclidean_l2=0.17),
    ):
        self.model = load_model()
        self.threshold = threshold


def load_model(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
) -> Model:
    """
    Construct DeepId model, download its weights and load
    """

    my_input = Input(shape=(55, 47, 3))

    x = Conv2D(20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3))(
        my_input
    )
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
    x = Dropout(rate=0.99, name="D1")(x)

    x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
    x = Dropout(rate=0.99, name="D2")(x)

    x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
    x = Dropout(rate=0.99, name="D3")(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name="fc11")(x1)

    x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name="fc12")(x2)

    y = Add()([fc11, fc12])
    y = Activation("relu", name="deepid")(y)

    model = Model(inputs=[my_input], outputs=y)

    model.load_weights(weights(url, logger_delegate=logger))

    return model
