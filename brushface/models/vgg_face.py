from types import MappingProxyType
from typing import List

import numpy as np
from keras.layers import (
    Activation,
    Convolution2D,
    Dropout,
    Flatten,
    MaxPooling2D,
    ZeroPadding2D,
)
from keras.models import Model, Sequential

from brushface.algorithm.distance import (
    CosineDistance,
    EuclideanDistance,
    EuclideanL2Distance,
)
from brushface.algorithm.l2 import l2_normalize
from brushface.internal.logger import Logger
from brushface.internal.file import get_weights
from brushface.models.abstract.recognizer import Recognizer, Threshold

logger = Logger(__name__)


class VggFaceClient(Recognizer):
    name = "VGG-Face"
    input_shape = (224, 224)
    output_shape = 4096

    def __init__(
        self,
        threshold: Threshold = MappingProxyType(
            {
                CosineDistance: 0.68,
                EuclideanDistance: 1.17,
                EuclideanL2Distance: 1.17,
            }
        ),
    ):
        self.model = self.load_model()
        self.threshold = threshold

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """
        find embeddings with VGG-Face model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        # having normalization layer in descriptor troubles for some gpu users (e.g. issue 957, 966)
        # instead we are now calculating it with traditional way not with keras backend
        embedding = self.model(img, training=False).numpy()[0].tolist()
        embedding = l2_normalize(embedding)
        return embedding.tolist()

    @staticmethod
    def base_model():
        """
        Constructs the base model of the VGG-Face model for classification - not to find embeddings.

        Returns:
            The base model trained to classify 2622 identities.
        """

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def load_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
    ):
        """
        Final VGG-Face model being used for finding embeddings
        Returns:
            model (Model): returning 4096 dimensional vectors
        """

        model = VggFaceClient.base_model()

        model.load_weights(get_weights(url, logger_delegate=logger))

        # 2622d dimensional model
        # vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

        # 4096 dimensional model offers 6% to 14% increasement on accuracy!
        # - softmax causes underfitting
        # - added normalization layer to avoid underfitting with euclidean
        # as described here: https://github.com/serengil/deepface/issues/944
        base_model_output = Flatten()(model.layers[-5].output)
        # keras backend's l2 normalization layer troubles some gpu users (e.g. issue 957, 966)
        # base_model_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(
        #     base_model_output
        # )
        vgg_face_descriptor = Model(inputs=model.inputs, outputs=base_model_output)

        return vgg_face_descriptor
