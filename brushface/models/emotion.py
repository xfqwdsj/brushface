import cv2
import numpy as np
from keras.layers import AveragePooling2D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from brushface.internal.file import get_weights
from brushface.internal.logger import Logger
from brushface.models.abstract.analyzer import (
    Analyzer,
    LabeledAnalyzer,
    LabeledResult,
)

logger = Logger(__name__)


class EmotionClient(LabeledAnalyzer, Analyzer):
    name = "Emotion"
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        self.model = self.build_model()

    def analyze(self, img: np.ndarray) -> LabeledResult:
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)

        emotion_predictions = self.model.predict(img_gray, verbose=0)[0, :]
        return self.get_result(emotion_predictions)

    @staticmethod
    def build_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
    ):
        num_classes = 7

        model = Sequential()

        # 1st convolution layer
        model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # fully connected neural networks
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation="softmax"))

        model.load_weights(get_weights(url, logger_delegate=logger))

        return model
