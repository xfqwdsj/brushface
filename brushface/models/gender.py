import numpy as np

from brushface.internal.logger import Logger
from brushface.models.abstract.analyzer import (
    LabeledAnalyzer,
    LabeledResult,
    VggFaceBasedAnalyzer,
)

logger = Logger(__name__)


class GenderClient(LabeledAnalyzer, VggFaceBasedAnalyzer):
    name = "Gender"
    labels = ["woman", "man"]

    def __init__(self):
        self.model = VggFaceBasedAnalyzer.build_model(
            weights_url="https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
            classes=2,
            logger_delegate=logger,
        )

    def analyze(self, img: np.ndarray) -> LabeledResult:
        return self.get_result(self.predict(img))
