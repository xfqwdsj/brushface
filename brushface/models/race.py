import numpy as np

from brushface.internal.logger import Logger
from brushface.models.abstract.analyzer import (
    LabeledAnalyzer,
    LabeledResult,
    VggFaceBasedAnalyzer,
)

logger = Logger(__name__)


class RaceClient(LabeledAnalyzer, VggFaceBasedAnalyzer):
    name = "Race"
    labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

    def __init__(self):
        self.model = VggFaceBasedAnalyzer.build_model(
            weights_url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
            classes=6,
            logger_delegate=logger,
        )

    def analyze(self, img: np.ndarray) -> LabeledResult:
        return self.get_result(self.predict(img))
