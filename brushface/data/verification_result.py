from typing import List, Optional, TypedDict

from brushface.algorithm.distance import DistanceCalculator
from brushface.data.face_area import FaceArea
from brushface.models.abstract.detector import Detector
from brushface.models.abstract.recognizer import Recognizer


class VerificationResult(TypedDict):
    """
    A dictionary containing verification results.

    Attributes:
        verified: Indicates whether the images represent the same person or different persons.
        distance: The measured distance between the face vectors.
        threshold: The threshold used for verification.
            If the distance is below this threshold, the images are considered a match.
        detector: The detector used.
        recognizer: The recognizer used.
        distance_calculator: The similarity metric used for measuring distances.
        face_areas: Rectangular regions of faces in both images.
        time: Time taken for the verification process in seconds.
    """

    verified: bool
    distance: float
    threshold: float
    detector: Optional[Detector]
    recognizer: Recognizer
    distance_calculator: DistanceCalculator
    face_areas: List[FaceArea]
    time: float
