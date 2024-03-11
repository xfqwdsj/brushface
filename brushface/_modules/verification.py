import time
from typing import List, Optional, Tuple, Union

import numpy as np

from brushface import extract_faces, represent
from brushface._modules.type import _Detector, _Img, _Recognizer, extract_type
from brushface.algorithm import (
    CosineDistance,
    DistanceCalculator,
    Normalizer,
    default_normalizer,
)
from brushface.data.face_area import FaceArea
from brushface.data.verification_result import VerificationResult
from brushface.internal import Logger
from brushface.models import default_detector, default_recognizer
from brushface.models.abstract import Detector, Recognizer

logger = Logger(__name__)


def verify(
    img1: Union[_Img, List[float]],
    img2: Union[_Img, List[float]],
    normalizer=default_normalizer,
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    recognizer: _Recognizer = default_recognizer,
    distance_calculator: DistanceCalculator = CosineDistance(),
    threshold: Optional[float] = None,
    silent=False,
) -> VerificationResult:
    """
    Verifies if an image pair represents the same person or different persons.

    The verification function converts face images to vectors and calculates the similarity
    between those vectors. Vectors of images of the same person should exhibit higher similarity
    (or lower distance) than vectors of images of different persons.

    Args:
        img1: The path or URL to the first image, a numpy array in BGR format, a base64 encoded image,
            or pre-calculated embeddings.
        img2: The path or URL to the second image, a numpy array in BGR format, a base64 encoded image,
            or pre-calculated embeddings.
        normalizer: The normalizer to use for preprocessing the image.
        detector: The face detector to use.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        expand_percentage: The percentage to expand the detected face area.
        recognizer: The face recognizer to use.
        distance_calculator: The distance calculator to use for comparing embeddings.
        threshold: The threshold to determine whether a pair represents the same person or different individuals.
            This threshold is used for comparing distances. If left unset, default pre-tuned threshold values
            will be applied based on the specified model name and distance metric.
        silent: Suppress some log messages for a quieter process.

    Returns:
        A dictionary containing verification results.
    """

    tic = time.time()

    detector = extract_type(detector)
    recognizer = extract_type(recognizer)

    face_areas_1, embeddings_1 = extract_faces_and_embeddings(
        img=img1,
        normalizer=normalizer,
        detector=detector,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        recognizer=recognizer,
        silent=silent,
    )

    face_areas_2, embeddings_2 = extract_faces_and_embeddings(
        img=img2,
        normalizer=normalizer,
        detector=detector,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        recognizer=recognizer,
        silent=silent,
    )

    no_face_area: FaceArea = {
        "x": 0,
        "y": 0,
        "w": 0,
        "h": 0,
        "confidence": 0,
        "left_eye": None,
        "right_eye": None,
    }

    distances = []
    face_areas = []

    for idx, img1_embedding in enumerate(embeddings_1):
        for idy, img2_embedding in enumerate(embeddings_2):
            distance = distance_calculator(img1_embedding, img2_embedding)
            distances.append(distance)
            face_areas.append(
                (
                    face_areas_1[idx] or no_face_area,
                    face_areas_2[idy] or no_face_area,
                )
            )

    # find the face pair with minimum distance
    if not threshold:
        threshold = getattr(recognizer.threshold, distance_calculator.name)
    distance = float(min(distances))  # best distance
    face_areas = face_areas[np.argmin(distances)]

    toc = time.time()

    resp_obj: VerificationResult = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "recognizer": recognizer,
        "detector": detector,
        "distance_calculator": distance_calculator,
        "face_areas": list(face_areas),
        "time": round(toc - tic, 2),
    }

    return resp_obj


def extract_faces_and_embeddings(
    img: Union[_Img, List[float]],
    normalizer: Normalizer,
    detector: Detector,
    enforce_detection: bool,
    align: bool,
    expand_percentage: int,
    recognizer: Recognizer,
    silent: bool,
) -> Tuple[Union[List[FaceArea], List[None]], List[List[float]]]:
    """Extract face areas and find corresponding embeddings for given image."""

    if recognizer is None:
        raise ValueError("This should never happen.")

    dims = recognizer.output_shape

    if isinstance(img, list):
        # given image is already pre-calculated embedding
        if not all(isinstance(dim, float) for dim in img):
            raise ValueError(
                "When passing img as a list, ensure that all its items are of type float."
            )

        if not silent:
            logger.warn(
                "You passed pre-calculated embeddings."
                f"Please ensure that embeddings have been calculated for the {recognizer} model."
            )

        if len(img) != dims:
            raise ValueError(
                f"Embeddings of {recognizer} should have {dims} dimensions, "
                f"but given embeddings have {len(img)} dimensions."
            )

        return [None], [img]

    embeddings = []
    face_areas = []

    target_size = recognizer.input_shape

    try:
        faces = extract_faces(
            img=img,
            target_size=target_size,
            grayscale=False,
            detector=detector,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
        )
    except ValueError as err:
        raise ValueError("Exception while processing an image.") from err

    # find embeddings for each face
    for face in faces:
        representation = represent(
            img=face["img"],
            normalizer=normalizer,
            detector=None,
            enforce_detection=enforce_detection,
            align=align,
            recognizer=recognizer,
        )
        # already extracted face given, safe to access its 1st item
        embedding = representation[0]["embedding"]
        embeddings.append(embedding)
        face_areas.append(face["face_area"])

    return face_areas, embeddings
