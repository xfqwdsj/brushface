import time
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from brushface.algorithm.distance import CosineDistance, DistanceCalculator
from brushface.algorithm.normalization import Normalizer
from brushface.data.detected_face import DetectedFace
from brushface.data.recognition import RecognitionData, RecognitionResult
from brushface.internal.hash import find_hash_of_file
from brushface.internal.logger import Logger
from brushface.models.abstract.detector import Detector
from brushface.models.abstract.recognizer import Recognizer
from brushface.modules._type import (
    _Detector,
    _Img,
    _Path,
    _Recognizer,
    extract_model,
    get_path,
)
from brushface.modules.datastore import (
    get_datastore_path,
    load_datastore_or_empty,
    update_datastore,
)
from brushface.modules.defaults import default_detector, default_recognizer
from brushface.modules.detection import extract_faces
from brushface.modules.representation import represent_from_faces

logger = Logger(__name__)


def find_from_faces(
    faces: List[DetectedFace],
    db_path: _Path,
    skip_update=False,
    normalizer: Optional[Normalizer] = None,
    detector: Optional[_Detector] = default_detector,
    recognizer: _Recognizer = default_recognizer,
    distance_calculator: DistanceCalculator = CosineDistance(),
    threshold: Optional[float] = None,
    silent=False,
) -> List[Tuple[DetectedFace, RecognitionResult]]:
    """
    Identifies individuals in the database.

    Args:
        faces: The list of detected faces.
        db_path: Path to the directory containing image files.
        skip_update: If False, the datastore will be updated with new images found in the db_path.
            This function compares the embeddings of the detected faces in the input image
            with the embeddings of the faces in the datastore.
            Only set this to True if you are sure that the datastore is up-to-date.
        normalizer: The normalizer to use for preprocessing the image.
        detector: The face detector to use.
        recognizer: The face recognizer to use.
        distance_calculator: The distance calculator to use for comparing embeddings.
        threshold: The threshold to determine whether a pair represents the same person or different individuals.
            This threshold is used for comparing distances. If left unset, default pre-tuned threshold values
            will be applied based on the specified model name and distance metric.
        silent: Suppress some log messages for a quieter process.

    Returns:
        A list of dictionaries containing the result of face recognition.
    """

    tic = time.time()

    db_path = get_path(db_path)

    if not db_path.is_dir():
        raise ValueError(f"{db_path} is not a valid directory.")

    if detector is not None:
        detector = extract_model(detector)
    recognizer = extract_model(recognizer)

    datastore_path = get_datastore_path(db_path, detector, recognizer)
    file_name = datastore_path.name

    datas = load_datastore_or_empty(datastore_path)

    if not skip_update:
        datas = update_datastore(
            db_path=db_path,
            datas=datas,
            normalizer=normalizer,
            detector=None,
            enforce_detection=False,
            align=False,
            recognizer=recognizer,
            silent=silent,
        )
    elif not silent:
        logger.info(f"Skipping database update for {file_name}.")

    results: List[Tuple[DetectedFace, RecognitionResult]] = []

    if len(datas) == 0:
        if not silent:
            logger.info(f"No data found in {file_name}.")
        return results

    if not silent:
        logger.info(
            f"Searching {len(faces)} faces in datastore containing {len(datas)} elements."
        )

    for face in faces:
        embedding = represent_from_faces(
            faces=[face],
            normalizer=normalizer,
            recognizer=recognizer,
        )[0][1]

        if threshold is None:
            threshold = recognizer.threshold[type(distance_calculator)]

        for data in datas:
            target_embedding = data["embedding"]
            if target_embedding is None:
                continue

            source_dims = len(embedding)
            target_dims = len(target_embedding)
            if source_dims != target_dims:
                raise ValueError(
                    f"Source and target embeddings must have same dimensions but found {source_dims}:{target_dims}. "
                    f"Model structure may change after pickle created. Delete the {file_name} and re-run."
                )

            distance = distance_calculator(embedding, target_embedding)

            if distance <= threshold:
                results.append(
                    (
                        face,
                        {
                            "img_path": data["img_path"],
                            "hash": data["hash"],
                            "target_x": data["x"],
                            "target_y": data["y"],
                            "target_w": data["w"],
                            "target_h": data["h"],
                            "distance": distance,
                            "threshold": threshold,
                        },
                    )
                )

        results.sort(key=lambda x: x[1]["distance"])

    if not silent:
        toc = time.time()
        logger.info(f"Recognition took {toc - tic} seconds.")

    return results


def find(
    img: _Img,
    db_path: _Path,
    skip_update=False,
    normalizer: Optional[Normalizer] = None,
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    recognizer: _Recognizer = default_recognizer,
    distance_calculator: DistanceCalculator = CosineDistance(),
    threshold: Optional[float] = None,
    silent=False,
) -> List[Tuple[DetectedFace, RecognitionResult]]:
    """
    Identifies individuals in the database.

    Args:
        img: The path or URL to the image, a NumPy array in BGR format, or a base64 encoded image.
        db_path: Path to the directory containing image files.
        skip_update: If False, the datastore will be updated with new images found in the db_path.
            This function compares the embeddings of the detected faces in the input image
            with the embeddings of the faces in the datastore.
            Only set this to True if you are sure that the datastore is up-to-date.
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
        A list of dictionaries containing the result of face recognition.
    """

    faces = extract_faces(
        img=img,
        target_size=extract_model(recognizer).input_shape,
        grayscale=False,
        detector=detector,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        return_rgb=True,
    )

    return find_from_faces(
        faces=faces,
        db_path=db_path,
        skip_update=skip_update,
        normalizer=normalizer,
        detector=detector,
        recognizer=recognizer,
        distance_calculator=distance_calculator,
        threshold=threshold,
        silent=silent,
    )


def list_images(path: Path) -> List[Path]:
    """Lists images in a given path."""

    images = []
    for r, _, f in path.walk():
        for file in f:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(r / file)
    return images


def represent_in_bulk(
    img_paths: List[Path],
    normalizer: Optional[Normalizer] = None,
    detector: Optional[Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    recognizer: Recognizer = default_recognizer,
    silent=False,
) -> List[RecognitionData]:
    """
    Finds embeddings of a list of images.

    Args:
        img_paths: List of image paths.
        normalizer: The normalizer to use for preprocessing the image.
        detector: The face detector to use.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        expand_percentage: The percentage to expand the detected face area.
        recognizer: The face recognizer to use.
        silent: Suppress some log messages for a quieter process.
    Returns:
        A list of dictionaries, where each dictionary represents the representation results for a detected face.
    """

    datas: List[RecognitionData] = []

    for img_path in tqdm(img_paths, desc="Finding representations.", disable=silent):
        file_hash = find_hash_of_file(img_path)

        faces: list[DetectedFace] = []

        try:
            faces = extract_faces(
                img=img_path,
                target_size=recognizer.input_shape,
                grayscale=False,
                detector=detector,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
            )
        except ValueError as err:
            logger.error(
                f"Exception while extracting faces from {img_path}: {str(err)}"
            )

        if len(faces) == 0:
            datas = [
                {
                    "img_path": str(img_path),
                    "hash": file_hash,
                    "embedding": None,
                    "x": 0,
                    "y": 0,
                    "w": 0,
                    "h": 0,
                }
            ]
        else:
            for face in faces:
                area = face["face_area"]

                embedding = represent_from_faces(
                    faces=[face],
                    normalizer=normalizer,
                    recognizer=recognizer,
                )[0][1]

                datas.append(
                    {
                        "img_path": str(img_path),
                        "hash": file_hash,
                        "embedding": embedding,
                        "x": area["x"],
                        "y": area["y"],
                        "w": area["w"],
                        "h": area["h"],
                    }
                )

    return datas
