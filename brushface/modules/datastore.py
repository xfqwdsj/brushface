import pickle
from pathlib import Path
from typing import List, Optional

from brushface.algorithm.normalization import Normalizer
from brushface.data.recognition import RecognitionData
from brushface.internal.hash import find_hash_of_file
from brushface.internal.logger import Logger
from brushface.models.abstract import Detector, Recognizer
from brushface.modules._type import (
    _Detector,
    _Path,
    _Recognizer,
    extract_model,
    get_path,
)
from brushface.modules.defaults import default_detector, default_recognizer

logger = Logger(__name__)


def get_datastore_path(
    db_path: Path, detector: Optional[Detector], recognizer: Recognizer
) -> Path:
    """
    Gets the path to the datastore file.

    Args:
        db_path: Path to the directory containing image files.
        detector: The face detector used to name the file.
        recognizer: The face recognizer used to name the file.

    Returns:
        The path to the datastore file.
    """

    if detector is not None:
        detector_name = detector.name
    else:
        detector_name = "no_detector"

    file_name = f"bf_ds_{detector_name}_{recognizer.name}.pkl"
    file_name = file_name.replace("-", "").lower()
    return db_path / file_name


def load_datastore(datastore_path: Path) -> List[RecognitionData]:
    """
    Loads the datastore from the file system without any checks.

    Args:
        datastore_path: Path to the datastore file.

    Returns:
        The loaded datastore.
    """

    with datastore_path.open("rb") as f:
        result = pickle.load(f)
        f.close()

    return result


def load_datastore_or_none(datastore_path: Path) -> Optional[List[RecognitionData]]:
    """
    Loads the datastore from the file system or returns None if the file does not exist.

    Args:
        datastore_path: Path to the datastore file.

    Returns:
        The loaded datastore or None.
    """

    if not datastore_path.exists() or not datastore_path.is_file():
        return None

    return load_datastore(datastore_path)


def load_datastore_or_empty(datastore_path: Path) -> List[RecognitionData]:
    """
    Loads the datastore from the file system or returns an empty list if the file does not exist.

    Args:
        datastore_path: Path to the datastore file.

    Returns:
        The loaded datastore or an empty list.
    """

    return load_datastore_or_none(datastore_path) or []


def save_datastore(datastore_path: Path, datas: List[RecognitionData]):
    """
    Saves the datastore to the file system.

    Args:
        datastore_path: Path to the datastore file.
        datas: The datastore to save.
    """

    with datastore_path.open("wb") as f:
        pickle.dump(datas, f)
        f.close()


def update_datastore(
    db_path: _Path,
    datas: Optional[List[RecognitionData]] = None,
    normalizer: Optional[Normalizer] = None,
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    recognizer: _Recognizer = default_recognizer,
    silent=False,
) -> List[RecognitionData]:
    """
    Updates the datastore.

    Args:
        db_path: Path to the directory containing image files.
        datas: List of dictionaries containing the representation results for detected faces.
        normalizer: The normalizer to use for preprocessing the image.
        detector: The face detector to use.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        recognizer: The face recognizer to use.
        silent: Suppress some log messages for a quieter process.
    """

    db_path = get_path(db_path)

    if not db_path.is_dir():
        raise ValueError(f"{db_path} is not a valid directory.")

    from brushface.modules.recognition import list_images, represent_in_bulk

    found_images = list_images(db_path)

    if len(found_images) == 0:
        raise ValueError(f"No item found in {db_path}.")

    detector = extract_model(detector)
    recognizer = extract_model(recognizer)

    datastore_path = get_datastore_path(db_path, detector, recognizer)

    datas = datas or load_datastore_or_empty(datastore_path)

    new_images = found_images.copy()
    removed_images = 0
    replaced_images = []

    results = []

    for data in datas:
        img_path = Path(data["img_path"])

        if img_path not in found_images:
            removed_images += 1
            continue

        if img_path in new_images:
            new_images.remove(img_path)

        hash1 = data["hash"]
        hash2 = find_hash_of_file(img_path)
        if hash1 != hash2:
            logger.debug(f"The hash of {img_path} has changed from {hash1} to {hash2}.")
            replaced_images.append(img_path)
            continue

        results.append(data)

    if not silent and (
        len(new_images) > 0 or removed_images or len(replaced_images) > 0
    ):
        logger.info(
            f"Found {len(new_images)} newly added images, "
            f"{removed_images} removed images, "
            f"{len(replaced_images)} replaced images."
        )

    new_images = new_images + replaced_images

    if len(new_images) > 0:
        logger.info(f"Updating {len(new_images)} new images.")

        results += represent_in_bulk(
            img_paths=new_images,
            normalizer=normalizer,
            detector=detector,
            enforce_detection=enforce_detection,
            align=align,
            recognizer=recognizer,
            silent=silent,
        )

    save_datastore(datastore_path, results)
    if not silent:
        logger.info(f"There are now {len(results)} datas in datastore.")

    return results
