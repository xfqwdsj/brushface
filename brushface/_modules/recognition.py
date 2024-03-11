import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from brushface._modules import detection, representation
from brushface._modules.type import (
    _Detector,
    _Img,
    _Path,
    _Recognizer,
    extract_type,
    get_path,
)
from brushface.algorithm import CosineDistance, DistanceCalculator, default_normalizer
from brushface.internal import Logger
from brushface.internal.hash import find_hash_of_file
from brushface.models import default_detector, default_recognizer

logger = Logger(__name__)


def find(
    img: _Img,
    db_path: _Path,
    normalizer=default_normalizer,
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    recognizer: _Recognizer = default_recognizer,
    distance_calculator: DistanceCalculator = CosineDistance(),
    threshold: Optional[float] = None,
    silent=False,
) -> List[pd.DataFrame]:
    """
    Identifies individuals in the database.

    Each returned dataframe corresponds to the identity information for an individual detected in the source image.

    The DataFrame columns include:

    - 'identity': Identity label of the detected individual.

    - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the target face in the database.

    - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the detected face in the source image.

    - 'threshold': Threshold to determine a pair whether same person or different persons.

    - 'distance': Similarity score between the faces based on the specified model and distance metric.

    Args:
        img: The path or URL to the image, a numpy array in BGR format, or a base64 encoded image.
        db_path: Path to the folder containing image files.
            All detected faces in the database will be considered in the decision-making process.
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
        A list of pandas dataframes.
    """

    tic = time.time()

    db_path = get_path(db_path)

    if not db_path.is_dir():
        raise ValueError(f"{db_path} is not a valid directory.")

    detector = extract_type(detector)
    recognizer = extract_type(recognizer)

    target_size = recognizer.input_shape

    file_name = f"ds_{recognizer.name}_{detector.name}_v2.pkl"
    file_name = file_name.replace("-", "").lower()
    datastore_path = db_path / file_name

    # required columns for representations
    df_cols = [
        "identity",
        "hash",
        "embedding",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    # Ensure the proper pickle file exists
    if not datastore_path.exists():
        with open(datastore_path, "wb") as f:
            pickle.dump([], f)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    # check each item of representations list has required keys
    for i, current_representation in enumerate(representations):
        missing_keys = list(set(df_cols) - set(current_representation.keys()))
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # embedded images
    pickled_images = [x["identity"] for x in representations]

    # Get the list of images on storage
    storage_images = list_images(path=db_path)

    if len(storage_images) == 0:
        raise ValueError(f"No item found in {db_path}")

    # Enforce data consistency amongst on disk images and pickle file
    must_save_pickle = False
    new_images = list(
        set(storage_images) - set(pickled_images)
    )  # images added to storage
    old_images = list(
        set(pickled_images) - set(storage_images)
    )  # images removed from storage

    # detect replaced images
    replaced_images = []
    for current_representation in representations:
        identity = current_representation["identity"]
        if identity in old_images:
            continue
        alpha_hash = current_representation["hash"]
        beta_hash = find_hash_of_file(identity)
        if alpha_hash != beta_hash:
            logger.debug(
                f"Even though {identity} represented before, it's replaced later."
            )
            replaced_images.append(identity)

    if not silent and (
        len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0
    ):
        logger.info(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s)."
        )

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images = new_images + replaced_images
    old_images = old_images + replaced_images

    # remove old images first
    if len(old_images) > 0:
        representations = [
            rep for rep in representations if rep["identity"] not in old_images
        ]
        must_save_pickle = True

    # find representations for new images
    if len(new_images) > 0:
        representations += find_bulk_embeddings(
            employees=new_images,
            recognizer=recognizer,
            target_size=target_size,
            detector=detector,
            enforce_detection=enforce_detection,
            align=align,
            normalizer=normalizer,
            silent=silent,
        )  # add new images
        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)
        if not silent:
            logger.info(
                f"There are now {len(representations)} representations in {file_name}"
            )

    # Should we have no representations bailout
    if len(representations) == 0:
        if not silent:
            toc = time.time()
            logger.info(f"find function duration {toc - tic} seconds")
        return []

    # ----------------------------
    # now, we got representations for facial database
    df = pd.DataFrame(representations)

    if silent is False:
        logger.info(f"Searching {img} in {df.shape[0]} length datastore")

    # img path might have more than once face
    source_objs = detection.extract_faces(
        img=img,
        target_size=target_size,
        grayscale=False,
        detector=detector,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    resp_obj = []

    for source_obj in source_objs:
        source_img = source_obj["img"]
        source_region = source_obj["face_area"]
        target_embedding_obj = representation.represent(
            img=source_img,
            normalizer=normalizer,
            detector=None,
            enforce_detection=enforce_detection,
            align=align,
            recognizer=recognizer,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = source_region["x"]
        result_df["source_y"] = source_region["y"]
        result_df["source_w"] = source_region["w"]
        result_df["source_h"] = source_region["h"]

        distances = []
        for _, instance in df.iterrows():
            source_representation = instance["embedding"]
            if source_representation is None:
                distances.append(float("inf"))  # no representation for this image
                continue

            target_dims = len(list(target_representation))
            source_dims = len(list(source_representation))
            if target_dims != source_dims:
                raise ValueError(
                    f"Source and target embeddings must have same dimensions but {target_dims}:{source_dims}. Model "
                    f"structure may change after pickle created. Delete the {file_name} and re-run."
                )

            distance = distance_calculator(source_representation, target_representation)

            distances.append(distance)

            # ---------------------------
        target_threshold = threshold or getattr(
            recognizer.threshold, distance_calculator.name
        )

        result_df["threshold"] = target_threshold
        result_df["distance"] = distances

        result_df = result_df.drop(columns=["embedding"])

        result_df = result_df[result_df["distance"] <= target_threshold]
        result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(
            drop=True
        )

        resp_obj.append(result_df)

    # -----------------------------------

    if not silent:
        toc = time.time()
        logger.info(f"find function duration {toc - tic} seconds")

    return resp_obj


def list_images(path: _Path) -> List[Path]:
    """Lists images in a given path."""

    path = get_path(path)
    images = []
    for r, _, f in path.walk():
        for file in f:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                exact_path = r / file
                images.append(exact_path)
    return images


def find_bulk_embeddings(
    employees: List[str],
    recognizer: _Recognizer = default_recognizer,
    target_size: Tuple[int, int] = (224, 224),
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    normalizer=default_normalizer,
    silent=False,
) -> List[Dict[str, Any]]:
    """
    Find embeddings of a list of images

    Args:
        employees (list): list of exact image paths

        recognizer: facial recognition model

        target_size (tuple): expected input shape of facial recognition model

        detector: face detector model

        enforce_detection (bool): set this to False if you
            want to proceed when you cannot detect any face

        align (bool): enable or disable alignment of image
            before feeding to facial recognition model

        expand_percentage (int): expand detected facial area with a
            percentage (default is 0).

        normalizer: normalization technique

        silent (bool): enable or disable informative logging
    Returns:
        representations (list): pivot list of dict with
            image name, hash, embedding and detected face area's coordinates
    """

    detector = extract_type(detector)
    recognizer = extract_type(recognizer)

    representations = []
    for employee in tqdm(employees, desc="Finding representations", disable=silent):
        file_hash = find_hash_of_file(employee)

        try:
            img_objs = detection.extract_faces(
                img=employee,
                target_size=target_size,
                grayscale=False,
                detector=detector,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
            )

        except ValueError as err:
            logger.error(
                f"Exception while extracting faces from {employee}: {str(err)}"
            )
            img_objs = []

        if len(img_objs) == 0:
            representations.append(
                {
                    "identity": employee,
                    "hash": file_hash,
                    "embedding": None,
                    "target_x": 0,
                    "target_y": 0,
                    "target_w": 0,
                    "target_h": 0,
                }
            )
        else:
            for img_obj in img_objs:
                img_content = img_obj["img"]
                img_region = img_obj["face_area"]
                embedding_obj = representation.represent(
                    img=img_content,
                    normalizer=normalizer,
                    detector=None,
                    enforce_detection=enforce_detection,
                    align=align,
                    recognizer=recognizer,
                )

                img_representation = embedding_obj[0]["embedding"]
                representations.append(
                    {
                        "identity": employee,
                        "hash": file_hash,
                        "embedding": img_representation,
                        "target_x": img_region["x"],
                        "target_y": img_region["y"],
                        "target_w": img_region["w"],
                        "target_h": img_region["h"],
                    }
                )

    return representations
