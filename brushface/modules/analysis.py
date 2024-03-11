from typing import Dict, List, Optional, Sequence, Tuple, Type

from tqdm import tqdm

from brushface.data.detected_face import DetectedFace
from brushface.models.abstract.analyzer import Analyzer, AnalyzerResult
from brushface.models.apparent_age import ApparentAgeClient
from brushface.models.emotion import EmotionClient
from brushface.models.gender import GenderClient
from brushface.models.race import RaceClient
from brushface.modules.defaults import default_detector
from brushface.modules.detection import extract_faces
from brushface.modules._type import _Analyzer, _Detector, _Img, extract_model


def analyze_from_faces(
    faces: List[DetectedFace],
    analyzers: Sequence[_Analyzer] = (
        EmotionClient,
        ApparentAgeClient,
        GenderClient,
        RaceClient,
    ),
    silent=False,
):
    """
    Analyzes facial attributes from given faces.

    Args:
        faces: The list of detected faces.
        analyzers: The list of models to use for analysis.
        silent: Suppress some log messages for a quieter process.

    Returns:
        A list of dictionaries, where each dictionary represents the analysis results for a detected face.
    """

    analysis_results: List[
        Tuple[DetectedFace, Dict[Type[Analyzer], AnalyzerResult]]
    ] = []

    for face in faces:
        img = face["img"]
        if img.shape[0] > 0 and img.shape[1] > 0:
            results = {}

            pbar = tqdm(
                range(0, len(analyzers)),
                desc="Analyzing",
                disable=silent if len(analyzers) > 1 else True,
            )
            for index in pbar:
                analyzer = extract_model(analyzers[index])
                pbar.set_description(f"Using analyzer: {analyzer}")

                results[type(analyzer)] = analyzer.analyze(img)

            analysis_results.append((face, results))

    return analysis_results


def analyze(
    img: _Img,
    analyzers: Sequence[_Analyzer] = (
        EmotionClient,
        ApparentAgeClient,
        GenderClient,
        RaceClient,
    ),
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    silent=False,
):
    """
    Analyzes facial attributes from given image.

    Args:
        img: The path or URL to the image, a NumPy array in BGR format, or a base64 encoded image.
        analyzers: The list of models to use for analysis.
        detector: Face detector.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        expand_percentage: The percentage to expand the detected face area.
        silent: Suppress some log messages for a quieter process.

    Returns:
        A list of dictionaries, where each dictionary represents the analysis results for a detected face.
    """

    faces = extract_faces(
        img=img,
        target_size=(224, 224),
        grayscale=False,
        detector=detector,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        return_rgb=False,
    )

    return analyze_from_faces(faces, analyzers, silent)
