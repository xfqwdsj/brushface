from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from brushface._modules import detection
from brushface._modules.type import _Detector, _Img
from brushface.models import (
    ApparentAgeClient,
    EmotionClient,
    GenderClient,
    RaceClient,
    default_detector,
)


class Action(Enum):
    emotion = "emotion"
    age = "age"
    gender = "gender"
    race = "race"


# TODO: TypedDict
def analyze(
    img: _Img,
    actions: List[Action] = (Action.emotion, Action.age, Action.gender, Action.race),
    detector: Optional[_Detector] = default_detector,
    enforce_detection=True,
    align=True,
    expand_percentage=0,
    silent=False,
) -> List[Dict[str, Any]]:
    """
    Analyzes facial attributes such as age, gender, emotion, and race in the provided image.

    Each returned dictionary in the list contains the following keys:

    - 'region' (dict): Represents the rectangular region of the detected face in the image.
       - 'x': x-coordinate of the top-left corner of the face.
       - 'y': y-coordinate of the top-left corner of the face.
       - 'w': Width of the detected face region.
       - 'h': Height of the detected face region.

    - 'age' (float): Estimated age of the detected face.

    - 'face_confidence' (float): Confidence score for the detected face.
        Indicates the reliability of the face detection.

    - 'dominant_gender' (str): The dominant gender in the detected face.
        Either "Man" or "Woman."

    - 'gender' (dict): Confidence scores for each gender category.
       - 'Man': Confidence score for the male gender.
       - 'Woman': Confidence score for the female gender.

    - 'dominant_emotion' (str): The dominant emotion in the detected face.
        Possible values include "sad," "angry," "surprise," "fear," "happy,"
        "disgust," and "neutral."

    - 'emotion' (dict): Confidence scores for each emotion category.
       - 'sad': Confidence score for sadness.
       - 'angry': Confidence score for anger.
       - 'surprise': Confidence score for surprise.
       - 'fear': Confidence score for fear.
       - 'happy': Confidence score for happiness.
       - 'disgust': Confidence score for disgust.
       - 'neutral': Confidence score for neutrality.

    - 'dominant_race' (str): The dominant race in the detected face.
        Possible values include "indian," "asian," "latino hispanic,"
        "black," "middle eastern," and "white."

    - 'race' (dict): Confidence scores for each race category.
       - 'indian': Confidence score for Indian ethnicity.
       - 'asian': Confidence score for Asian ethnicity.
       - 'latino hispanic': Confidence score for Latino/Hispanic ethnicity.
       - 'black': Confidence score for Black ethnicity.
       - 'middle eastern': Confidence score for Middle Eastern ethnicity.
       - 'white': Confidence score for White ethnicity.

    Args:
        img: The path or URL to the image, a numpy array in BGR format, or a base64 encoded image.
        actions: Attributes to analyze.
        detector: Face detector.
        enforce_detection: If no face is detected in an image, raise an exception.
            Not enforcing detection can be useful for low-resolution images.
        align: Perform alignment based on the eye positions.
        expand_percentage: The percentage to expand the detected face area.
        silent: Suppress some log messages for a quieter process.

    Returns:
        A list of dictionaries, where each dictionary represents the analysis results for a detected face.
    """

    resp_objects = []

    img_objs = detection.extract_faces(
        img=img,
        target_size=(224, 224),
        grayscale=False,
        detector=detector,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    emotion_model = EmotionClient()
    age_model = ApparentAgeClient()
    gender_model = GenderClient()
    race_model = RaceClient()

    for img_obj in img_objs:
        img_content = img_obj["img"]
        img_region = img_obj["face_area"]
        img_confidence = img_region["confidence"]
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {}
            # facial attribute analysis
            pbar = tqdm(
                range(0, len(actions)),
                desc="Finding actions",
                disable=silent if len(actions) > 1 else True,
            )
            for index in pbar:
                action = actions[index]
                pbar.set_description(f"Action: {action}")

                if action == Action.emotion:
                    emotion_predictions = emotion_model.predict(img_content)
                    sum_of_predictions = emotion_predictions.sum()

                    obj["emotion"] = {}
                    for i, emotion_label in enumerate(EmotionClient.labels):
                        emotion_prediction = (
                            100 * emotion_predictions[i] / sum_of_predictions
                        )
                        obj["emotion"][emotion_label] = emotion_prediction

                    obj["dominant_emotion"] = EmotionClient.labels[
                        np.argmax(emotion_predictions)
                    ]

                elif action == Action.age:
                    apparent_age = age_model.predict(img_content)
                    # int cast is for exception - object of type 'float32' is not JSON serializable
                    obj["age"] = int(apparent_age)

                elif action == Action.gender:
                    gender_predictions = gender_model.predict(img_content)
                    obj["gender"] = {}
                    for i, gender_label in enumerate(GenderClient.labels):
                        gender_prediction = 100 * gender_predictions[i]
                        obj["gender"][gender_label] = gender_prediction

                    obj["dominant_gender"] = GenderClient.labels[
                        np.argmax(gender_predictions)
                    ]

                elif action == Action.race:
                    race_predictions = race_model.predict(img_content)
                    sum_of_predictions = race_predictions.sum()

                    obj["race"] = {}
                    for i, race_label in enumerate(RaceClient.labels):
                        race_prediction = 100 * race_predictions[i] / sum_of_predictions
                        obj["race"][race_label] = race_prediction

                    obj["dominant_race"] = RaceClient.labels[
                        np.argmax(race_predictions)
                    ]

                # mention facial areas
                obj["region"] = img_region
                # include image confidence
                obj["face_confidence"] = img_confidence

            resp_objects.append(obj)

    return resp_objects
