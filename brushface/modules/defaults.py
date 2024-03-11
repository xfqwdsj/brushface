from typing import Type

from brushface.models.abstract.detector import Detector
from brushface.models.abstract.recognizer import Recognizer
from brushface.models.facenet import FaceNet512dClient
from brushface.models.yolo import YoloClient

default_detector: Type[Detector] = YoloClient

default_recognizer: Type[Recognizer] = FaceNet512dClient
