from typing import Type

from brushface.models.abstract import Detector, Recognizer
from .facenet import FaceNet512dClient
from .yolo import YoloClient

default_detector: Type[Detector] = YoloClient
default_recognizer: Type[Recognizer] = FaceNet512dClient
