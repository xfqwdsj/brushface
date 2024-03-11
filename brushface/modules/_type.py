from pathlib import Path
from typing import Type, TypeVar, Union

import numpy as np

from brushface.models.abstract.analyzer import Analyzer
from brushface.models.abstract.detector import Detector
from brushface.models.abstract.model import BrushFaceModel
from brushface.models.abstract.recognizer import Recognizer

type _Path = Union[str, Path]
type _Img = Union[_Path, np.ndarray]

type _Detector = Union[Type[Detector], Detector]
type _Recognizer = Union[Type[Recognizer], Recognizer]
type _Analyzer = Union[Type[Analyzer], Analyzer]


def get_path(target: _Path) -> Path:
    if isinstance(target, Path):
        return target
    return Path(target)


MODEL_TYPE = TypeVar("MODEL_TYPE", bound=BrushFaceModel)


def extract_model(target: Union[Type[MODEL_TYPE], MODEL_TYPE]) -> MODEL_TYPE:
    if target is None:
        raise ValueError("This should never happen. The target cannot be None.")

    if isinstance(target, type):
        # noinspection PyUnresolvedReferences
        return target.default()
    return target
