import os

os.environ["KERAS_BACKEND"] = "torch"

from ._brushface import cli
from ._modules.analysis import analyze
from ._modules.detection import extract_faces
from ._modules.recognition import find
from ._modules.representation import represent
from ._modules.verification import verify

__version__ = "0.1.0"
