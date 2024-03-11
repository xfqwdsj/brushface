import warnings

from brushface.internal import file

__all__ = ["cli"]

warnings.filterwarnings("ignore")

file.initialize()


def cli():
    import fire

    fire.Fire()
