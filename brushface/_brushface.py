import warnings

from brushface.internal import file

warnings.filterwarnings("ignore")

# Create required folders if necessary to store model weights
file.initialize()


def cli() -> None:
    """
    Command line interface function will be offered in this block
    """

    import fire

    fire.Fire()
