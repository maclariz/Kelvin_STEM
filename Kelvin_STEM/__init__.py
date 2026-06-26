import Kelvin_STEM.io
import Kelvin_STEM.process
import Kelvin_STEM.viz
import Kelvin_STEM.maths

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("Kelvin_STEM")
except PackageNotFoundError:
    __version__ = "unknown"
