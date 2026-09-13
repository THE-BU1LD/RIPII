from importlib.metadata import PackageNotFoundError, version

from .models.ripii import RIPIIModel

try:
    __version__ = version("ripii")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["RIPIIModel", "__version__"]
