from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from refrax.lens import Lens as Lens
from refrax.traversal import Traversal as Traversal
from refrax.mixin import OpticsMixin as OpticsMixin