from typing import Self
from refrax.lens import Lens

class OpticsMixin:
    """
    Adds an `.at` property to your class which returns `refrax.Lens` on `self`.
    """
    @property
    def at(self) -> Lens[Self]:
        return Lens(self)