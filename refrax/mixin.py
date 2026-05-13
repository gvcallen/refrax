from typing import Self
from refrax.lens import Lens

class OpticsMixin:
    """
    Mixin to add the fluent `.at` property to class.
    """
    @property
    def at(self) -> Lens[Self]:
        return Lens(self)