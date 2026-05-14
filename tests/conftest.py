import equinox as eqx
import pytest
from refrax import Lens

class Component(eqx.Module):
    name: str
    value: float
    is_active: bool = True

    def __post_init__(self):
        # A simple check to ensure Equinox validation runs during rebuilds
        if self.value < 0:
            raise ValueError("Component value cannot be negative.")

class System(eqx.Module):
    main_comp: Component
    backup_comp: Component
    comp_list: list[Component]
    comp_dict: dict[str, Component]
    
    @property
    def at(self) -> Lens["System"]:
        return Lens(self)

@pytest.fixture
def base_system() -> System:
    return System(
        main_comp=Component(name="Main", value=10.0),
        backup_comp=Component(name="Backup", value=5.0),
        comp_list=[
            Component(name="L1", value=1.0),
            Component(name="L2", value=2.0)
        ],
        comp_dict={
            "D1": Component(name="D1", value=100.0, is_active=False),
            "D2": Component(name="D2", value=200.0, is_active=True)
        }
    )