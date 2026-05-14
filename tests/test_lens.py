from conftest import System

# ==========================================
# Tests: Single-Target Lens Operations
# ==========================================

def test_lens_get(base_system: System):
    """Test retrieving a value deep in the tree."""
    val = base_system.at.main_comp.value.get()
    assert val == 10.0

def test_lens_set_immutability(base_system: System):
    """Test setting an attribute doesn't mutate the original tree."""
    new_system = base_system.at.main_comp.value.set(99.0)
    
    assert new_system.main_comp.value == 99.0
    assert base_system.main_comp.value == 10.0 # Original is unchanged
    assert new_system.main_comp is not base_system.main_comp

def test_lens_getitem_index(base_system: System):
    """Test updating an item inside a list using indexing."""
    new_system = base_system.at.comp_list[0].value.apply(lambda x: x * 10)
    
    assert new_system.comp_list[0].value == 10.0
    assert new_system.comp_list[1].value == 2.0 # Other list item unchanged

def test_lens_getitem_string_attr(base_system: System):
    """Test using string keys in __getitem__ to access attributes."""
    attr_name = "backup_comp"
    new_system = base_system.at[attr_name].value.set(42.0)
    assert new_system.backup_comp.value == 42.0

def test_lens_filter_attributes(base_system: System):
    """Test filtering attributes of a class (Lens.filter)."""
    from conftest import Component
    
    # Freeze all Component instances at the root of System
    new_sys = base_system.at.where(lambda x: isinstance(x, Component)).is_active.set(False)
    
    assert new_sys.main_comp.is_active is False
    assert new_sys.backup_comp.is_active is False
    
    # comp_list is a list, not a Component, so items inside shouldn't be affected
    assert new_sys.comp_list[0].is_active is True 

# ==========================================
# Tests: Lens String Path Extensions
# ==========================================

def test_lens_path_string_simple(base_system: System):
    """Test updating via a simple string path."""
    new_system = base_system.at.path(".main_comp.value").set(50.0)
    assert new_system.main_comp.value == 50.0

def test_lens_path_string_complex(base_system: System):
    """Test updating via a complex string path containing indices and keys."""
    new_system = base_system.at.path(".comp_list[1].value").apply(lambda x: x * 2)
    new_system = new_system.at.path(".comp_dict['D1'].value").set(0.0)
    
    assert new_system.comp_list[1].value == 4.0
    assert new_system.comp_dict["D1"].value == 0.0