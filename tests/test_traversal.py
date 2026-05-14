import pytest
from conftest import System

# ==========================================
# Tests: Multi-Target Traversal (Select & Path)
# ==========================================

def test_traversal_select_get(base_system: System):
    """Test retrieving multiple targets simultaneously."""
    results = base_system.at.select("main_comp", "backup_comp").value.get()
    assert results == [10.0, 5.0]

def test_traversal_select_string_paths(base_system: System):
    """Test that select correctly parses JAX-style string paths."""
    new_sys = base_system.at.select(".comp_list[0]", ".comp_list[1]").value.set(0.0)
    
    assert new_sys.comp_list[0].value == 0.0
    assert new_sys.comp_list[1].value == 0.0

def test_traversal_path_string(base_system: System):
    """Test appending a string path to multiple diverging branches."""
    # Target both main_comp and backup_comp, then string path down to their values
    new_sys = base_system.at.select("main_comp", "backup_comp").path(".value").apply(lambda x: x + 1.0)
    
    assert new_sys.main_comp.value == 11.0
    assert new_sys.backup_comp.value == 6.0

def test_traversal_select_apply(base_system: System):
    """Test applying a function to multiple distinct attributes."""
    new_sys = base_system.at.select("main_comp", "backup_comp").value.apply(lambda x: x + 1.0)
    
    assert new_sys.main_comp.value == 11.0
    assert new_sys.backup_comp.value == 6.0

# ==========================================
# Tests: Collections (Each)
# ==========================================

def test_traversal_each_list(base_system: System):
    """Test iterating over a list."""
    new_sys = base_system.at.comp_list.each().value.apply(lambda x: x * 2)
    
    assert new_sys.comp_list[0].value == 2.0
    assert new_sys.comp_list[1].value == 4.0

def test_traversal_each_dict(base_system: System):
    """Test iterating over dictionary values."""
    new_sys = base_system.at.comp_dict.each().value.set(0.0)
    
    assert new_sys.comp_dict["D1"].value == 0.0
    assert new_sys.comp_dict["D2"].value == 0.0

def test_each_type_error(base_system: System):
    """Ensure .each() throws an error on non-collections."""
    with pytest.raises(TypeError, match="Cannot iterate"):
        base_system.at.main_comp.each()

# ==========================================
# Tests: Conditional Targeting (Filter & Chain)
# ==========================================

def test_traversal_filter_items(base_system: System):
    """Test filtering items within a collection (Traversal.filter)."""
    # Set value to 999 only for ACTIVE components in the dictionary
    new_sys = (
        base_system.at.comp_dict.each()
        .filter(lambda comp: comp.is_active)
        .value.set(999.0)
    )
    
    assert new_sys.comp_dict["D1"].value == 100.0 # Was inactive, unchanged
    assert new_sys.comp_dict["D2"].value == 999.0 # Was active, updated

def test_complex_deep_chaining(base_system: System):
    """Test combining each, filter, select, and deep attribute access."""
    
    def double_if_active(sys: System) -> System:
        return (
            sys.at.comp_list.each()
            .filter(lambda c: c.is_active)
            .value.apply(lambda x: x * 2)
        )
    
    new_sys = double_if_active(base_system)
    assert new_sys.comp_list[0].value == 2.0
    assert new_sys.comp_list[1].value == 4.0