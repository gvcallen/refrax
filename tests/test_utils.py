from refrax.utils import parse_string_path

def test_parse_string_path_attributes():
    path = parse_string_path(".res.R")
    assert path == [("attr", "res"), ("attr", "R")]

def test_parse_string_path_indices():
    path = parse_string_path("[0][-1]")
    assert path == [("item", 0), ("item", -1)]

def test_parse_string_path_dict_keys():
    path = parse_string_path("['key_1'][\"key2\"]")
    assert path == [("item", "key_1"), ("item", "key2")]

def test_parse_string_path_complex():
    path = parse_string_path(".cascade[0]['value'].is_active")
    assert path == [
        ("attr", "cascade"), 
        ("item", 0), 
        ("item", "value"), 
        ("attr", "is_active")
    ]