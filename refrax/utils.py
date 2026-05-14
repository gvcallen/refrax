import re
from refrax.custom_types import PathStep

def parse_string_path(path_str: str) -> list[PathStep]:
    """Converts a JAX-style string path (e.g., '.a[0].b' or 'a[0]') into PathSteps."""
    
    # Allow no dot at start
    if path_str and not path_str.startswith(('.', '[')):
        path_str = '.' + path_str

    # Matches: .attribute_name OR [integer] OR ['string_key']
    pattern = re.compile(r"\.([a-zA-Z_]\w*)|\[(-?\d+)\]|\[['\"](.*?)['\"]\]")
    
    steps: list[PathStep] = []
    for match in pattern.finditer(path_str):
        attr_name, list_idx, dict_key = match.groups()
        
        if attr_name is not None:
            steps.append(("attr", attr_name))
        elif list_idx is not None:
            steps.append(("item", int(list_idx)))
        elif dict_key is not None:
            steps.append(("item", dict_key))
            
    return steps