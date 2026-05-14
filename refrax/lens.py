from typing import Any, Callable, Generic, cast
import re

import equinox as eqx
import jax.tree_util as jtu

from refrax.custom_types import TRoot, PathOp, PathStep
from refrax.traversal import Traversal
from refrax.utils import parse_string_path, translate_jax_path

class Lens(Generic[TRoot]):
    """A fluent interface for mutating immutable Equinox PyTrees.

    Uses `eqx.tree_at` under the hood to functionally swap leaves in the PyTree,
    making it completely safe for use inside JAX JIT/vmap boundaries.

    Args:
        tree (TRoot): The root immutable object to be mutated.
        path (list[PathStep] | None): The current traversal path from the root, by default None.
    """
    _tree: TRoot
    _path: list[PathStep]

    def __init__(self, tree: TRoot, path: list[PathStep] | None = None) -> None:
        self._tree = tree
        self._path = path if path is not None else []

    def __getattr__(self, name: str) -> "Lens[TRoot]":
        """Focuses the lens on a named attribute.

        Args:
            name (str): The name of the attribute.

        Returns:
            Lens[TRoot]: A new Lens focused on the specified attribute.
        """
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return Lens(self._tree, self._path + [("attr", name)])

    def __getitem__(self, key: Any) -> "Lens[TRoot]":
        """Focuses the lens on a collection item, dictionary key, or attribute.

        If the key is a string, it is treated as an attribute access 
        (equivalent to `getattr`). Otherwise, it is treated as an item access.

        Args:
            key (Any): The index, key, or attribute name.

        Returns:
            Lens[TRoot]: A new Lens focused on the specified item or attribute.
        """
        op: PathOp = "attr" if isinstance(key, str) else "item"
        return Lens(self._tree, self._path + [(op, key)])
    
    def _get_target_from(self, tree: Any) -> Any:
        """Traverses the recorded path from a specified tree to return the target node.
        
        This is designed to be passed cleanly into `eqx.tree_at`.

        Args:
            tree (Any): The tree to traverse.

        Returns:
            Any: The resolved target node.
        """
        curr: Any = tree
        for op_type, val in self._path:
            if op_type == "attr":
                curr = getattr(curr, cast(str, val))
            elif op_type == "item":
                curr = curr[val]
        return curr    

    def get(self) -> Any:
        """Extracts the currently focused value.

        Returns:
            Any: The value at the end of the Lens path.
        """
        return self._get_target_from(self._tree)

    def set(self, value: Any) -> TRoot:
        """Sets the focused target to a specific value.

        Args:
            value (Any): The new value to assign.

        Returns:
            TRoot: A new instance of the root tree with the updated value.
        """
        if not self._path:
            return cast(TRoot, value)
            
        return eqx.tree_at(self._get_target_from, self._tree, replace=value)
    
    def apply(self, func: Callable[[Any], Any]) -> TRoot:
        """Applies a transformation function to the focused target.

        Args:
            func (Callable[[Any], Any]): The function to transform the current value.

        Returns:
            TRoot: A new instance of the root tree with the updated value.
        """
        if not self._path:
            return cast(TRoot, func(self._tree))
            
        return eqx.tree_at(self._get_target_from, self._tree, replace_fn=func)    
    
    def path(self, target_path: str | tuple) -> "Lens[TRoot]":
        """Advances the Lens focus based on a string path or JAX KeyPath tuple.

        Args:
            target_path (str | tuple): The path string (e.g., '.res.R') or native 
                JAX KeyPath tuple.

        Returns:
            Lens[TRoot]: A new Lens focused on the parsed path.
        """
        if isinstance(target_path, str):
            parsed_steps = parse_string_path(target_path)
        elif isinstance(target_path, tuple):
            parsed_steps = translate_jax_path(target_path)
        else:
            raise TypeError(f"Expected string or JAX path tuple, got {type(target_path).__name__}")
            
        return Lens(self._tree, self._path + parsed_steps)    
    
    def select(self, *paths: str | tuple) -> Traversal[TRoot]:
        """Branches the Lens into a Traversal targeting multiple attributes.
        
        Supports JAX-style string paths (e.g., '.a[0].b') OR native JAX 
        KeyPath tuples returned by `jax.tree_util`.

        Args:
            *paths (str | tuple): A variable number of attribute paths to target.

        Returns:
            Traversal[TRoot]: A Traversal object focused on the specified attributes.
        """
        sub_paths: list[list[PathStep]] = []
        for p in paths:
            if isinstance(p, str):
                sub_paths.append(parse_string_path(p))
            elif isinstance(p, tuple):
                sub_paths.append(translate_jax_path(p))
            else:
                raise TypeError(f"Expected string or JAX path tuple, got {type(p).__name__}")
            
        return Traversal(self._tree, self._path, sub_paths)
    
    def each(self) -> Traversal[TRoot]:
        """Transforms a focus on a collection into a Traversal of its elements.

        Returns:
            Traversal[TRoot]: A Traversal object focused on every item in the target collection.

        Raises:
            TypeError: If the current focus is not a dictionary, list, or tuple.

        Examples:
            >>> new_model = focus(model).sources.each().apply(lambda x: x * 2)
        """
        target = self.get()
        sub_paths: list[list[PathStep]] = []
        
        if isinstance(target, dict):
            sub_paths = [[("item", key)] for key in target.keys()]
        elif isinstance(target, (list, tuple)):
            sub_paths = [[("item", i)] for i in range(len(target))]
        else:
            raise TypeError(f"Cannot iterate over {type(target).__name__} with .each()")
            
        return Traversal(self._tree, self._path, sub_paths)

    def where(self, predicate: Callable[[Any], bool]) -> Traversal[TRoot]:
        """Traverses immediate items/attributes of the current focus that match a condition.

        Args:
            predicate (Callable[[Any], bool]): Returns True if the immediate child 
                should be included in the Traversal.

        Returns:
            Traversal[TRoot]: A Traversal focused on matching immediate children.
        """
        target = self.get()
        sub_paths: list[list[PathStep]] = []
        
        # 1. Standard Python Collections
        if isinstance(target, dict):
            for k, val in target.items():
                if predicate(val):
                    sub_paths.append([("item", k)])
                    
        elif isinstance(target, (list, tuple)):
            for i, val in enumerate(target):
                if predicate(val):
                    sub_paths.append([("item", i)])
                    
        elif hasattr(target, '__dict__'):
            for attr_name, val in vars(target).items():
                if not attr_name.startswith('_'):
                    if predicate(val):
                        sub_paths.append([("attr", attr_name)])
                        
        return Traversal(self._tree, self._path, sub_paths)
    
    def leaves(self, is_leaf: Callable[[Any], bool] | None = None) -> Traversal[TRoot]:
        """Instantly branches the Lens into a Traversal of all leaf nodes (arrays/scalars).
        
        Powered by JAX's C++ backend (tree_leaves_with_path), making it lightning fast.
        """
        target = self.get()
        leaves_with_paths = jtu.tree_leaves_with_path(target, is_leaf=is_leaf)
        
        sub_paths = []
        for jax_path, _val in leaves_with_paths:
            relative_path = translate_jax_path(jax_path)
            sub_paths.append(relative_path)
                
        return Traversal(self._tree, self._path, sub_paths)
    
    
def focus(tree: TRoot) -> Lens[TRoot]:
    """
    Returns a lens focused on self.
    """
    return Lens(tree)