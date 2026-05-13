from typing import Generic, Any, Callable, cast

import equinox as eqx

from refrax.custom_types import TRoot, PathStep, PathOp


class Traversal(Generic[TRoot]):
    """Represents a multi-target focus within an immutable PyTree.

    Applies mutations across all targets simultaneously using `eqx.tree_at`, 
    making it fully compatible with JAX JIT compilation and tracing.

    Args:
        tree (TRoot): The root immutable object (e.g., Equinox module or dataclass).
        base_path (list[PathStep]): The shared path from the root to the divergence point.
        sub_paths (list[list[PathStep]]): A list of diverging paths, one for each targeted element.
    """
    def __init__(self, tree: TRoot, base_path: list[PathStep], sub_paths: list[list[PathStep]]) -> None:
        self._tree = tree
        self._base_path = base_path
        self._sub_paths = sub_paths

    def __getattr__(self, name: str) -> "Traversal[TRoot]":
        """Broadens the traversal by appending an attribute access to every currently focused target.

        Args:
            name (str): The name of the attribute to access on all targets.

        Returns:
            Traversal[TRoot]: A new Traversal focused one level deeper.
        """
        new_sub_paths = [path + [("attr", name)] for path in self._sub_paths]
        return Traversal(self._tree, self._base_path, new_sub_paths)

    def __getitem__(self, key: Any) -> "Traversal[TRoot]":
        """Broadens the traversal by appending an item/index access to every currently focused target. 
        
        Strings are treated as attributes.

        Args:
            key (Any): The index, dictionary key, or attribute string to access.

        Returns:
            Traversal[TRoot]: A new Traversal focused one level deeper.
        """
        op: PathOp = "attr" if isinstance(key, str) else "item"
        new_sub_paths = [path + [(op, key)] for path in self._sub_paths]
        return Traversal(self._tree, self._base_path, new_sub_paths)

    def _get_targets_from(self, tree: Any) -> tuple[Any, ...]:
        """Internal generator for eqx.tree_at. 
        
        Dynamically walks the paths to return all target nodes from a given root tree 
        (or Equinox tracer).
        
        Args:
            tree (Any): The root tree or Equinox tracer to walk.
            
        Returns:
            tuple[Any, ...]: A tuple of all resolved target nodes.
        """
        targets = []
        for path in self._sub_paths:
            curr = tree
            for op, val in self._base_path + path:
                if op == "attr":
                    curr = getattr(curr, cast(str, val))
                elif op == "item":
                    curr = curr[val]
            targets.append(curr)
        return tuple(targets)

    def apply(self, func: Callable[[Any], Any]) -> TRoot:
        """Applies a function to all selected targets simultaneously.

        Args:
            func (Callable[[Any], Any]): The transformation function to apply to each target.

        Returns:
            TRoot: A new instance of the root tree with all targets updated.
        """
        if not self._sub_paths:
            return self._tree
            
        return eqx.tree_at(self._get_targets_from, self._tree, replace_fn=func)

    def set(self, value: Any) -> TRoot:
        """Sets all selected targets to a specific value.

        Args:
            value (Any): The new value to assign to all targets.

        Returns:
            TRoot: A new instance of the root tree with the updated values.
        """
        if not self._sub_paths:
            return self._tree

        # eqx.tree_at requires the replacements tuple to match the length of the targets tuple
        replacements = tuple(value for _ in self._sub_paths)
        return eqx.tree_at(self._get_targets_from, self._tree, replace=replacements)

    def get(self) -> list[Any]:
        """Extracts all focused values.

        Returns:
            list[Any]: A list containing the values of all currently focused targets.
        """
        return list(self._get_targets_from(self._tree))
    
    def filter(self, predicate: Callable[[Any], bool]) -> "Traversal[TRoot]":
        """Filters the currently focused targets, keeping only those that match the condition.

        Args:
            predicate (Callable[[Any], bool]): A function that returns True to keep a target, 
                or False to drop it.

        Returns:
            Traversal[TRoot]: A new Traversal focused only on the targets that passed the filter.
        """
        from refrax.lens import Lens
        
        filtered_paths = []
        for path in self._sub_paths:
            target_val = Lens(self._tree, self._base_path + path).get()
            if predicate(target_val):
                filtered_paths.append(path)
                
        return Traversal(self._tree, self._base_path, filtered_paths)