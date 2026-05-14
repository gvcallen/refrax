from typing import Generic, Any, Callable, cast

import equinox as eqx
import jax.tree_util as jtu

from refrax.custom_types import TRoot, PathStep, PathOp
from refrax.utils import parse_string_path, translate_jax_path


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
    
    def get(self) -> list[Any]:
        """Extracts all focused values.

        Returns:
            list[Any]: A list containing the values of all currently focused targets.
        """
        return list(self._get_targets_from(self._tree))    

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
    
    def path(self, target_path: str | tuple) -> "Traversal[TRoot]":
        """
        Broadens the traversal by appending a string path or JAX KeyPath to every currently focused target.

        Args:
            target_path (str | tuple): The path string (e.g., '.res.R') or native 
                JAX KeyPath tuple.

        Returns:
            Traversal[TRoot]: A new Traversal focused one level deeper across all branches.
        """
        if isinstance(target_path, str):
            parsed_steps = parse_string_path(target_path)
        elif isinstance(target_path, tuple):
            parsed_steps = translate_jax_path(target_path)
        else:
            raise TypeError(f"Expected string or JAX path tuple, got {type(target_path).__name__}")
        
        # Append the parsed steps to all diverging paths
        new_sub_paths = [path + parsed_steps for path in self._sub_paths]
        return Traversal(self._tree, self._base_path, new_sub_paths)
    
    def select(self, *paths: str | tuple) -> "Traversal[TRoot]":
        parsed_additions = []
        for p in paths:
            if isinstance(p, str):
                parsed_additions.append(parse_string_path(p))
            elif isinstance(p, tuple):
                parsed_additions.append(translate_jax_path(p))
            else:
                raise TypeError("Expected string or JAX path tuple")
                
        new_sub_paths = []
        for existing_path in self._sub_paths:
            for addition in parsed_additions:
                new_sub_paths.append(existing_path + addition)
                
        return Traversal(self._tree, self._base_path, new_sub_paths)    
    
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

    def exclude(self, predicate: Callable[[Any], bool]) -> "Traversal[TRoot]":
        """Filters the currently focused targets, dropping those that match the condition.
        
        This is the logical inverse of `.filter()`.

        Args:
            predicate (Callable[[Any], bool]): A function that returns True to drop a target.

        Returns:
            Traversal[TRoot]: A new Traversal with the matching targets removed.
        """
        return self.filter(lambda x: not predicate(x))
    
    def prune(self, *paths: str | tuple) -> "Traversal[TRoot]":
        """
        Filters out targets from the Traversal that intersect with the given paths.
        
        Crucially, this prunes a target if it is exactly the excluded path, inside 
        the excluded path, OR a parent of the excluded path. (Mutating a parent 
        implicitly mutates its children, so parents of excluded paths must be pruned).

        Args:
            *paths (str | tuple): String paths or JAX KeyPath tuples to protect from mutation.

        Returns:
            Traversal[TRoot]: A new Traversal with the intersecting paths removed.
        """
        # Parse the user's paths into refrax PathSteps
        parsed_excludes = []
        for p in paths:
            if isinstance(p, str):
                parsed_excludes.append(parse_string_path(p))
            elif isinstance(p, tuple):
                parsed_excludes.append(translate_jax_path(p))
            else:
                raise TypeError(f"Expected string or JAX path tuple, got {type(p).__name__}")
        
        kept_sub_paths = []
        for path in self._sub_paths:
            # Reconstruct the absolute path from the root to the target
            full_target_path = self._base_path + path
            
            should_prune = False
            for ex_path in parsed_excludes:
                # 1. Is the target a PARENT of the excluded path? (or an exact match)
                if len(full_target_path) <= len(ex_path) and ex_path[:len(full_target_path)] == full_target_path:
                    should_prune = True
                    break
                
                # 2. Is the target a CHILD of the excluded path?
                if len(full_target_path) > len(ex_path) and full_target_path[:len(ex_path)] == ex_path:
                    should_prune = True
                    break
                    
            if not should_prune:
                kept_sub_paths.append(path)
                
        return Traversal(self._tree, self._base_path, kept_sub_paths)
    
    def leaves(self, is_leaf: Callable[[Any], bool] | None = None) -> "Traversal[TRoot]":
        targets = self.get()
        new_sub_paths = []
        
        for path, target_val in zip(self._sub_paths, targets):
            leaves_with_paths = jtu.tree_leaves_with_path(target_val, is_leaf=is_leaf)
            for jax_path, _val in leaves_with_paths:
                relative_path = translate_jax_path(jax_path)
                new_sub_paths.append(path + relative_path)
                
        return Traversal(self._tree, self._base_path, new_sub_paths)
    
    def where(self, predicate: Callable[[Any], bool]) -> "Traversal[TRoot]":
        targets = self.get()
        new_sub_paths = []
        
        for path, target_val in zip(self._sub_paths, targets):
            if isinstance(target_val, dict):
                for k, val in target_val.items():
                    if predicate(val):
                        new_sub_paths.append(path + [("item", k)])
            elif isinstance(target_val, (list, tuple)):
                for i, val in enumerate(target_val):
                    if predicate(val):
                        new_sub_paths.append(path + [("item", i)])
            elif hasattr(target_val, '__dict__'):
                for attr_name, val in vars(target_val).items():
                    if not attr_name.startswith('_') and predicate(val):
                        new_sub_paths.append(path + [("attr", attr_name)])
                        
        return Traversal(self._tree, self._base_path, new_sub_paths)
    
    def each(self) -> "Traversal[TRoot]":
        targets = self.get()
        new_sub_paths = []
        
        for path, target_val in zip(self._sub_paths, targets):
            if isinstance(target_val, dict):
                for k in target_val.keys():
                    new_sub_paths.append(path + [("item", k)])
            elif isinstance(target_val, (list, tuple)):
                for i in range(len(target_val)):
                    new_sub_paths.append(path + [("item", i)])
            else:
                raise TypeError(f"Cannot iterate over {type(target_val).__name__} with .each()")
                
        return Traversal(self._tree, self._base_path, new_sub_paths)