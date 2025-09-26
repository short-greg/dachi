from __future__ import annotations
from typing import Dict, Tuple, Any, Union


class Scope:
    """Hierarchical data storage with index and tag-based access patterns"""
    
    def __init__(self):
        self.full_path: Dict[Tuple, Any] = {}  # Indexed storage
        self.aliases: Dict[str, Tuple[int, ...]] = {}
        self.fields: Dict[str, Any] = {}  # Unindexed storage
    
    def ctx(self, *index_path: int, tag: str = None) -> 'Ctx':
        """Get or create context at index path, optionally register tag alias"""
        if tag:
            # Remove old tag if it exists
            old_tag = None
            for existing_tag, existing_path in self.aliases.items():
                if existing_path == index_path:
                    old_tag = existing_tag
                    break
            if old_tag:
                del self.aliases[old_tag]
            # Register new tag
            self.aliases[tag] = index_path
        
        return Ctx(self, index_path)
    
    def bind(self, ctx: Ctx, bindings: Dict[str, str]):
        """Bind variables in the context to the scope

        Args:
            ctx (Ctx): The context to bind from
            bindings (Dict[str, str]): The bindings to apply
            inherit (bool, optional): Whether to inherit from the parent context. Defaults to True.
        """
        return BoundScope(self, ctx, bindings)

    def __getitem__(self, key: Union[str, Tuple]) -> Any:
        """Support both indexed and unindexed data access"""
        if isinstance(key, tuple):
            # Handle indexed access - key is tuple like (0, "goal") or (0, 1, "status")
            return self.full_path[key]
        elif isinstance(key, str):
            # Handle string path resolution
            resolved_key = self._resolve_path(key)
            return self.full_path[resolved_key]
        else:
            raise ValueError(f"Invalid key format: {key}")
    
    def __setitem__(self, key: Union[str, Tuple], value: Any):
        """Support both indexed and unindexed data assignment"""
        if isinstance(key, tuple):
            # Handle indexed assignment - key is tuple like (0, "goal") or (0, 1, "status")
            self.full_path[key] = value
            # Also store in unindexed if it's a simple field reference
            if len(key) >= 2 and isinstance(key[-1], str):
                self.fields[key[-1]] = value
                print(key[-1])
                self.aliases[key[-1]] = key
        elif isinstance(key, str):
            # Handle string path resolution
            resolved_key = self._resolve_path(key)
            self.full_path[resolved_key] = value
            # Also store in unindexed if it's a simple field reference
            if len(resolved_key) >= 2 and isinstance(resolved_key[-1], str):
                self.fields[resolved_key[-1]] = value
                self.aliases[resolved_key[-1]] = resolved_key
        else:
            raise ValueError(f"Invalid key format: {key}")
    
    def set(self, path: Union[Tuple, str], field: str, value: Any, index: Union[str, int] = None):
        """Set value at both indexed and unindexed locations"""
        # Build the full key using existing logic
        if isinstance(path, str):
            # Use existing _resolve_path but extract just the path part
            # full_key = self._resolve_path(f"{path}.{field}")
            path = path.split('.')
        full_key = path + (field,)
        
        # Store indexed version
        if index is not None:
            # Handle sub-field setting
            if full_key not in self.full_path:
                raise KeyError(f"Cannot set sub-field on non-existent field: {full_key}")
            self._set_subfield(self.full_path[full_key], index, value)
        else:
            self.full_path[full_key] = value
            # Store unindexed version (always set the full value for unindexed)
            self.fields[field] = value
            self.aliases[field] = full_key
        return value
    
    def path(self, path: Union[Tuple, str], field: str, index: Union[str, int] = None) -> Any:
        """Get value from indexed location with a full path
        
        Args:
            path (Union[Tuple, str]): The path or the tuple representing the path
            field (str): The field name to access
        
        """
        # Build the full key using existing logic
        if isinstance(path, str):
            path = path.split('.')
        full_key = path + (field,)
        
        value = self.full_path[full_key]
        
        if index is not None:
            return self._get_subfield(value, index)
        return value
    
    def _get_subfield(self, value: Any, index: Union[str, int]) -> Any:
        """Get sub-field from value using index"""
        if isinstance(index, int):
            return value[index]  # For tuples, lists
        elif isinstance(index, str):
            return getattr(value, index)  # For object attributes
        else:
            raise ValueError(f"Invalid index type: {type(index)}")
    
    def _set_subfield(self, value: Any, index: Union[str, int], new_value: Any):
        """Set sub-field in value using index"""
        if isinstance(index, int):
            value[index] = new_value  # For mutable sequences
        elif isinstance(index, str):
            setattr(value, index, new_value)  # For object attributes
        else:
            raise ValueError(f"Invalid index type: {type(index)}")
    
    def _resolve_path(self, path: str) -> Tuple[Union[int, str], ...]:
        """Convert string path to tuple key, handling tags and indices"""
        is_path = path.startswith("/")
        if is_path:
            path = path[1:]  # Remove leading slash for root-relative paths

        parts = path.split('.')

        if is_path:
            resolved_parts = []
            # Add leading digits
            for i, val in enumerate(parts):
                if not val.isdigit():
                    break
                resolved_parts.append(int(val))
            
            if i == len(parts) - 1:  # then last item
                # Just a variable name
                resolved_parts.append(parts[i])
            elif i == len(parts) - 2:  # then second to last item  
                # Tag followed by member name
                tag = parts[i]
                if tag in self.aliases:
                    tag_path = self.aliases[tag]
                    # Only extend with the part not already covered
                    uncovered_path = tag_path[len(resolved_parts):]
                    resolved_parts.extend(uncovered_path)
                else:
                    raise KeyError(f"Unknown tag: {tag}")
                # Add final member
                final_part = parts[i + 1]
                if final_part.isdigit():
                    resolved_parts.append(int(final_part))
                else:
                    resolved_parts.append(final_part)
            else:
                raise KeyError(f"Invalid path structure: {path}")
                
        else:  # it is an alias or direct variable
            if parts[0] in self.aliases:
                if len(parts) == 1:
                    # <alias>.<variable>
                    tag_path = self.aliases[parts[0]]
                    resolved_parts = list(tag_path)
                elif len(parts) == 2:
                    # <alias>.<variable>.<index/member>
                    tag_path = self.aliases[parts[0]]
                    resolved_parts = list(tag_path)
                    if parts[1].isdigit():
                        resolved_parts.append(int(parts[1]))
                    else:
                        resolved_parts.append(parts[1])
                else:
                    raise KeyError(f"Invalid alias path structure: {path}")
            else:
                raise KeyError(f"Unknown tag: {parts[0]}")
        
        return tuple(resolved_parts)
    
# $.x
# BT(
#  root=SomeAction()
# )
# @x
# bindings={"x": "@0.goal"} # this will bind to the context incoming at the root
# bindings={"x": "goal"}
# bindings={"x"}: "/goal"
# bindings={"x": "/goal"} # will use the path resolution to find the goal at the root level
# bindings={"x": "/0.goal"}

class BoundScope(Scope):
    """A scope that has bindings applied"""
    
    def __init__(self, base_scope: Scope, base_ctx: Ctx, bindings: Dict[str, str]):
        super().__init__()
        self.base_scope = base_scope
        self.base_ctx = base_ctx
        self.bindings = bindings
    
    def __getitem__(self, key: Union[str, Tuple]) -> Any:
        """Retrieve data with bindings applied"""
        if isinstance(key, str) and key.startswith("@"):
            key = key[1:]
            if key in self.bindings:
                bound_key = self.bindings[key]
                return self.base_ctx[bound_key]
            else:
                raise KeyError(f"Binding for {key} not found")

        return self.base_scope[key]
    
    def __setitem__(self, key: Union[str, Tuple], value: Any):
        """Store data with bindings applied
        
        
        """
        if isinstance(key, str) and key.startswith("@"):
            key = key[1:]
            if key in self.bindings:
                bound_key = self.bindings[key]
                self.base_ctx[bound_key] = value
                return value
            else:
                raise KeyError(f"Binding for {key} not found")
        self.base_scope[key] = value
        return value

    def __contains__(self, key) :
        """Check if key exists with bindings applied"""
        if isinstance(key, str) and key.startswith("@"):
            key = key[1:]
            if key in self.bindings:
                bound_key = self.bindings[key]
                return bound_key in self.base_ctx
            else:
                raise KeyError(f"Binding for {key} not found")
        return key in self.base_scope

    def path(self, index_path: Tuple[int, ...], field: str, index: Union[str, int] = None) -> Any:
        """Get value from indexed location at this context's path"""
        return self.bound_scope.path(index_path, field, index)
    
    def set(self, index_path: Tuple[int, ...], field: str, value: Any, index: Union[str, int] = None):
        """Set value at both indexed and unindexed locations"""
        return self.bound_scope.set(index_path, field, value, index)


class Ctx:
    """Context proxy that knows its position in the Scope"""
    
    def __init__(self, scope: Scope, index_path: Tuple[int, ...]):
        self.scope = scope
        self.index_path = index_path
    
    def __setitem__(self, key: str, value: Any):
        """Store data at both indexed and unindexed locations"""
        if key.startswith('/'):
            self.scope[key] = value
            return value
        key = key.split('.')
        if len(key) == 1:
            key = key[0]
            index = None
        else:
            key, index = key[0], key[1]
        
        self.scope.set(self.index_path, key, value, index)
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Retrieve data from scope at this context's index path"""
        if key.startswith('/'):
            return self.scope[key]
        return self.scope[key]
    
    def path(self, field: str, index: Union[str, int] = None) -> Any:
        """Get value from indexed location at this context's path"""
        return self.scope.path(self.index_path, field, index)
    
    def child(self, index: int, tag: str = None) -> 'Ctx':
        """Create child context with extended index path"""
        child_path = self.index_path + (index,)
        return self.scope.ctx(*child_path, tag=tag)
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update scope with data at this context's path (dict-like behavior)"""
        for key, value in data.items():
            self[key] = value
    
    def keys(self):
        """Return keys from scope.fields for dict-like behavior"""
        return self.scope.fields.keys()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in scope.fields for dict-like behavior"""
        return key in self.scope.fields

    def bind(self, bindings: Dict[str, str]) -> BoundCtx:
        """Bind variables in the context to the scope

        Args:
            ctx (Ctx): The context to bind from
            bindings (Dict[str, str]): The bindings to apply
            inherit (bool, optional): Whether to inherit from the parent context. Defaults to True.
        """
        return BoundCtx(self, bindings)

    def ctx(self, *index_path: int, tag: str = None) -> 'Ctx':
        """Get or create context at index path, optionally register tag alias"""
        if tag:
            # Remove old tag if it exists
            old_tag = None
            for existing_tag, existing_path in self.aliases.items():
                if existing_path == index_path:
                    old_tag = existing_tag
                    break
            if old_tag:
                del self.aliases[old_tag]
            # Register new tag
            self.aliases[tag] = index_path
        
        return Ctx(self, index_path)


# bound_ctx.path("")

class BoundCtx(Ctx):
    """A context that has bindings applied"""
    
    def __init__(self, base_ctx: Ctx, bindings: Dict[str, str]):
        super().__init__(base_ctx.scope, base_ctx.index_path)
        self.bindings = bindings
        self.base_ctx = base_ctx
    
    def __getitem__(self, key: str) -> Any:
        """Retrieve data with bindings applied"""
        if key in self.bindings:
            bound_key = self.bindings[key]
            return self.base_ctx[bound_key]
        return self.base_ctx[key]
    
    def path(self, field: str, index: Union[str, int] = None) -> Any:
        """Get the value specified by the path. Since the 
        
        Args:
            field (str): The field name to access
            index (Union[str, int], optional): The index or attribute to access within the field. Defaults to None.
        
        """
        if field in self.bindings:
            bound_field = self.bindings[field]
            return self.base_ctx.path(bound_field, index)
        return self.base_ctx.path(field, index)
    
    def __setitem__(self, key: str, value: Any):
        """Store data with bindings applied"""
        if key in self.bindings:
            bound_key = self.bindings[key]
            self.base_ctx[bound_key] = value
        else:
            self.base_ctx[key] = value

    def __contains__(self, key):
        """Check if key exists with bindings applied"""
        if key in self.bindings:
            bound_key = self.bindings[key]
            return bound_key in self.base_ctx
        return key in self.base_ctx
    
    def keys(self):
        """Return keys with bindings applied"""
        base_keys = set(self.base_ctx.keys())
        bound_keys = set(self.bindings.keys())
        return base_keys.union(bound_keys)
    
    def update(self, data):
        """Update scope with data with bindings applied"""
        for key, value in data.items():
            if key in self.bindings:
                bound_key = self.bindings[key]
                self.base_ctx[bound_key] = value
            else:
                self.base_ctx[key] = value

    def child(self, index, tag = None):
        """Create child context with extended index path"""
        child_path = self.index_path + (index,)
        return self.scope.ctx(*child_path, tag=tag).bind(self.bindings)
    
