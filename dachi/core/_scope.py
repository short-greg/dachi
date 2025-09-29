from __future__ import annotations
from typing import Dict, Tuple, Any, Union


class Scope:
    """Hierarchical data storage with index and tag-based access patterns"""
    
    def __init__(self, parent: 'Scope' = None, name: str = None):
        self.parent = parent
        self.name = name
        self.children: Dict[str, 'Scope'] = {}
        self.full_path: Dict[Tuple, Any] = {}  # Indexed storage
        self.aliases: Dict[str, Tuple[int, ...]] = {}
        self.fields: Dict[str, Any] = {}  # Unindexed storage
    
    def ctx(self, *index_path: int) -> 'Ctx':
        """Get or create context at index path, optionally register tag alias"""
        # if tag:
        #     # Remove old tag if it exists
        #     old_tag = None
        #     for existing_tag, existing_path in self.aliases.items():
        #         if existing_path == index_path:
        #             old_tag = existing_tag
        #             break
        #     if old_tag:
        #         del self.aliases[old_tag]
        #     # Register new tag
        #     self.aliases[tag] = index_path
        
        return Ctx(self, index_path)
    
    def child(self, name: str) -> 'Scope':
        """Get or create named child scope"""
        if name not in self.children:
            self.children[name] = Scope(parent=self, name=name)
        return self.children[name]
    
    def base_scope(self) -> 'Scope':
        """Navigate to root scope"""
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope
    
    def change_scope(self, step: str) -> 'Scope':
        """Change scope by one step: '.', '..', or child name"""
        if step == '.' or step == '':
            return self
        elif step == '..':
            if self.parent is None:
                raise ValueError("Cannot go up - no parent scope")
            return self.parent
        else:
            # Child scope name
            return self.child(step)
    
    def bind(self, ctx: Ctx, bindings: Dict[str, str]):
        """Bind variables in the context to the scope

        Args:
            ctx (Ctx): The context to bind from
            bindings (Dict[str, str]): The bindings to apply
            inherit (bool, optional): Whether to inherit from the parent context. Defaults to True.
        """
        return BoundScope(self, ctx, bindings)

    def __getitem__(self, key: Union[str, Tuple]) -> Any:
        scope, key, index = self._resolve_var(key)
        if key[-1] not in scope.fields:
            raise KeyError(f"Unknown tag: {key[-1]}")
        if index is not None:
            return scope.full_path[key][index]
        return scope.full_path[key]
        # """Support both indexed and unindexed data access"""
        # if isinstance(key, tuple):
        #     # Handle indexed access - key is tuple like (0, "goal") or ('./', 0, "goal")
        #     # scope, key = self._resolve_scope(key)
            
        #     return scope.full_path[key]
        # elif isinstance(key, str):
        #     # Handle string path resolution
        #     scope, key = self._resolve_scope(key)
        #     key = '.'.join(key)
        #     resolved_key = scope._resolve_path(key)
        #     return scope.full_path[resolved_key]
        # else:
        #     raise ValueError(f"Invalid key format: {key}")
    
    def __setitem__(self, key: Union[str, Tuple], value: Any):
        """Support both indexed and unindexed data assignment"""
        scope, key, index = self._resolve_var(key)
        if index is not None:
            scope.full_path[key][index] = value
            return value
        scope.full_path[key] = value
        scope.fields[key[-1]] = value
        scope.aliases[key[-1]] = key
        return value
        # if isinstance(key, tuple):
        #     # Handle indexed assignment - key is tuple like (0, "goal") or ('./', 0, "goal")
        #     scope, key = self._resolve_scope(key)
        #     scope.full_path[key] = value
        #     # Also store in unindexed if it's a simple field reference
        #     if len(key) >= 2 and isinstance(key[-1], str):
        #         scope.fields[key[-1]] = value
        #         print(key[-1])
        #         scope.aliases[key[-1]] = key
        # elif isinstance(key, str):
        #     # Handle string path resolution
        #     scope, key = self._resolve_scope(key)
        #     key = '.'.join(key)
        #     resolved_key = scope._resolve_path(key)
        #     scope.full_path[resolved_key] = value
        #     # Also store in unindexed if it's a simple field reference
        #     if len(resolved_key) >= 2 and isinstance(resolved_key[-1], str):
        #         scope.fields[resolved_key[-1]] = value
        #         scope.aliases[resolved_key[-1]] = resolved_key
        # else:
        #     raise ValueError(f"Invalid key format: {key}")
    
    def set(self, path: Union[Tuple, str], field: str, value: Any, index: Union[str, int] = None):
        """Set value at both indexed and unindexed locations"""
        if isinstance(path, str):
            # Use same pattern as __setitem__
            if path.endswith('/'):
                key = f"{path}{field}"
            else:
                key = f"{path}.{field}"
            if index is not None:
                key = f"{key}.{index}"
        else: # isinstance(path, tuple):
            key = path + (field,)
            if index is not None:
                key = key + (index,)
        
        self[key] = value
        return value
        
        #     scope, key_parts = self._resolve_scope(key)
        #     key = '.'.join(key_parts)
        #     resolved_key = scope._resolve_path(key)
        # else:
        #     # Tuple path

        #     resolved_key = path + (field,)
        #     scope, resolved_key = self._resolve_scope(resolved_key)
        
        # # Store indexed version
        # if index is not None:
        #     # Handle sub-field setting
        #     if resolved_key not in scope.full_path:
        #         raise KeyError(f"Cannot set sub-field on non-existent field: {resolved_key}")
        #     scope._set_subfield(scope.full_path[resolved_key], index, value)
        # else:
        #     scope.full_path[resolved_key] = value
        #     # Store unindexed version (always set the full value for unindexed)
        #     scope.fields[field] = value
        #     scope.aliases[field] = resolved_key
        # return value
    
    def path(self, path: Union[Tuple, str], field: str, index: Union[str, int] = None) -> Any:
        """Get value from indexed location with a full path
        
        Args:
            path (Union[Tuple, str]): The path or the tuple representing the path
            field (str): The field name to access
        
        """
        if isinstance(path, str):
            # Use same pattern as __setitem__
            if path.endswith('/'):
                key = f"{path}{field}"
            else:
                key = f"{path}.{field}"
            if index is not None:
                key = f"{key}.{index}"
        else: # isinstance(path, tuple):
            key = path + (field,)
            if index is not None:
                key = key + (index,)
        return self[key]
    
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
    
    def _resolve_var(self, key: Tuple | str) -> Tuple['Scope', Tuple, Union[str, int, None]]:
        """Resolve scope from lexical scoping, return (target_scope, remaining_key)"""
        if isinstance(key, str) and '/' in key:
            # Explicit path navigation
            scope, key_parts = self._resolve_scope(key)
            print(scope, key_parts)
            print(id(scope))
            resolved_key = self._resolve_path('.'.join(key_parts))
            return scope, resolved_key, None
        elif isinstance(key, str):
            key_split = key.split('.')
            if len(key_split) == 1:
                alias = key_split[0]
                index = None
            elif len(key_split) == 2:
                alias, index = key_split
            else:
                # Too many parts - this is invalid for lexical scoping
                raise KeyError(f"Invalid alias pattern: {key}")
            # resolve alias scope will propagate upward to find the correct scope
            # will raise an error if not available
            # Resolve alias using lexical scoping (check parent scopes)
            scope, key_parts = self._resolve_alias_scope(alias)
            return scope, key_parts, index
        elif len(key) >= 0 and isinstance(key[0], str):
            # Tuple with explicit scope navigation
            scope, key_parts = self._resolve_scope(key)
            return scope, key_parts, None
        # else key is a tuple and current scope
        return self, key, None
    
    def _resolve_scope(self, key: Tuple | str) -> Tuple['Scope', Tuple]:
        """Resolve scope from explicit path, return (target_scope, remaining_key)"""
        
        scope = self
        if isinstance(key, str):
            is_base = key[0] == '/'
            if is_base:
                key = key[1:]
            
            scope_part = key.split('/')
            field_part = scope_part[-1].split('.')
            scope_part = scope_part[:-1]
        elif '/' in key[0]:
            is_base = key[0][0] == '/'
            if is_base:
                key = (key[0][1:], *key[1:])
            
            scope_part = key[0].split('/')
            field_part = key[1:]
        else:
            is_base = False
            scope_part = []
            field_part = key

        if is_base:
            scope = self.base_scope()

        for part in scope_part:
            try:
                print('Changing: ', part)
                scope = scope.change_scope(part)
            except KeyError:
                raise KeyError(f"Unknown scope part: {part}")
        return scope, tuple(field_part)


            # # Handle string paths for explicit navigation only
            # parts = key.split('/')
            # if len(parts) > 1:
            #     # Has explicit scope path like "/target/0.field" or "../0.field"
            #     if parts[0] == '':
            #         # Absolute path: starts with /
            #         scope = self.base_scope()
            #         scope_parts = [p for p in parts[1:-1] if p]  # Remove empty first and last field part
            #     else:
            #         # Relative path: starts with .. or scope name
            #         scope = self
            #         scope_parts = parts[:-1]  # All except last field part
                
            #     # Navigate through scope parts
            #     for step in scope_parts:
            #         scope = scope.change_scope(step)
                
            #     # Parse field part
            #     field_part = parts[-1]
            #     field_key = tuple(field_part.split('.')) if field_part else ()
            #    return scope, field_key
            # else:
            #     # No explicit scope path - this should use lexical scoping (handled elsewhere)
            #     return self, tuple(key.split('.'))
        
        # # Handle tuple input for explicit scope navigation
        # if not key or not isinstance(key[0], str) or '/' not in key[0]:
        #     return self, key
        
        # # Tuple scope references like ('../', 0, 'field')
        # scope_ref = key[0]
        # remaining_key = key[1:]
        
        # if scope_ref == './':
        #     return self, remaining_key
        # elif scope_ref.startswith('../'):
        #     levels = scope_ref.count('../')
        #     scope = self
        #     for _ in range(levels):
        #         scope = scope.change_scope('..')
        #     return scope, remaining_key
        # elif scope_ref.startswith('/'):
        #     scope = self.base_scope()
        #     if len(scope_ref) > 1:  # More than just '/'
        #         path_parts = [p for p in scope_ref[1:].split('/') if p]
        #         for part in path_parts:
        #             scope = scope.change_scope(part)
        #     return scope, remaining_key
        # else:
        #     raise ValueError(f"Invalid scope reference: {scope_ref}")
    
    def _resolve_path(self, path: str) -> Tuple[Union[int, str], ...]:
        """Convert string path to tuple key, handling tags and indices"""

        # path = path.split('/')
        # is_path = len(path) > 1
        
        parts = path.split('.')

        resolved_parts = []
        # Add leading digits
        for i, val in enumerate(parts):
            if val.isdigit():
                resolved_parts.append(int(val))
            else:
                break
        if i == len(parts) - 1:  # then last item
            # Just a variable name
            resolved_parts.append(parts[i])
            tag = parts[i]
            # if tag not in self.aliases:
            #     raise KeyError(f"Unknown tag: {tag}")
        elif i == len(parts) - 2:  # then second to last item
            tag = parts[i]
            index = parts[i + 1]
            if index.isdigit():
                index = int(index)

            # if tag not in self.aliases:
            #     raise KeyError(f"Unknown tag: {tag}")
            resolved_parts.extend((tag, index))
        else:
            raise KeyError(f"Invalid path structure: {path}")

        # elif i == len(parts) - 2:  # then second to last item  
        #     # Tag followed by member name
        #     tag = parts[i]
        #     if tag in self.aliases:
        #         tag_path = self.aliases[tag]
        #         # Only extend with the part not already covered
        #         uncovered_path = tag_path[len(resolved_parts):]
        #         resolved_parts.extend(uncovered_path)
        #     else:
        #         raise KeyError(f"Unknown tag: {tag}")
        #     # Add final member
        #     final_part = parts[i + 1]
        #     if final_part.isdigit():
        #         resolved_parts.append(int(final_part))
        #     else:
        #         resolved_parts.append(final_part)
        # else:
        #     raise KeyError(f"Invalid path structure: {path}")
        
        # if len(path) > 1:
        #     scope = '/'.join(path[:-1])
        #     scope += '/'
        #     return (scope, *resolved_parts)

        return tuple(resolved_parts)
    
    def _resolve_alias(self, alias: str) -> Tuple[Tuple[int, ...], 'Scope']:
        """Resolve alias using lexical scoping - search up parent chain"""
        
        parts = alias.split('.')
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
                raise KeyError(f"Invalid alias path structure: {alias}")
        else:
            raise KeyError(f"Unknown tag: {parts[0]}")
        return tuple(resolved_parts)

    
    def _resolve_alias_scope(self, alias: str) -> Tuple['Scope', Tuple]:
        """Resolve alias using lexical scoping - search up parent chain"""
        scope = self
        while scope is not None:
            if alias in scope.aliases:
                return scope, scope.aliases[alias]
            scope = scope.parent
        raise KeyError(f"Alias '{alias}' not found in scope chain")
    
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
            return False
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
    
    def child(self, index: int) -> 'Ctx':
        """Create child context with extended index path"""
        child_path = self.index_path + (index,)
        return self.scope.ctx(*child_path)
    
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

    def ctx(self, *index_path: int) -> 'Ctx':
        """Get or create context at index path, optionally register tag alias"""
        # if tag:
        #     # Remove old tag if it exists
        #     old_tag = None
        #     for existing_tag, existing_path in self.aliases.items():
        #         if existing_path == index_path:
        #             old_tag = existing_tag
        #             break
        #     if old_tag:
        #         del self.aliases[old_tag]
        #     # Register new tag
        #     self.aliases[tag] = index_path
        
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

    def child(self, index):
        """Create child context with extended index path"""
        child_path = self.index_path + (index,)
        return self.scope.ctx(*child_path).bind(self.bindings)
    
