from __future__ import annotations
from typing import Dict, Tuple, Any, Union


class Scope(dict):
    """Hierarchical data storage with index and tag-based access patterns"""
    
    def __init__(self):
        super().__init__()
        self.aliases: Dict[str, Tuple[int, ...]] = {}
    
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
    
    def __getitem__(self, key: Union[str, Tuple]) -> Any:
        """Support both string paths and tuple keys"""
        if isinstance(key, str):
            resolved_key = self._resolve_path(key)
            return super().__getitem__(resolved_key)
        else:
            return super().__getitem__(key)
    
    def _resolve_path(self, path: str) -> Tuple[Union[int, str], ...]:
        """Convert string path to tuple key, handling tags and indices"""
        parts = path.split('.')
        
        if parts[0].isdigit():
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
                
        else:  # handle the alias case
            if parts[0] in self.aliases:
                if len(parts) == 2:
                    # <alias>.<variable>
                    tag_path = self.aliases[parts[0]]
                    resolved_parts = list(tag_path)
                    resolved_parts.append(parts[1])
                elif len(parts) == 3:
                    # <alias>.<variable>.<index/member>
                    tag_path = self.aliases[parts[0]]
                    resolved_parts = list(tag_path)
                    resolved_parts.append(parts[1])
                    if parts[2].isdigit():
                        resolved_parts.append(int(parts[2]))
                    else:
                        resolved_parts.append(parts[2])
                else:
                    raise KeyError(f"Invalid alias path structure: {path}")
            else:
                raise KeyError(f"Unknown tag: {parts[0]}")
        
        return tuple(resolved_parts)


class Ctx:
    """Context proxy that knows its position in the Scope"""
    
    def __init__(self, scope: Scope, index_path: Tuple[int, ...]):
        self.scope = scope
        self.index_path = index_path
    
    def __setitem__(self, key: str, value: Any):
        """Store data in scope at this context's path"""
        full_path = self.index_path + (key,)
        self.scope[full_path] = value
    
    def __getitem__(self, key: str) -> Any:
        """Retrieve data from scope at this context's path"""
        full_path = self.index_path + (key,)
        return self.scope[full_path]
    
    def child(self, index: int, tag: str = None) -> 'Ctx':
        """Create child context with extended index path"""
        child_path = self.index_path + (index,)
        return self.scope.ctx(*child_path, tag=tag)