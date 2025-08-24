from __future__ import annotations

# 1st party
import uuid
import time
import threading
from typing import TypeVar, Generic, Callable, Dict, List, Optional, TypedDict, Any, Union, ClassVar
from collections import defaultdict

# 3rd party
from pydantic import BaseModel, Field, PrivateAttr

# local
from ._utils import singleton

T = TypeVar("T", bound=BaseModel)


class Post(TypedDict):
    id: str
    item: Any
    locked: bool
    timestamp: float
    expires_at: Optional[float]


class Bulletin(BaseModel, Generic[T]):
    """
    A bulletin board for posting and retrieving items between tasks or agents.
    
    Supports publishing items with optional locking, retrieving items with
    filtering capabilities, releasing locked items, expiration, and callbacks.
    """
    
    # Event types
    ON_PUBLISH: ClassVar[str] = "publish"
    ON_RELEASE: ClassVar[str] = "release"
    ON_REMOVE: ClassVar[str] = "remove"
    ON_EXPIRE: ClassVar[str] = "expire"
    
    _posts: Dict[str, Post] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _type_registry: Dict[type, List[str]] = PrivateAttr(default_factory=lambda: defaultdict(list))
    _callbacks: List[Callable[[Post, str], None]] = PrivateAttr(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        
    def register_callback(self, callback: Callable[[Post, str], None]) -> bool:
        """
        Register a callback for all bulletin events.
        
        Args:
            callback: Function to call with (Post, event_type) when any event occurs
            
        Returns:
            bool: True if callback was registered, False if already exists
        """
        if callback in self._callbacks:
            return False
        self._callbacks.append(callback)
        return True
    
    def unregister_callback(self, callback: Callable[[Post, str], None]) -> bool:
        """
        Unregister a callback for bulletin events.
        
        Args:
            callback: Function to remove
            
        Returns:
            bool: True if callback was removed, False if not found
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def _trigger_callbacks(self, event: str, post: Post) -> None:
        """Trigger all callbacks for an event."""
        for callback in self._callbacks:
            try:
                callback(post, event)
            except Exception:
                pass  # Don't let callback errors break the bulletin
    
    def _cleanup_expired_posts(self) -> List[str]:
        """Remove expired posts and return their IDs."""
        now = time.time()
        expired_ids = []
        
        for post_id, post in list(self._posts.items()):
            if post["expires_at"] is not None and now >= post["expires_at"]:
                expired_ids.append(post_id)
                # Trigger expire callback before removing
                self._trigger_callbacks(self.ON_EXPIRE, post)
                
                # Remove from posts and type registry
                del self._posts[post_id]
                item_type = type(post["item"])
                if item_type in self._type_registry:
                    try:
                        self._type_registry[item_type].remove(post_id)
                        if not self._type_registry[item_type]:
                            del self._type_registry[item_type]
                    except ValueError:
                        pass
                        
        return expired_ids
    
    def publish(self, item: T, lock: bool = True, ttl: Optional[float] = None, scope: Optional[str] = None) -> str:
        """
        Publish an item to the bulletin board.
        
        Args:
            item: The item to publish (must be a BaseModel instance)
            lock: Whether to lock the item (default: True)
            ttl: Time to live in seconds (None for no expiration)
            scope: Optional scope to namespace the post
            
        Returns:
            str: The unique ID of the published post (local ID if scoped)
        """
        if not isinstance(item, BaseModel):
            raise TypeError("Published items must be BaseModel instances")
            
        local_id = str(uuid.uuid4())
        post_id = f"{scope}.{local_id}" if scope else local_id
        now = time.time()
        expires_at = now + ttl if ttl is not None else None
        
        with self._lock:
            post: Post = {
                "id": post_id,
                "item": item,
                "locked": lock,
                "timestamp": now,
                "expires_at": expires_at
            }
            self._posts[post_id] = post
            self._type_registry[type(item)].append(post_id)
            
            # Trigger publish callbacks
            self._trigger_callbacks(self.ON_PUBLISH, post)
            
        return local_id if scope else post_id
    
    def retrieve_first(
        self, 
        id: Optional[str] = None,
        filter_func: Optional[Callable[[T], bool]] = None,
        order_func: Optional[Callable[[T], Any]] = None,
        item_type: Optional[type] = None
    ) -> Optional[Post]:
        """
        Retrieve the first matching post.
        
        Args:
            id: Specific post ID to retrieve
            filter_func: Function to filter items 
            order_func: Function to extract sort key from items (like sorted() key param)
            item_type: Type of item to retrieve
            
        Returns:
            Post or None if no matching post found
        """
        with self._lock:
            self._cleanup_expired_posts()
            
            if id is not None:
                return self._posts.get(id)
            
            posts_to_check = []
            if item_type is not None:
                post_ids = self._type_registry.get(item_type, [])
                posts_to_check = [self._posts[pid] for pid in post_ids if pid in self._posts]
            else:
                posts_to_check = list(self._posts.values())
            
            # Filter posts
            if filter_func is not None:
                posts_to_check = [p for p in posts_to_check if filter_func(p["item"])]
            
            if not posts_to_check:
                return None
                
            # Sort posts if order function provided
            if order_func is not None:
                posts_to_check.sort(key=lambda p: order_func(p["item"]))
            else:
                # Default: sort by timestamp (FIFO)
                posts_to_check.sort(key=lambda p: p["timestamp"])
                
            return posts_to_check[0]
    
    def retrieve_all(
        self,
        filter_func: Optional[Callable[[T], bool]] = None,
        order_func: Optional[Callable[[T], Any]] = None,
        item_type: Optional[type] = None,
        include_locked: bool = True
    ) -> List[Post]:
        """
        Retrieve all matching posts.
        
        Args:
            filter_func: Function to filter items
            order_func: Function to extract sort key from items (like sorted() key param)
            item_type: Type of item to retrieve
            include_locked: Whether to include locked items
            
        Returns:
            List of matching posts sorted by order_func (or timestamp if none provided)
        """
        with self._lock:
            self._cleanup_expired_posts()
            
            posts_to_check = []
            if item_type is not None:
                post_ids = self._type_registry.get(item_type, [])
                posts_to_check = [self._posts[pid] for pid in post_ids if pid in self._posts]
            else:
                posts_to_check = list(self._posts.values())
            
            results = []
            for post in posts_to_check:
                if not include_locked and post["locked"]:
                    continue
                if filter_func is None or filter_func(post["item"]):
                    results.append(post)
            
            # Sort results
            if order_func is not None:
                results.sort(key=lambda p: order_func(p["item"]))
            else:
                # Default: sort by timestamp (FIFO)
                results.sort(key=lambda p: p["timestamp"])
                    
        return results
    
    def release(self, id: str) -> bool:
        """
        Release a locked post, making it available for retrieval again.
        
        Args:
            id: Post ID string
            
        Returns:
            bool: True if the post was successfully released, False otherwise
        """
        with self._lock:
            if id in self._posts:
                self._posts[id]["locked"] = False
                self._trigger_callbacks(self.ON_RELEASE, self._posts[id])
                return True
        return False
    
    def remove(self, id: str) -> bool:
        """
        Remove a post from the bulletin board.
        
        Args:
            id: Post ID string
            
        Returns:
            bool: True if the post was successfully removed, False otherwise
        """
        with self._lock:
            if id in self._posts:
                post = self._posts[id]
                item_type = type(post["item"])
                
                # Trigger remove callback before removing
                self._trigger_callbacks(self.ON_REMOVE, post)
                
                # Remove from posts
                del self._posts[id]
                
                # Remove from type registry
                if item_type in self._type_registry:
                    try:
                        self._type_registry[item_type].remove(id)
                        if not self._type_registry[item_type]:
                            del self._type_registry[item_type]
                    except ValueError:
                        pass  # Already removed
                        
                return True
        return False
    
    def clear(self) -> int:
        """
        Clear all posts from the bulletin board.
        
        Returns:
            int: Number of posts that were cleared
        """
        with self._lock:
            count = len(self._posts)
            self._posts.clear()
            self._type_registry.clear()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the bulletin board.
        
        Returns:
            Dict with stats including total posts, locked posts, and type breakdown
        """
        with self._lock:
            total_posts = len(self._posts)
            locked_posts = sum(1 for post in self._posts.values() if post["locked"])
            type_breakdown = {
                type_name: len(post_ids) 
                for type_name, post_ids in self._type_registry.items()
            }
            
            return {
                "total_posts": total_posts,
                "locked_posts": locked_posts,
                "available_posts": total_posts - locked_posts,
                "type_breakdown": type_breakdown
            }
    


class Blackboard(BaseModel):
    """
    Shared state storage with reactive callbacks for behavior trees and agents.
    
    Supports attribute-style access with automatic callback triggering on changes.
    Provides scoping for isolated namespaces and TTL for automatic expiration.
    
    WARNING: Direct mutations of mutable objects will not trigger callbacks:
        blackboard.items = [1, 2, 3]        # ✓ Triggers callbacks
        blackboard.items.append(4)          # ✗ Silent mutation, no callback
        
    Solution: Reassign the entire object:
        items = blackboard.items.copy()
        items.append(4) 
        blackboard.items = items             # ✓ Triggers callbacks
    """
    
    # Event types
    ON_SET: ClassVar[str] = "set"
    ON_DELETE: ClassVar[str] = "delete"
    ON_EXPIRE: ClassVar[str] = "expire"
    
    _data: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _ttl_data: Dict[str, float] = PrivateAttr(default_factory=dict)  # key -> expiration time
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _callbacks: List[Callable[[str, Any, str], None]] = PrivateAttr(default_factory=list)  # (key, value, event)
    _scopes: Dict[str, 'BlackboardScope'] = PrivateAttr(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def register_callback(self, callback: Callable[[str, Any, str], None]) -> bool:
        """
        Register a callback for all blackboard events.
        
        Args:
            callback: Function to call with (key, value, event_type) when any event occurs
            
        Returns:
            bool: True if callback was registered, False if already exists
        """
        if callback in self._callbacks:
            return False
        self._callbacks.append(callback)
        return True
    
    def unregister_callback(self, callback: Callable[[str, Any, str], None]) -> bool:
        """
        Unregister a callback for blackboard events.
        
        Args:
            callback: Function to remove
            
        Returns:
            bool: True if callback was removed, False if not found
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def _trigger_callbacks(self, key: str, value: Any, event: str) -> None:
        """Trigger all callbacks for an event."""
        for callback in self._callbacks:
            try:
                callback(key, value, event)
            except Exception:
                pass  # Don't let callback errors break the blackboard
    
    def _cleanup_expired(self) -> List[str]:
        """Remove expired keys and return their names."""
        now = time.time()
        expired_keys = []
        
        for key, expires_at in list(self._ttl_data.items()):
            if now >= expires_at:
                expired_keys.append(key)
                value = self._data.get(key)
                
                # Trigger expire callback before removing
                self._trigger_callbacks(key, value, self.ON_EXPIRE)
                
                # Remove from both data and TTL tracking
                self._data.pop(key, None)
                del self._ttl_data[key]
                        
        return expired_keys
    
    def set_with_ttl(self, key: str, value: Any, ttl: float) -> None:
        """
        Set a value with time-to-live expiration.
        
        Args:
            key: The attribute name
            value: The value to set
            ttl: Time to live in seconds
        """
        with self._lock:
            self._cleanup_expired()
            old_value = self._data.get(key)
            self._data[key] = value
            self._ttl_data[key] = time.time() + ttl
            self._trigger_callbacks(key, value, self.ON_SET)
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the blackboard.
        
        Args:
            key: The attribute name to delete
            
        Returns:
            bool: True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._data:
                value = self._data[key]
                del self._data[key]
                self._ttl_data.pop(key, None)
                self._trigger_callbacks(key, value, self.ON_DELETE)
                return True
        return False
    
    def clear(self) -> int:
        """
        Clear all data from the blackboard.
        
        Returns:
            int: Number of keys that were cleared
        """
        with self._lock:
            count = len(self._data)
            for key, value in self._data.items():
                self._trigger_callbacks(key, value, self.ON_DELETE)
            self._data.clear()
            self._ttl_data.clear()
            return count
    
    def keys(self) -> List[str]:
        """Get all keys in the blackboard (excluding expired ones)."""
        with self._lock:
            self._cleanup_expired()
            return list(self._data.keys())
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the blackboard."""
        with self._lock:
            self._cleanup_expired()
            return key in self._data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the blackboard.
        
        Returns:
            Dict with stats including total keys, expired keys, and TTL breakdown
        """
        with self._lock:
            expired_keys = self._cleanup_expired()
            total_keys = len(self._data)
            ttl_keys = len(self._ttl_data)
            
            return {
                "total_keys": total_keys,
                "ttl_keys": ttl_keys,
                "permanent_keys": total_keys - ttl_keys,
                "expired_this_check": len(expired_keys)
            }
    
    def scope(self, namespace: str) -> 'BlackboardScope':
        """
        Create a scoped view of the blackboard.
        
        Args:
            namespace: The scope identifier
            
        Returns:
            BlackboardScope: A scoped view that prefixes all keys
        """
        if namespace not in self._scopes:
            self._scopes[namespace] = BlackboardScope(self, namespace)
        return self._scopes[namespace]
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Override setattr to trigger callbacks for non-private attributes."""
        if key.startswith('_') or key in self.__fields__ or key in self.__pydantic_private__:
            # Private attributes, model fields, or pydantic internals - use normal behavior
            super().__setattr__(key, value)
        else:
            # Public attributes - store in _data and trigger callbacks
            with self._lock:
                self._cleanup_expired()
                old_value = self._data.get(key)
                self._data[key] = value
                # Remove from TTL tracking if it was there (now permanent)
                self._ttl_data.pop(key, None)
                self._trigger_callbacks(key, value, self.ON_SET)
    
    def __getattr__(self, key: str) -> Any:
        """Override getattr to retrieve from _data storage."""
        if key.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
        with self._lock:
            self._cleanup_expired()
            if key in self._data:
                return self._data[key]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __delattr__(self, key: str) -> None:
        """Override delattr to remove from _data and trigger callbacks."""
        if key.startswith('_') or key in self.__fields__:
            super().__delattr__(key)
        else:
            with self._lock:
                if key in self._data:
                    value = self._data[key]
                    del self._data[key]
                    self._ttl_data.pop(key, None)
                    self._trigger_callbacks(key, value, self.ON_DELETE)
                else:
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


class BlackboardScope:
    """
    A scoped view of a blackboard that prefixes all keys with a namespace.
    Provides the same interface as Blackboard but operates on prefixed keys.
    """
    
    def __init__(self, blackboard: Blackboard, namespace: str):
        self._blackboard = blackboard
        self._namespace = namespace
        self._prefix = f"{namespace}."
    
    def _scoped_key(self, key: str) -> str:
        """Convert a local key to a scoped key."""
        return f"{self._prefix}{key}"
    
    def register_callback(self, callback: Callable[[str, Any, str], None]) -> bool:
        """Register a callback that only fires for this scope's keys."""
        def scoped_callback(key: str, value: Any, event: str) -> None:
            if key.startswith(self._prefix):
                # Strip the prefix for the callback
                local_key = key[len(self._prefix):]
                callback(local_key, value, event)
        
        return self._blackboard.register_callback(scoped_callback)
    
    def set_with_ttl(self, key: str, value: Any, ttl: float) -> None:
        """Set a value with TTL in this scope."""
        self._blackboard.set_with_ttl(self._scoped_key(key), value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete a key from this scope."""
        return self._blackboard.delete(self._scoped_key(key))
    
    def clear(self) -> int:
        """Clear all keys in this scope."""
        count = 0
        for key in list(self._blackboard.keys()):
            if key.startswith(self._prefix):
                self._blackboard.delete(key)
                count += 1
        return count
    
    def keys(self) -> List[str]:
        """Get all keys in this scope (without prefix)."""
        all_keys = self._blackboard.keys()
        return [key[len(self._prefix):] for key in all_keys if key.startswith(self._prefix)]
    
    def has(self, key: str) -> bool:
        """Check if a key exists in this scope."""
        return self._blackboard.has(self._scoped_key(key))
    
    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            setattr(self._blackboard, self._scoped_key(key), value)
    
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        return getattr(self._blackboard, self._scoped_key(key))
    
    def __delattr__(self, key: str) -> None:
        if key.startswith('_'):
            super().__delattr__(key)
        else:
            delattr(self._blackboard, self._scoped_key(key))