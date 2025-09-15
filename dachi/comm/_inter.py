from __future__ import annotations

# 1st party
import uuid
import time
import threading
from typing import TypeVar, Generic, Callable, Dict, List, Optional, TypedDict, Any, ClassVar
from collections import defaultdict

# 3rd party
from pydantic import BaseModel, Field, PrivateAttr

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
    filtering capabilities, releasing locked items, expiration, callbacks, and scoping.
    
    Essential for inter-agent communication, task coordination, and message passing
    in multi-agent systems and behavior trees. Provides thread-safe operations with
    scoping to prevent cross-agent interference.
    
    Usage Patterns:
    
    1. **Basic Post → Retrieve → Ack Pattern**:
        ```python
        from pydantic import BaseModel
        
        class TaskRequest(BaseModel):
            task_id: str
            priority: int
            payload: str
        
        # Agent A posts a task request
        bulletin = Bulletin[TaskRequest]()
        request = TaskRequest(task_id="task_001", priority=1, payload="process data")
        post_id = bulletin.publish(request, lock=True)
        
        # Agent B retrieves the locked request
        post = bulletin.retrieve_first(item_type=TaskRequest)
        if post and not post["locked"]:
            # Process the task
            process_task(post["item"])
            # Acknowledge completion by removing the post
            bulletin.remove(post["id"])
        ```
    
    2. **Scoped Communication (Multi-tenant safety)**:
        ```python
        # Each agent uses its own scope to avoid interference
        agent_scope = "agent_alpha"
        
        # Publish within agent's scope
        bulletin.publish(message, scope=agent_scope)
        
        # Retrieve only messages within scope
        posts = bulletin.retrieve_all(scope=agent_scope)
        
        # Other agents in different scopes won't see these messages
        ```
    
    3. **Event-driven Communication with Callbacks**:
        ```python
        def on_bulletin_event(post, event_type):
            if event_type == Bulletin.ON_PUBLISH:
                print(f"New message posted: {post['item']}")
            elif event_type == Bulletin.ON_EXPIRE:
                print(f"Message expired: {post['id']}")
        
        bulletin.register_callback(on_bulletin_event)
        
        # Messages with TTL for automatic cleanup
        bulletin.publish(message, ttl=300)  # Expires in 5 minutes
        ```
    
    4. **Filtered Retrieval for Specific Tasks**:
        ```python
        # Retrieve high-priority tasks only
        high_priority = bulletin.retrieve_all(
            filter_func=lambda task: task.priority >= 8,
            item_type=TaskRequest,
            include_locked=False
        )
        
        # Retrieve oldest pending task
        oldest_task = bulletin.retrieve_first(
            order_func=lambda task: task.timestamp,
            include_locked=False
        )
        ```
    
    Thread Safety:
        All operations are thread-safe using internal RLock. Multiple agents
        can safely post/retrieve concurrently without data corruption.
        
    Singleton + Namespace Notes:
        While Bulletin instances are independent, consider using scoping
        when sharing instances across agents to prevent message leakage.
        Use unique scope identifiers (e.g., agent IDs) for isolation.
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
        """Register a callback for all bulletin events."""
        if callback in self._callbacks:
            return False
        self._callbacks.append(callback)
        return True
    
    def unregister_callback(self, callback: Callable[[Post, str], None]) -> bool:
        """Unregister a callback for bulletin events."""
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
    
    def _scoped_id(self, local_id: str, scope: Optional[str]) -> str:
        """Convert local ID to scoped ID if scope provided."""
        return f"{scope}.{local_id}" if scope else local_id
    
    def _local_id(self, scoped_id: str, scope: Optional[str]) -> str:
        """Convert scoped ID back to local ID if scope provided."""
        if scope and scoped_id.startswith(f"{scope}."):
            return scoped_id[len(scope) + 1:]
        return scoped_id
    
    def post(self, item: T, lock: bool = True, ttl: Optional[float] = None, scope: Optional[str] = None) -> str:
        """Publish an item to the bulletin board."""
        if not isinstance(item, BaseModel):
            raise TypeError("Published items must be BaseModel instances")
            
        local_id = str(uuid.uuid4())
        post_id = self._scoped_id(local_id, scope)
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
            self._trigger_callbacks(self.ON_PUBLISH, post)
            
        return local_id if scope else post_id
    
    def get_first(
        self, 
        id: Optional[str] = None,
        filter_func: Optional[Callable[[T], bool]] = None,
        order_func: Optional[Callable[[T], Any]] = None,
        item_type: Optional[type] = None,
        scope: Optional[str] = None
    ) -> Optional[Post]:
        """Retrieve the first matching post."""
        with self._lock:
            self._cleanup_expired_posts()
            
            if id is not None:
                scoped_id = self._scoped_id(id, scope)
                post = self._posts.get(scoped_id)
                if post and scope:
                    # Return with local ID
                    local_post = post.copy()
                    local_post["id"] = id
                    return local_post
                return post
            
            # Get candidate posts
            posts_to_check = []
            if item_type is not None:
                post_ids = self._type_registry.get(item_type, [])
                posts_to_check = [self._posts[pid] for pid in post_ids if pid in self._posts]
            else:
                posts_to_check = list(self._posts.values())
            
            # Filter by scope if specified
            if scope is not None:
                scope_prefix = f"{scope}."
                posts_to_check = [p for p in posts_to_check if p["id"].startswith(scope_prefix)]
            
            # Filter by function
            if filter_func is not None:
                posts_to_check = [p for p in posts_to_check if filter_func(p["item"])]
            
            if not posts_to_check:
                return None
                
            # Sort posts
            if order_func is not None:
                posts_to_check.sort(key=lambda p: order_func(p["item"]))
            else:
                posts_to_check.sort(key=lambda p: p["timestamp"])
                
            post = posts_to_check[0]
            if scope and post["id"].startswith(f"{scope}."):
                # Return with local ID
                local_post = post.copy()
                local_post["id"] = self._local_id(post["id"], scope)
                return local_post
            return post
    
    def get_all(
        self,
        filter_func: Optional[Callable[[T], bool]] = None,
        order_func: Optional[Callable[[T], Any]] = None,
        item_type: Optional[type] = None,
        include_locked: bool = True,
        scope: Optional[str] = None
    ) -> List[Post]:
        """Retrieve all matching posts."""
        with self._lock:
            self._cleanup_expired_posts()
            
            # Get candidate posts
            posts_to_check = []
            if item_type is not None:
                post_ids = self._type_registry.get(item_type, [])
                posts_to_check = [self._posts[pid] for pid in post_ids if pid in self._posts]
            else:
                posts_to_check = list(self._posts.values())
            
            # Filter by scope if specified
            if scope is not None:
                scope_prefix = f"{scope}."
                posts_to_check = [p for p in posts_to_check if p["id"].startswith(scope_prefix)]
            
            # Apply filters
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
                results.sort(key=lambda p: p["timestamp"])
            
            # Convert to local IDs if scoped
            if scope:
                for post in results:
                    if post["id"].startswith(f"{scope}."):
                        post["id"] = self._local_id(post["id"], scope)
                    
        return results
    
    def release(self, id: str, scope: Optional[str] = None) -> bool:
        """Release a locked post."""
        scoped_id = self._scoped_id(id, scope)
        with self._lock:
            if scoped_id in self._posts:
                self._posts[scoped_id]["locked"] = False
                self._trigger_callbacks(self.ON_RELEASE, self._posts[scoped_id])
                return True
        return False
    
    def remove(self, id: str, scope: Optional[str] = None) -> bool:
        """Remove a post from the bulletin board."""
        scoped_id = self._scoped_id(id, scope)
        with self._lock:
            if scoped_id in self._posts:
                post = self._posts[scoped_id]
                item_type = type(post["item"])
                
                # Trigger remove callback before removing
                self._trigger_callbacks(self.ON_REMOVE, post)
                
                # Remove from posts
                del self._posts[scoped_id]
                
                # Remove from type registry
                if item_type in self._type_registry:
                    try:
                        self._type_registry[item_type].remove(scoped_id)
                        if not self._type_registry[item_type]:
                            del self._type_registry[item_type]
                    except ValueError:
                        pass  # Already removed
                        
                return True
        return False
    
    def clear(self, scope: Optional[str] = None) -> int:
        """Clear all posts (or posts in a specific scope)."""
        with self._lock:
            if scope is None:
                # Clear everything
                count = len(self._posts)
                for post in self._posts.values():
                    self._trigger_callbacks(self.ON_REMOVE, post)
                self._posts.clear()
                self._type_registry.clear()
                return count
            else:
                # Clear only scoped posts
                scope_prefix = f"{scope}."
                posts_to_remove = [
                    (pid, post) for pid, post in self._posts.items() 
                    if pid.startswith(scope_prefix)
                ]
                
                for post_id, post in posts_to_remove:
                    self._trigger_callbacks(self.ON_REMOVE, post)
                    del self._posts[post_id]
                    
                    # Remove from type registry
                    item_type = type(post["item"])
                    if item_type in self._type_registry:
                        try:
                            self._type_registry[item_type].remove(post_id)
                            if not self._type_registry[item_type]:
                                del self._type_registry[item_type]
                        except ValueError:
                            pass
                
                return len(posts_to_remove)
    
    def get_stats(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the bulletin board."""
        with self._lock:
            if scope is None:
                # Global stats
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
            else:
                # Scoped stats
                scope_prefix = f"{scope}."
                scope_posts = [
                    post for post_id, post in self._posts.items() 
                    if post_id.startswith(scope_prefix)
                ]
                
                total_posts = len(scope_posts)
                locked_posts = sum(1 for post in scope_posts if post["locked"])
                
                return {
                    "total_posts": total_posts,
                    "locked_posts": locked_posts,
                    "available_posts": total_posts - locked_posts,
                    "scope": scope
                }


class Blackboard(BaseModel):
    """Shared state storage with reactive callbacks for behavior trees and agents."""
    
    # Event types
    ON_SET: ClassVar[str] = "set"
    ON_DELETE: ClassVar[str] = "delete"
    ON_EXPIRE: ClassVar[str] = "expire"
    
    _data: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _ttl_data: Dict[str, float] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _callbacks: List[Callable[[str, Any, str], None]] = PrivateAttr(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def register_callback(self, callback: Callable[[str, Any, str], None]) -> bool:
        """Register a callback for all blackboard events."""
        if callback in self._callbacks:
            return False
        self._callbacks.append(callback)
        return True
    
    def unregister_callback(self, callback: Callable[[str, Any, str], None]) -> bool:
        """Unregister a callback for blackboard events."""
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
                self._trigger_callbacks(key, value, self.ON_EXPIRE)
                self._data.pop(key, None)
                del self._ttl_data[key]
                        
        return expired_keys
    
    def _scoped_key(self, key: str, scope: Optional[str]) -> str:
        """Convert key to scoped key if scope provided."""
        return f"{scope}.{key}" if scope else key
    
    def _local_key(self, scoped_key: str, scope: Optional[str]) -> str:
        """Convert scoped key back to local key if scope provided."""
        if scope and scoped_key.startswith(f"{scope}."):
            return scoped_key[len(scope) + 1:]
        return scoped_key
    
    def set_with_ttl(self, key: str, value: Any, ttl: float, scope: Optional[str] = None) -> None:
        """Set a value with time-to-live expiration."""
        scoped_key = self._scoped_key(key, scope)
        with self._lock:
            self._cleanup_expired()
            self._data[scoped_key] = value
            self._ttl_data[scoped_key] = time.time() + ttl
            self._trigger_callbacks(scoped_key, value, self.ON_SET)
    
    def get(self, key: str, default: Any = None, scope: Optional[str] = None) -> Any:
        """Get a value from the blackboard."""
        scoped_key = self._scoped_key(key, scope)
        with self._lock:
            self._cleanup_expired()
            return self._data.get(scoped_key, default)
    
    def delete(self, key: str, scope: Optional[str] = None) -> bool:
        """Delete a key from the blackboard."""
        scoped_key = self._scoped_key(key, scope)
        with self._lock:
            if scoped_key in self._data:
                value = self._data[scoped_key]
                del self._data[scoped_key]
                self._ttl_data.pop(scoped_key, None)
                self._trigger_callbacks(scoped_key, value, self.ON_DELETE)
                return True
        return False
    
    def clear(self, scope: Optional[str] = None) -> int:
        """Clear all data (or data in a specific scope)."""
        with self._lock:
            if scope is None:
                count = len(self._data)
                for key, value in self._data.items():
                    self._trigger_callbacks(key, value, self.ON_DELETE)
                self._data.clear()
                self._ttl_data.clear()
                return count
            else:
                scope_prefix = f"{scope}."
                keys_to_remove = [
                    key for key in self._data.keys() 
                    if key.startswith(scope_prefix)
                ]
                
                for key in keys_to_remove:
                    value = self._data[key]
                    self._trigger_callbacks(key, value, self.ON_DELETE)
                    del self._data[key]
                    self._ttl_data.pop(key, None)
                
                return len(keys_to_remove)
    
    def keys(self, scope: Optional[str] = None) -> List[str]:
        """Get all keys (excluding expired ones)."""
        with self._lock:
            self._cleanup_expired()
            if scope is None:
                return list(self._data.keys())
            else:
                scope_prefix = f"{scope}."
                return [
                    self._local_key(key, scope)
                    for key in self._data.keys() 
                    if key.startswith(scope_prefix)
                ]
    
    def has(self, key: str, scope: Optional[str] = None) -> bool:
        """Check if a key exists."""
        scoped_key = self._scoped_key(key, scope)
        with self._lock:
            self._cleanup_expired()
            return scoped_key in self._data
    
    def get_stats(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the blackboard."""
        with self._lock:
            self._cleanup_expired()
            
            if scope is None:
                total_keys = len(self._data)
                ttl_keys = len(self._ttl_data)
                return {
                    "total_keys": total_keys,
                    "ttl_keys": ttl_keys,
                    "permanent_keys": total_keys - ttl_keys,
                }
            else:
                scope_prefix = f"{scope}."
                scoped_keys = [k for k in self._data.keys() if k.startswith(scope_prefix)]
                scoped_ttl_keys = [k for k in self._ttl_data.keys() if k.startswith(scope_prefix)]
                
                total_keys = len(scoped_keys)
                ttl_keys = len(scoped_ttl_keys)
                
                return {
                    "total_keys": total_keys,
                    "ttl_keys": ttl_keys,
                    "permanent_keys": total_keys - ttl_keys,
                    "scope": scope
                }
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Override setattr to trigger callbacks for non-private attributes."""
        if key.startswith('_') or key in self.__fields__ or key in self.__pydantic_private__:
            super().__setattr__(key, value)
        else:
            # For blackboard, we can't automatically scope __setattr__ since we don't know the scope
            # Users should use set_with_ttl() method for scoped access
            with self._lock:
                self._cleanup_expired()
                self._data[key] = value
                self._ttl_data.pop(key, None)  # Remove TTL if present
                self._trigger_callbacks(key, value, self.ON_SET)
    
    def __getattr__(self, key: str) -> Any:
        """Override getattr to retrieve from _data storage."""
        if key.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
        # For blackboard, we can't automatically scope __getattr__ since we don't know the scope
        # Users should use get() method for scoped access
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
            # For blackboard, we can't automatically scope __delattr__ since we don't know the scope
            # Users should use delete() method for scoped access
            with self._lock:
                if key in self._data:
                    value = self._data[key]
                    del self._data[key]
                    self._ttl_data.pop(key, None)
                    self._trigger_callbacks(key, value, self.ON_DELETE)
                else:
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
