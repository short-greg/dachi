
# 1st party
from __future__ import annotations
import typing
import typing as t
from abc import abstractmethod
from typing import Self, Literal
import base64

# 3rd party
from pydantic import PrivateAttr
import pydantic

# local
from . import Renderable


Msg: t.TypeAlias = t.Dict | pydantic.BaseModel

# class FieldRenderer(object):

#     def __init__(self, field: str='text'):
#         """Renderer to render a specific field in the message

#         Args:
#             field (str, optional): The field name. Defaults to 'text'.
#         """
#         self.field = field

#     def __call__(self, msg: t.Union[Msg, BaseDialog]) -> str:
#         """Render a message

#         Args:
#             msg (Msg): The message to render

#         Returns:
#             str: The result
#         """
#         if isinstance(msg, BaseDialog):
#             messages = list(msg)
#         else:
#             messages = [msg]
#         return '\n'.join(
#             f'{m.role}: {getattr(m, self.field, "")}'
#             for m in messages
#         )


class BaseDialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """
    _renderer: typing.Callable[[typing.List[Msg]], str] = pydantic.PrivateAttr(
        default=None
    )

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        pass

    @abstractmethod
    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        pass

    @abstractmethod
    def pop(self, index: int=-1) -> Msg:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        pass

    @abstractmethod
    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        pass

    def append(self, message: Msg) -> 'BaseDialog':
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add

        Raises:
            ValueError: If the index is not correct
        """
        return self.insert(
            message=message, ind=None
        )

    def add(self, message: Msg) -> Msg:
        """Alias for append to add a message to the end of the current path."""
        self.append(message)
        return message

    @abstractmethod
    def replace(
        self, message: Msg, ind: int
    ) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (int): The index to add at

        Raises:
            ValueError: If the index is not correct
        """
        pass

    @abstractmethod
    def insert(self, message: Msg, ind: int) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (int): The index to add at

        Raises:
            ValueError: If the index is not correct
        """
        pass

    @abstractmethod
    def extend(
        self, dialog: t.Union['BaseDialog', t.Iterable[Msg]], 
        _inplace: bool=False
    ) -> 'BaseDialog':
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog: The dialog or list of messages to extend with
        """
        pass

    def render(self) -> str:
        """Render the dialog as a series of turns 
        <role>: <text>

        Returns:
            str: The dialog
        """
        if self._renderer is not None:
            return self._renderer(self)
        return '\n'.join(
            message.render()
            for message in self
        )

    def aslist(self) -> t.List['Msg']:
        """Retrieve the message list

        Returns:
            typing.List[Msg]: the messages in the dialog
        """
        return list(self)
    
    @abstractmethod
    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        pass
        
    @abstractmethod
    def clone(self) -> 'BaseDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        pass

    @abstractmethod
    def spawn(self) -> 'BaseDialog':
        """Create a new empty dialog of the same type as the current dialog.

        Returns:
            Dialog: A new empty dialog of the same type
        """
        pass


class ListDialog(BaseDialog):
    """A Dialog that uses a list data structure.
    """

    messages: t.List[Msg] = pydantic.Field(default_factory=list)

    def __iter__(self) -> t.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self.messages:
            yield message

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        return self.messages[idx]

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        if idx == len(self.messages):
            self.messages.append(message)
        else:
            self.messages[idx] = message
        return self

    def clone(self) -> 'ListDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        
        return ListDialog(
            messages=[*self.messages]
        )
    
    def spawn(self) -> 'ListDialog':
        """Create a new empty dialog of the same type as the current dialog.

        Returns:
            Dialog: A new empty dialog of the same type
        """
        return ListDialog(messages=[])

    def pop(self, index: int=-1, get_msg: bool=False) -> t.Union['ListDialog', Msg]:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        msg = self.messages.pop(index)
        if get_msg:
            return self.messages, msg
        return msg

    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        self.messages.remove(message)

    def extend(self, dialog: t.Iterable[t.Union[Msg]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog: The dialog or list of messages to extend with
        """
        if isinstance(dialog, BaseDialog):
            dialog = dialog.aslist()
        
        validated = []
        for msg in dialog:
            validated.append(msg)

        self.messages.extend(validated)
        return self

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self.messages)
        
    def append(self, message: Msg) -> Self:
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add

        Raises:
            ValueError: If the index is not correct
        """
        self.messages.append(message)
        return self

    def insert(self, ind: int, message: Msg) -> Self:
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (int): The index to add at

        Raises:
            ValueError: If the index is not correct
        """
        self.messages.insert(ind, message)
        return self

    def replace(self, idx: int, message: Msg) -> 'BaseDialog':
        """Replace a message in the dialog

        Args:
            idx (int): The index to replace at
            message (Msg): The message to add

        Raises:
            ValueError: If the index is not correct
        """
        self.messages[idx] = message
        return self


def exclude_messages(
    dialog: BaseDialog, 
    val: t.Union[t.Any, t.Set], 
    field='role'
) -> ListDialog:
    """Exclude messages from the dialog

    Args:
        dialog (BaseDialog): The dialog to filter
        val (typing.Union[typing.Any, typing.Set]): The value to exclude based on
        field (str, optional): The field to exclude based on. Defaults to 'role'.

    Returns:
        ListDialog: The resulting dialog
    """
    if not isinstance(val, t.Set):
        val = {val}

    return ListDialog(
        messages=[msg for msg in dialog if getattr(msg, field) not in val]
    )

            
def include_messages(
    dialog: BaseDialog, 
    val: t.Union[t.Any, t.Set], 
    field='role'
) -> ListDialog:
    """Include messages in the resulting dialog

    Args:
        dialog (BaseDialog): The dialog to filter
        val (typing.Union[typing.Any, typing.Set]): The value to include based on
        field (str, optional): The field to include based on. Defaults to 'role'.

    Returns:
        ListDialog: The resulting dialog
    """
    if not isinstance(val, t.Set):
        val = {val}

    return ListDialog(
        messages=[msg for msg in dialog if getattr(msg, field) in val]
    )


def to_dialog(prompt: Msg) -> BaseDialog:
    """Convert a prompt to a dialog

    Args:
        prompt (Msg): The prompt to convert
    """
    if not isinstance(prompt, BaseDialog):
        prompt = ListDialog(messages=[prompt])

    return prompt


def exclude_role(
    messages: t.Iterable[Msg], 
    *role: str
) -> t.List[Msg]:
    """
    Filter messages by excluding specified roles.
    
    Args:
        messages (typing.Iterable[Msg]): An iterable of message objects
        *role (str): Variable number of role strings to exclude
    Returns:
        typing.List[Msg]: A list of messages excluding those with specified roles
    Example:
        >>> messages = [Msg(role="user", text="hi"), Msg(role="system", text="hello")]
        >>> exclude_role(messages, "system")
        [Msg(role="user", text="hi")]
    """
    exclude = set(role)
    return [message for message in messages
        if message.role not in exclude]


def include_role(
    messages: t.Iterable[Msg],
    *role: str
) -> t.List[Msg]:
    """Filter the iterable of messages by a particular role

    Args:
        messages (typing.Iterable[Msg]): Messages to filter

    Returns:
        typing.List[Msg]: Filtered messages
    """
    include = set(role)
    return [message for message in messages
        if message.role in include]


class TreeDialog(BaseDialog):
    """A tree-based dialog implementation using node ID mappings instead of DialogTurn objects.
    
    This implementation provides the same interface as TreeDialog but uses a more efficient
    mapping-based approach internally. Messages are stored in a tree structure where each
    node has a unique ID and relationships are maintained through parent/child mappings.
    
    Key Features:
        - Tree structure for conversation branching
        - Efficient node-based navigation
        - Same API as TreeDialog for compatibility
        - Support for insertion, replacement, and removal operations
        - Navigation through rise(), leaf_child(), leaf_sibling() methods
        - Maintains indices and counts for path tracking
        
    Internal Structure:
        - _messages: Maps node IDs to Msg objects
        - _parent: Maps child node IDs to parent node IDs  
        - _children: Maps parent node IDs to lists of child node IDs
        - _root: ID of the root node
        - _leaf: ID of the current leaf node being tracked
        - _indices: Current path indices from root to leaf
        - _counts: Number of children at each level in current path
    
    Usage:
        dialog = TreeDialog()
        dialog.append(Msg(role="user", text="Hello"))
        dialog.append(Msg(role="assistant", text="Hi there!"))
        
        # Navigate and branch
        dialog.rise(1)  # Go up one level
        dialog.append(Msg(role="assistant", text="Alternative response"))
        
        # Access messages
        for msg in dialog:
            print(msg.render())
    """
    _root: t.Optional[str] = PrivateAttr(default=None)
    _messages: t.Dict[str, t.Dict | pydantic.BaseModel] = PrivateAttr(default_factory=dict)
    _leaf: t.Optional[str] = PrivateAttr(default=None)
    _parent: t.Dict[str, str] = PrivateAttr(default_factory=dict)
    _children: t.Dict[str, t.List[str]] = PrivateAttr(default_factory=dict)
    _indices: t.List[int] = PrivateAttr(default_factory=list)
    _counts: t.List[int] = PrivateAttr(default_factory=list)
    _next_id: int = PrivateAttr(default=0)
        
    def model_post_init(self, __context) -> None:
        """Initialize after model validation."""
        super().model_post_init(__context)
        self._update()
        
    def _generate_id(self) -> str:
        """Generate a unique node ID."""
        node_id = str(self._next_id)
        self._next_id += 1
        return node_id
        
    def _get_path_to_leaf(self) -> t.List[str]:
        """Get the path from root to current leaf as a list of node IDs."""
        if self._leaf is None:
            return []
        
        path = []
        current = self._leaf
        while current is not None:
            path.insert(0, current)
            current = self._parent.get(current)
        return path
        
    def _update_indices(self):
        """Update the indices array based on current leaf position."""
        if self._leaf is None:
            self._indices = []
            return
            
        path = self._get_path_to_leaf()
        indices = []
        
        for i in range(len(path)):
            if i == 0:
                indices.append(0)  # Root is always at index 0
            else:
                parent_id = path[i-1]
                child_id = path[i]
                children = self._children.get(parent_id, [])
                try:
                    idx = children.index(child_id)
                    indices.append(idx)
                except ValueError:
                    indices.append(0)
                    
        self._indices = indices
        
    def _update_counts(self):
        """Update the counts array based on current path."""
        if self._leaf is None:
            self._counts = []
            return
            
        path = self._get_path_to_leaf()
        counts = []
        
        for node_id in path:
            children = self._children.get(node_id, [])
            if len(children) == 0:
                counts.append(1)  # Leaf nodes count as 1
            else:
                counts.append(len(children))
                
        self._counts = counts
        
    def _update(self):
        """Update internal state after modifications."""
        self._update_indices()
        self._update_counts()
        
    @property
    def indices(self) -> t.List[int]:
        """Get a copy of the current path indices."""
        return [*self._indices]
        
    @property
    def counts(self) -> t.List[int]:
        """Get a copy of the current level counts."""
        return [*self._counts]
        
    @property
    def root(self) -> t.Optional[Msg]:
        """Get the root message."""
        if self._root is None:
            return None
        return self._messages[self._root]
        
    @property
    def leaf(self) -> t.Optional[Msg]:
        """Get the current leaf message."""
        if self._leaf is None:
            return None
        return self._messages[self._leaf]
        
    def rise(self, count: int):
        """Move the leaf pointer up by count levels."""
        if self._leaf is None:
            return
            
        current = self._leaf
        for _ in range(count):
            parent = self._parent.get(current)
            if parent is None:
                break
            current = parent
            
        self._leaf = current
        self._update()
        
    def leaf_child(self, idx: int):
        """Move to the specified child of the current leaf."""
        if self._leaf is None:
            return
            
        children = self._children.get(self._leaf, [])
        if 0 <= idx < len(children):
            self._leaf = children[idx]
            self._update()
            
    def leaf_sibling(self, idx: int):
        """Move to a sibling of the current leaf."""
        if self._leaf is None or self._leaf == self._root:
            return
            
        parent_id = self._parent.get(self._leaf)
        if parent_id is None:
            return
            
        siblings = self._children.get(parent_id, [])
        current_idx = siblings.index(self._leaf) if self._leaf in siblings else 0
        new_idx = current_idx + idx
        
        if 0 <= new_idx < len(siblings):
            self._leaf = siblings[new_idx]
            self._update()

    def __iter__(self) -> t.Iterator[Msg]:
        """Iterate over messages from root to current leaf."""
        if self._leaf is None:
            return
            
        path = self._get_path_to_leaf()
        for node_id in path:
            yield self._messages[node_id]

    def __getitem__(self, idx) -> Msg:
        """Get message at the specified index in the current path."""
        path = self._get_path_to_leaf()
        if 0 <= idx < len(path):
            return self._messages[path[idx]]
        raise IndexError("Index out of range")

    def __setitem__(self, idx, message) -> Self:
        """Set message at the specified index in the current path."""
        path = self._get_path_to_leaf()
        if 0 <= idx < len(path):
            self._messages[path[idx]] = message
        else:
            raise IndexError("Index out of range")
        return self

    def __len__(self) -> int:
        """Get the length of the current path from root to leaf."""
        return len(self._get_path_to_leaf())

    def pop(self, index: int = -1) -> Msg:
        """Remove and return a message at the specified index.
        
        Cannot remove the root node (index 0).
        """
        if index == 0:
            raise ValueError('Cannot remove root node in tree dialog.')
            
        path = self._get_path_to_leaf()
        if index == -1:
            index = len(path) - 1
            
        if not (0 <= index < len(path)):
            raise IndexError("Index out of range")
            
        target_id = path[index]
        target_msg = self._messages[target_id]
        
        # Remove the node and reconnect children to parent
        parent_id = self._parent.get(target_id)
        children_ids = self._children.get(target_id, [])
        
        # Update parent's children list
        if parent_id is not None:
            parent_children = self._children.get(parent_id, [])
            if target_id in parent_children:
                parent_children.remove(target_id)
                # Add the removed node's children to parent
                parent_children.extend(children_ids)
        
        # Update children's parent pointers
        for child_id in children_ids:
            self._parent[child_id] = parent_id
            
        # Clean up the removed node
        del self._messages[target_id]
        if target_id in self._parent:
            del self._parent[target_id]
        if target_id in self._children:
            del self._children[target_id]
            
        # Update leaf if it was the removed node
        if self._leaf == target_id:
            if parent_id is not None:
                self._leaf = parent_id
            elif children_ids:
                self._leaf = children_ids[0]
            else:
                self._leaf = None
                
        self._update()
        return target_msg

    def remove(self, message: Msg):
        """Remove a message from the dialog."""
        # Find the node ID for this message
        target_id = None
        for node_id, msg in self._messages.items():
            if msg == message:
                target_id = node_id
                break
                
        if target_id is None:
            raise ValueError(f'Message {message} does not exist so cannot remove.')
            
        if target_id == self._root:
            raise ValueError('Cannot remove root node in tree dialog.')
            
        # Remove using the node ID
        path = self._get_path_to_leaf()
        try:
            index = path.index(target_id)
            self.pop(index)
        except ValueError:
            # If not in current path, do direct removal
            parent_id = self._parent.get(target_id)
            children_ids = self._children.get(target_id, [])
            
            if parent_id is not None:
                parent_children = self._children.get(parent_id, [])
                if target_id in parent_children:
                    parent_children.remove(target_id)
                    parent_children.extend(children_ids)
            
            for child_id in children_ids:
                self._parent[child_id] = parent_id
                
            del self._messages[target_id]
            if target_id in self._parent:
                del self._parent[target_id]
            if target_id in self._children:
                del self._children[target_id]
                
            if self._leaf == target_id:
                if parent_id is not None:
                    self._leaf = parent_id
                elif children_ids:
                    self._leaf = children_ids[0]
                else:
                    self._leaf = None
                    
        self._update()

    def extend(
        self, 
        dialog: t.Union['BaseDialog', t.Iterable[Msg]], 
        _inplace: bool = False
    ) -> 'BaseDialog':
        """Extend the dialog with messages from another dialog or iterable."""
        if isinstance(dialog, BaseDialog):
            messages = list(dialog)
        else:
            messages = list(dialog)
            
        for msg in messages:
            self.append(msg)
            
        return self

    def append(self, message: Msg) -> Self:
        """Add a message to the end of the current path."""
            
        node_id = self._generate_id()
        self._messages[node_id] = message
        
        if self._root is None:
            # First message becomes root
            self._root = node_id
            self._leaf = node_id
            self._children[node_id] = []
        else:
            # Add as child of current leaf
            self._parent[node_id] = self._leaf
            if self._leaf not in self._children:
                self._children[self._leaf] = []
            self._children[self._leaf].append(node_id)
            self._children[node_id] = []
            self._leaf = node_id
            
        self._update()
        return self

    def insert(self, ind: int, message: Msg) -> Self:
        """Insert a message at the specified index in the current path."""

        if self._leaf is None:
            if ind != 0:
                raise RuntimeError("Cannot insert at non-zero index in empty dialog")
            return self.append(message)
            
        path = self._get_path_to_leaf()
        
        if ind > len(path):
            raise IndexError(f"Index {ind} out of range for path of length {len(path)}")
        elif ind == len(path):
            # Append to end
            return self.append(message)
        elif ind == 0:
            # Insert before root - create new root
            new_id = self._generate_id()
            self._messages[new_id] = message
            
            # New node becomes root
            old_root = self._root
            self._root = new_id
            self._parent[old_root] = new_id
            self._children[new_id] = [old_root]
            
            self._update()
        else:
            # Insert in the middle - create branch
            target_id = path[ind]
            parent_id = self._parent.get(target_id)
            
            # Create new node
            new_id = self._generate_id()
            self._messages[new_id] = message
            
            # Insert between parent and target
            if parent_id is not None:
                parent_children = self._children.get(parent_id, [])
                target_index = parent_children.index(target_id)
                parent_children[target_index] = new_id
                
            self._parent[new_id] = parent_id
            self._parent[target_id] = new_id
            self._children[new_id] = [target_id]
            
            # Update root if necessary
            if target_id == self._root:
                self._root = new_id
                
            self._update()
            
        return self

    def replace(self, idx: int, message: Msg) -> 'BaseDialog':
        """Replace the message at the specified index."""
            
        path = self._get_path_to_leaf()
        if not (0 <= idx < len(path)):
            raise ValueError(f"Index {idx} out of range for dialog of length {len(path)}")
            
        node_id = path[idx]
        self._messages[node_id] = message
        self._update()
        return self

    def clone(self) -> 'TreeDialog':
        """Create a deep copy of the dialog structure with shallow message copying."""
        clone = TreeDialog()
        
        if self._root is None:
            return clone
            
        # Copy all the mappings
        clone._messages = self._messages.copy()
        clone._parent = self._parent.copy()
        clone._children = {k: v.copy() for k, v in self._children.items()}
        clone._root = self._root
        clone._leaf = self._leaf
        clone._next_id = self._next_id
        
        clone._update()
        return clone
    
    def spawn(self):
        return TreeDialog()
