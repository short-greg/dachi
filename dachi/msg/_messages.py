# 1st party
import typing
from abc import abstractmethod
from typing import Self

# local
from ..core import Renderable

class _Final:
    """A unique object to mark the end of a streaming response."""
    def __repr__(self):
        return "<Final Token>"

END_TOK = _Final()

NULL_TOK = object()


class Msg(dict):
    """A Msg used for a dialog
    """
    def __init__(
        self, role: str, type_: str='data', 
        meta: typing.Dict=None, delta: typing.Dict=None, 
        _include_role: bool=True, _follow_up: typing.List['Msg']=None, _filtered: bool=False, **kwargs
    ):
        """Create a Msg

        Args:
            type_ (str, optional): The type of message. Defaults to 'data'.
            meta (typing.Dict, optional): Any additional information not related to the message specifically. Defaults to None.
            delta (typing.Dict, optional): The change in the message. Defaults to None.
        """
        super().__init__(
            role=role, _include_role=_include_role, type_=type_, meta=meta or {}, delta=delta or {}, **kwargs
        )
        self.follow_up = _follow_up if _follow_up is not None else []
        self.filtered = _filtered

    @property
    def type(self) -> str:
        """Get the type of the message"""
        return self['type_']
    
    def __getattr__(self, key) -> typing.Any:
        """Get an attribute of the message"""
        return self.__getitem__(key)
    
    def __setattr__(self, name, value):
        """Set an attribute of the message"""
        return self.__setitem__(name, value)
    
    def to_input(self) -> typing.Dict:
        """Convert the message to a dictionary"""
        exclude = {'meta', 'delta', '_include_role', 'type_'}
        d = {k: v for k, v in self.items() if k not in exclude}
        if self['_include_role'] is False:
            del d['role']

        return d
    
    def meta_(self, **kwargs) -> Self:
        """Update the values of the meta table

        Returns:
            Self: The message after update
        """
        self['meta'].update(**kwargs)
        return self
    
    @property
    def m(self) -> typing.Dict:
        """Get the meta data from the message """
        return self['meta']

    def to_list_input(self) -> typing.List[typing.Dict]:
        """Convert to an input appropriate for a list

        Returns:
            typing.List[typing.Dict]: The message converted to a list
        """
        if self.filtered:
            return []
        return [self.to_input()]
    
    def render(self) -> str:
        """

        Returns:
            str: A string of the message
        """
        vals = {
            key: val for key, val in self.items() if key not in (
                'role', 'meta', '_include_role',
                'delta', 'type_'
            )
        }
        return f'{self.role} {vals}'


# class StreamMsg(Msg):
#     """A message that is streamed
#     """

#     def __init__(
#         self, role: str, type_: str='data', 
#         meta: typing.Dict=None, delta: typing.Dict=None, 
#         _include_role: bool=True, is_last: bool=False, **kwargs
#     ):
#         """Create a Stream Msg

#         Args:
#             type_ (str, optional): The type of message. Defaults to 'data'.
#             meta (typing.Dict, optional): Any additional information not related to the message specifically. Defaults to None.
#             delta (typing.Dict, optional): The change in the message. Defaults to None.
#         """
#         super().__init__(
#             role=role, type_=type_, meta=meta or {}, delta=delta or {},_include_role=_include_role, **kwargs
#         )
#         self.is_last = is_last


class BaseDialog(Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """

    def __init__(
        self, 
        renderer: typing.Callable[[typing.List[Msg]], str] | None=None
    ):
        super().__init__()
        self._renderer = renderer

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
    def pop(self, index: int) -> Msg:
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
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        return self.insert(message=message, ind=None)

    @abstractmethod
    def replace(self, message: Msg, ind: int) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        pass

    @abstractmethod
    def insert(self, message: Msg, ind: int) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        pass

    @abstractmethod
    def extend(
        self, dialog: typing.Union['BaseDialog', typing.Iterable[Msg]], 
        _inplace: bool=False
    ) -> 'BaseDialog':
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
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
    
    def to_input(self) -> typing.List[typing.Dict]:
        """Convert the dialog to an input to pass into an API

        Returns:
            typing.List[typing.Dict]: A list of inputs
        """
        return [msg.to_input() for msg in self if msg.filtered is False]

    def to_list_input(self) -> typing.List[typing.Dict]:

        return self.to_input()

    def aslist(self) -> typing.List['Msg']:
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
        # return len(self.messages)
        
    def clone(self) -> 'BaseDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        pass

    
def to_input(inp: typing.Union[typing.Iterable[Msg], Msg]) -> typing.Union[typing.List[Msg], Msg]:
    """Convert a list of messages or a single message to an input

    Args:
        inp (typing.Union[typing.Iterable[Msg], Msg]): The inputs to convert

    Returns:
        typing.Union[typing.List[Msg], Msg]: The input
    """
    if isinstance(inp, Msg):
        return inp.to_input() if inp.filtered is False else None
    
    return {msg.to_input() for msg in inp if msg.filtered is False}


class ListDialog(BaseDialog):
    """A Dialog that uses a list data structure.
    """

    def __init__(
        self, 
        messages: typing.Iterable[Msg]=None,
        renderer: typing.Callable[[typing.List[Msg]], str] = None
    ):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(renderer)
        self._messages = messages or []

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self._messages:
            yield message

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        return self._messages[idx]

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        if idx == len(self._messages):
            self._messages.append(message)
        else:
            self._messages[idx] = message
        return self

    def clone(self) -> 'ListDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        
        return ListDialog(
            [*self._messages]
        )

    def pop(self, index: int, get_msg: bool=False) -> typing.Union['ListDialog', Msg]:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        msg = self._messages.pop(index)
        if get_msg:
            return self._messages, msg
        return msg

    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        self._messages.remove(message)

    def extend(self, dialog: typing.Iterable[Msg]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        if isinstance(dialog, BaseDialog):
            dialog = dialog.aslist()
        
        self._messages.extend(dialog)
        return self

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self._messages)
        
    def append(self, message: Msg) -> Self:
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        self._messages.append(message)
        return self

    def insert(self, ind: int, message: Msg) -> Self:
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        self._messages.insert(ind, message)
        return self
    
    def replace(self, idx: int, message: Msg) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            idx (typing.Optional[int], optional): The index to add. Defaults to None.
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        self._messages[idx] = message
        return self


class DialogTurn(object):
    """A Dialog that uses a list data structure.
    """
    # _message: Msg = pydantic.PrivateAttr()
    # _parent: 'DialogTurn' = pydantic.PrivateAttr(default=None)
    # _children: typing.List['DialogTurn'] = pydantic.PrivateAttr(default_factory=list)

    def __init__(
        self, message: Msg, 
        parent: Msg=None, 
        children: typing.List[Msg]=None
    ):
        super().__init__()
        self.message = message
        self._parent = parent
        self._children = children or []

    def root(self) -> 'DialogTurn':

        node = self
        while node.parent is not None:
            node = node._parent
        return node

    def leaf(self) -> 'DialogTurn':
        """Get the leftmost leaf node below the current node

        Returns:
            DialogTurn: The leaf node
        """
        if len(self._children) == 0:
            return self
        return self._children[0].leaf()
    
    @property
    def ancestors(self) -> typing.Iterator['DialogTurn']:

        turn = self
        while True:
            if turn._parent is None:
                return
            turn = turn._parent
            yield turn
    
    def children(self) -> typing.Iterator['DialogTurn']:
        """get the children to the dialog turn

        Returns:
            typing.List['DialogTurn']: The children
        """
        return iter(self._children)
    
    def prepend(self, message: 'Msg') -> 'DialogTurn':
        """

        Args:
            turn (DialogTurn): 

        Returns:
            DialogTurn: 
        """
        turn = DialogTurn(message)
        turn._parent = self._parent
        turn._children.append(self)
        self._parent = turn
        return turn
    
    def append(self, message: 'Msg') -> 'DialogTurn':
        """

        Args:
            message (Msg): _description_

        Returns:
            DialogTurn: _description_
        """
        turn = DialogTurn(message)
        turn._parent = self
        self._children.append(turn)
        return turn

    def ancestor(self, count: int) -> 'DialogTurn':
        """Get an an answer

        Args:
            count (int): The number to ascend

        Returns:
            DialogTurn: The ancestor
        """

        i = 0
        turn = self
        while True:
            if i == count:
                return turn.message
            if turn._parent is None:
                raise RuntimeError(
                    "There are only "
                    f"{i} ancestors yet"
                    f"you passed {count}"
                )
            turn = turn._parent
            i += 1

    def depth(self) -> int:
        """Calculate the depth of the current turn in the dialog tree.

        Returns:
            int: The number of turns to the root turn.
        """
        depth = 1
        turn = self
        while turn._parent is not None:
            depth += 1
            turn = turn._parent
        return depth
    
    def child(self, idx: int) -> 'DialogTurn':
        """Get the child specified by index

        Args:
            idx (int): The index to retrieve

        Returns:
            DialogTurn: 
        """
        return self._children[idx]
    
    def find_val(self, message: Msg) -> typing.Optional['DialogTurn']:
        """Search through all children to find the message.

        Args:
        message (Msg): The message to find.

        Returns:
        typing.Optional[DialogTurn]: The DialogTurn containing the message, or None if not found.
        """
        result = None
        if self.message == message:
            return self

        for child in self._children:
            result = child.find_val(message)

        return result

    def find(self, turn: 'DialogTurn') -> typing.Optional['DialogTurn']:
        """Search through all children to find the message.

        Args:
        message (Msg): The message to find.

        Returns:
        typing.Optional[DialogTurn]: The DialogTurn containing the message, or None if not found.
        """
        result = None
        if self == turn:
            return self

        for child in self._children:
            result = child.find(turn)
        return result

    def prune(self, idx: int) -> 'DialogTurn':
        """Remove the subtree specified by idx.

        Args:
            idx (int): The index of the child to prune.

        Returns:
            DialogTurn: The pruned subtree.
        """
        if idx < 0 or idx >= len(self._children):
            raise IndexError("Index out of range for pruning.")

        pruned_subtree = self._children.pop(idx)
        pruned_subtree._parent = None
        return pruned_subtree
    
    def sibling(self, idx: int) -> int:
        """
        Returns:
            int: The index for the sibling
        """
        if self._parent is None:
            if idx != 0:
                raise RuntimeError(
                    "There is no parent so must be 1."
                )
            return self
        return self._parent.child(idx)
    
    def ancestor(self, count: int) -> 'DialogTurn':
        """

        Args:
            count (int): 

        Returns:
            DialogTurn: 
        """
        turn = self
        i = 0
        while True:
            if i == count:
                return turn
            turn = self._parent
            if turn is None:
                raise ValueError(
                    f"Cannot ascend {count}."
                    f"Only {i} parents."
                )
            i += 1
    
    @property
    def parent(self) -> typing.Union['DialogTurn', None]:
        """get the parent to this dialgo turn

        Returns:
            DIalogTurn: the parent dialog turn
        """
        return self._parent

    @property
    def n_children(self) -> int:
        """
        Returns:
            int: The number of children
        """
        return len(self._children)
    
    @property
    def n_siblings(self) -> int:
        """The number of siblings for the turn

        Returns:
            int: 
        """
        if self._parent is None:
            return 1
        
        return self._parent.n_children


class TreeDialog(BaseDialog):
    """
    """

    def __init__(
        self, 
        renderer: typing.Callable[[typing.List[Msg]], str] | None=None
    ):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(renderer)
        self._leaf = None
        self._indices = []
        self._counts = []
        self._root: DialogTurn | None = None
        self._update()

    def ancestor(self, count: int):
        
        self._leaf = self._leaf.ancestor(count)
        self._update()

    def child(self, idx: int):
        """Descend 

        Args:
            idx (int): The index of the child
        """
        self._leaf = self._leaf.child(idx)
        self._update()

    def sibling(self, idx: int):
        """Update the tree to use a sibling

        Args:
            idx (int): The index of the sibling to use
        """
        self._leaf = self._leaf.sibling(idx)
        self._update()

    @property
    def indices(self):
        return [*self._indices]
    
    @property
    def counts(self):
        return [*self._counts]

    def _update_indices(self):

        if self._leaf is None:
            self._indices = []
            return
        
        indices = []
        turn = self._leaf
        for ancestor in self._leaf.ancestors:
            idx = ancestor._children.index(turn)
            indices.append(idx)
            turn = ancestor
        indices.append(0)
        self._indices = list(reversed(indices))

    def _update_counts(self):

        if self._leaf is None:
            self._counts = []
            return
        result = []
        for ancestor in self._leaf.ancestors:
            result.append(
                ancestor.n_children
            )
        result.append(1)

        self._counts = list(reversed(result))

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        if len(self) == 0:
            return
        ancestors = list(self._leaf.ancestors)
        for turn in reversed(ancestors):
            yield turn.message
        yield self._leaf.message

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        return list(self)[idx]
    
    def _turn_list(self):

        if len(self) == 0:
            return []
        turns = []
        ancestors = list(self._leaf.ancestors)
        for turn in reversed(ancestors):
            turns.append(turn)
        turns.append(self._leaf)
        return turns

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        self._turn_list()[idx].message = message

    def clone(self) -> 'TreeDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        if self._root is None:
            return TreeDialog()

        # Perform a BFS to clone the tree
        # root_clone = DialogTurn(self._root._message)

        from collections import deque
        # turns = deque()
        counts = deque()
        dialog = TreeDialog()
        
        dialog.append(self._root.message)
        turn = self._root
        count = 0
        counts.append(0)
        leaf = None
        if turn is self._leaf:
            leaf = dialog._leaf
        while True:
            if turn.n_children > count:
                turn = turn.child(count)
                dialog.append(turn.message)
                if turn is self._leaf:
                    leaf = dialog._leaf
                counts.append(count)
                count = 0
            else:
                turn = turn.parent
                if turn is None:
                    break
                count = counts.pop()
                dialog.ancestor(1)
                count += 1

        dialog._leaf = leaf
        dialog._update()
        return dialog

    def pop(
        self, idx: int, get_msg: bool=False
    ) -> 'TreeDialog':
        """Remove a value from the dialog
        Cannot remove the root node.
        Args:
            index (int): The index to pop
        """
        if idx == 0:
            raise ValueError(
                'Cannot remove root node in tree dialog.'
            )
        ancestors = self._turn_list()
        target = ancestors[idx]
        
        parent = target._parent
        for turn in target._children:
            turn._parent = parent
            if parent is not None:
                parent._children.append(turn)
        
        if target._parent is not None:
            target._parent._children.remove(target)
        
        target._parent = None
        target._children = []
        if self._leaf is target:
            if turn.parent is not None:
                self._leaf = turn
            elif len(turn._children) > 0:
                self._leaf = turn._children[0]
            else:
                self._leaf = None
        self._update()
        if get_msg:
            return self, target._message
        return self

    def _bfs_find(self, root: DialogTurn, message: Msg) -> typing.Optional[DialogTurn]:
        """Perform a breadth-first search to find a message in the tree.

        Args:
            root (DialogTurn): The root of the tree to search.
            message (Msg): The message to find.

        Returns:
            typing.Optional[DialogTurn]: The DialogTurn containing the message, or None if not found.
        """
        queue = [root]
        while queue:
            current = queue.pop(0)
            if current.message == message:
                return current
            queue.extend(current._children)
        return None

    def remove(self, message: Msg):
        """Remove a message from the dialog

        If turn is the current "leaf" then
        it will set the turn to the parent if that is not None
        Or the first child or None

        Args:
            message (Msg): The message to remove
        """
        turn = self._root.find_val(message)
        if turn is self._root:
            raise ValueError(
                'Cannot remove root node in tree dialog.'
            )
        for child in turn._children:
            child.parent = turn.parent
            if turn.parent is not None:
                turn.parent.append(child)
        if turn.parent is not None:
            turn.parent._children.remove(turn)
        if self._leaf is turn:
            if turn.parent is not None:
                self._leaf = turn
            elif len(turn._children) > 0:
                self._leaf = turn._children[0]
            else:
                self._leaf = None
        turn.parent = None
        turn._children.clear()
        self._update()
        return self

    def extend(self, dialog: typing.Iterable[Msg]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        for msg in dialog:
            if self._leaf is None:
                self._leaf = DialogTurn(msg)
                self._root = self._leaf
            else:
                self._leaf = self._leaf.append(msg)
        
        self._update()

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self._counts)
        
    def append(self, message: Msg) -> Self:
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        
        if self._leaf is None:
            self._leaf = DialogTurn(message)
            self._root = self._leaf
        else:
            self._leaf = self._leaf.append(message)
        self._update()
        return self

    def insert(self, ind: int, message: Msg) -> Self:
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if self._leaf is None:
            if ind != 0:
                raise RuntimeError()
            self._leaf = DialogTurn(message)
            self._root = self._leaf
        else:
            ancestors = self._turn_list()
            turn = ancestors[ind]
            turn.prepend(message)
        self._update()
        return self
    
    def replace(self, idx: int, message: Msg) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            idx (typing.Optional[int], optional): The index to add. Defaults to None.
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if self._leaf is None:
            raise ValueError(f"There are no nodes.")
        self[idx] = message
        return self
    
    def _update(self):
        self._update_counts()
        self._update_indices()


def exclude_messages(
    dialog: BaseDialog, 
    val: typing.Union[typing.Any, typing.Set], 
    field='role'
) -> ListDialog:
    """Exclude messages from the dialog

    Args:
        dialog (BaseDialog): The dialog to filter
        val (typing.Union[typing.Any, typing.Set]): The value to exclude based on
        field (str, optional): The field to exclude basd on. Defaults to 'role'.

    Returns:
        ListDialog: The resulting dialog
    """
    if not isinstance(val, typing.Set):
        val = {val}

    return ListDialog(
        [msg for msg in dialog if msg[field] not in val]
    )

            
def include_messages(
    dialog: BaseDialog, 
    val: typing.Union[typing.Any, typing.Set], 
    field='role'
) -> ListDialog:
    """Include messages in the resulting dialog

    Args:
        dialog (BaseDialog): The dialog to filter
        val (typing.Union[typing.Any, typing.Set]): The value to exclude based on
        field (str, optional): The field to exclude basd on. Defaults to 'role'.

    Returns:
        ListDialog: The resulting dialog
    """
    if not isinstance(val, typing.Set):
        val = {val}

    return ListDialog(
        [msg for msg in dialog if msg[field] in val]
    )


def to_dialog(prompt: typing.Union[BaseDialog, Msg]) -> BaseDialog:
    """Convert a prompt to a dialog

    Args:
        prompt (typing.Union[Dialog, Msg]): The prompt to convert
    """
    if isinstance(prompt, Msg):
        prompt = ListDialog([prompt])

    return prompt


def to_list_input(msg: typing.List | typing.Tuple | BaseDialog | Msg) -> typing.List:

    if not isinstance(msg, typing.List) and not isinstance(msg, typing.Tuple):
        return msg.to_list_input()
    return msg


def exclude_role(
    messages: typing.Iterable[Msg], 
    *role: str
) -> typing.List[Msg]:
    """
    Filter messages by excluding specified roles.
    This function takes an iterable of messages and one or more role strings, returning
    a new list containing only messages whose roles are not in the specified roles to exclude.
    Args:
        messages (typing.Iterable[Msg]): An iterable of message objects
        *role (str): Variable number of role strings to exclude
    Returns:
        typing.List[Msg]: A list of messages excluding those with specified roles
    Example:
        >>> messages = [Msg(role="user", content="hi"), Msg(role="system", content="hello")]
        >>> exclude_role(messages, "system")
        [Msg(role="user", content="hi")]
    """
    exclude = set(role)
    return [message for message in messages
        if message.role not in exclude]


def include_role(
    messages: typing.Iterable[Msg],
    *role: str
) -> typing.List[Msg]:
    """Filter the iterable of messages by a particular role

    Args:
        messages (typing.Iterable[Msg]): 

    Returns:
        typing.List[Msg]: 
    """
    include = set(role)
    return [message for message in messages
        if message.role in include]


class FieldRenderer(object):

    def __init__(self, field: str='content'):
        """Renderer to render a specific field in the message

        Args:
            field (str, optional): The field name. Defaults to 'content'.
        """
        self.field = field

    def __call__(self, msg: Msg | BaseDialog) -> str:
        """Render a message

        Args:
            msg (Msg): The message to render

        Returns:
            str: The result
        """
        messages = to_list_input(msg)
        return '\n'.join(
            f'{msg['role']}: {msg[self.field]}'
            for msg in messages
        )
