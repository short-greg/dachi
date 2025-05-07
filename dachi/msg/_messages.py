# 1st party
import typing
from abc import abstractmethod
from typing import Self

# local
from ..base._core import Renderable

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
        self._follow_up = _follow_up if _follow_up is not None else []
        self._filtered = _filtered

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
    
    def __add__(self, other) -> 'ListDialog':

        if isinstance(other, Msg):
            return ListDialog([self, other])
        messages = list(other)
        return ListDialog(
            [self, *messages]
        )

    @property
    def follow_up(self) -> typing.List['Msg'] | None:
        """Get the follow up message

        Returns:
            typing.List[Msg]: The follow up message
        """
        return [*self._follow_up]
    
    @follow_up.setter
    def follow_up(self, messages: typing.List['Msg'] | None) -> Self:
        """Set the follow up message

        Args:
            message (typing.Union[Msg, typing.List[Msg]]): The follow up message
        """
        if self._follow_up is None:
            self._follow_up = []
        else:
            self._follow_up = messages
        return messages

    @property
    def filtered(self) -> bool:
        """Get the filtered status of the message

        Returns:
            bool: The filtered status
        """
        return self._filtered
    
    @filtered.setter
    def filtered(self, val: bool) -> Self:
        """Set the filtered status of the message

        Args:
            val (bool): The filtered status
        """
        self._filtered = val
        return self


class StreamMsg(Msg):
    """A message that is streamed
    """

    def __init__(
        self, role: str, type_: str='data', 
        meta: typing.Dict=None, delta: typing.Dict=None, 
        _include_role: bool=True, is_last: bool=False, **kwargs
    ):
        """Create a Stream Msg

        Args:
            type_ (str, optional): The type of message. Defaults to 'data'.
            meta (typing.Dict, optional): Any additional information not related to the message specifically. Defaults to None.
            delta (typing.Dict, optional): The change in the message. Defaults to None.
        """
        super().__init__(
            role=role, type_=type_, meta=meta or {}, delta=delta or {},_include_role=_include_role, **kwargs
        )
        self.is_last = is_last


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

    @abstractmethod
    def list_messages(self) -> typing.List[Msg]:
        """Iterate over each message in the dialog"""
        pass

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self.list_messages():
            yield message

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
    def extend(self, dialog: typing.Union['BaseDialog', typing.Iterable[Msg]], _inplace: bool=False) -> 'BaseDialog':
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
            return self._renderer(self.list_messages())
        return '\n'.join(
            message.render()
            for message in self.list_messages()
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
        return list(self.list_messages())
    
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

    @abstractmethod
    def __add__(self, other) -> Self:
        pass

    
# How to handle a tree
# dialog tree

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

    def list_messages(self) -> typing.List[Msg]:
        return self._messages

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self._messages:
            yield message

    def __add__(self, other: BaseDialog | Msg) -> 'ListDialog':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        if isinstance(other, typing.List):
            return ListDialog(
                self._messages + other
            )
        return ListDialog(
            self._messages + other.aslist()
        )

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

    def extend(self, dialog: typing.Union['BaseDialog', typing.List[Msg]]):
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
        self._message = message
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
        
    @property
    def parent(self) -> typing.Union['DialogTurn', None]:
        """get the parent to this dialgo turn

        Returns:
            DIalogTurn: the parent dialog turn
        """
        return self._parent
    
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

    def ascend(self, count: int):
        i = 0
        turn = self
        while True:
            if i == count:
                return turn._message
            if turn._parent is None:
                raise RuntimeError(
                    "There are only "
                    f"{i} ancestors yet"
                    f"you passed {count}"
                )
            turn = turn._parent
            i += 1


class TreeDialog(BaseDialog):

    # _root: DialogTurn = pydantic.PrivateAttr(default=None)
    # _leaf: DialogTurn = pydantic.PrivateAttr(default=None)
    # _indices: typing.List = pydantic.PrivateAttr(default=list)
    # _counts: typing.List = pydantic.PrivateAttr(default=list)

    def __init__(
        self, 
        leaf: DialogTurn=None,
        renderer: typing.Callable[[typing.List[Msg]], str] | None=None
    ):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(renderer)
        self._leaf = leaf
        self._indices = []
        self._counts = []
        if leaf is None:
            self._root = None
        else:
            self._root = leaf.root()
        self._update()

    def ascend(self, count: int):
        
        self._leaf = self._leaf.ascend(count)
        self._update()

    def sibling(self, idx: int):
    
        if self._leaf.parent is None:
            raise RuntimeError(

            )
        self._leaf = self._leaf.parent._children[idx]
        self._update()

    def _indices(self):

        indices = []
        turn = self._leaf
        for ancestor in self._leaf.ancestors:
            idx = ancestor._children.index(turn)
            indices.append(idx)
        return indices

    @property
    def indices(self):
        return [*self._indices]
    
    @property
    def counts(self):
        return [*self._counts]

    def _counts(self) -> typing.List[int]:

        return list(
            reversed(
                len(ancestor._children) 
                for ancestor in self._leaf.ancestors
            )
        )

    def list_messages(self) -> typing.List[Msg]:
        """Iterate over each message in the dialog
        up to the current position

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        return list(self)

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in reversed(self._leaf.ancestors):
            yield message

    def __add__(self, other: BaseDialog | Msg) -> 'ListDialog':
        """Add the leaf to 

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        clone = self.clone()
        clone.extend(other)
        return clone

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        ancestors = list(self._leaf.ancestors)
        return ancestors[idx]._message

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        ancestors = list(self._leaf.ancestors())
        ancestors[idx]._message = message
        return message

    def clone(self) -> 'ListDialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        if self._root is None:
            return TreeDialog()

        # Perform a BFS to clone the tree
        root_clone = DialogTurn(self._root._message)
        queue = [(self._root, root_clone)]

        while queue:
            original_node, cloned_node = queue.pop(0)

            for child in original_node._children:
                child_clone = DialogTurn(child._message)
                cloned_node._children.append(child_clone)
                child_clone._parent = cloned_node
                queue.append((child, child_clone))

        return TreeDialog(leaf=root_clone.leaf())

    def pop(self, idx: int, get_msg: bool=False) -> typing.Union['ListDialog', Msg]:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        ancestors = list(self._leaf.ancestors)
        target = ancestors[idx]
        
        for turn in target._children:
            turn._parent = target._parent
            target._children.append(turn)
        
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
            if current._message == message:
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
        turn = self._bfs_find(self._root, message)
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

    def extend(self, dialog: typing.Union['BaseDialog', typing.List[Msg]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        for msg in dialog:
            turn = self._leaf.append(msg)
            self._leaf = turn
        
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
        ancestors = list(self._leaf.ancestors())
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
        ancestors = list(self._leaf.ancestors())
        ancestors[idx]._message = message
        self._update()
        return self
    
    def _update(self):
        self._counts()
        self._indices()


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
        [msg for msg in dialog.list_messages() if msg[field] not in val]
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
        [msg for msg in dialog.list_messages() if msg[field] in val]
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
