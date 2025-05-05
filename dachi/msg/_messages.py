# 1st party
import typing
from abc import abstractmethod
from typing import Self

# 3rd party
import pydantic
from ..proc import Module

# local
from ..base._core import Renderable
from ._render import render


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


class BaseDialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """

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
    _messages: typing.List[Msg] = pydantic.PrivateAttr(default_factory=list)

    def list_messages(self) -> typing.List[Msg]:
        return self._messages

    def __init__(
        self, messages: typing.Iterable[Msg]=None
    ):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__()
        self._messages = messages or []

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



class MsgRenderer(Module):

    @abstractmethod
    def forward(self, msg: Msg) -> str:
        pass


class FieldRenderer(MsgRenderer):

    def __init__(self, field: str='content', meta: bool=False):

        self.field = field
        self.meta = meta

    def forward(self, msg: Msg) -> str:
        
        if self.meta:
            return render(msg.m[self.field])
        return render(msg[self.field])


class TreeDialog(BaseDialog):

    pass


class DialogTurn(BaseDialog):
    """A Dialog that uses a list data structure.
    """
    _message: Msg = pydantic.PrivateAttr()
    _parent: 'DialogTurn' = pydantic.PrivateAttr(default=None)
    _children: typing.List['DialogTurn'] = pydantic.PrivateAttr(default_factory=list)

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
    
    @property
    def parent(self) -> typing.Union['DialogTurn', None]:
        """get the parent to this dialgo turn

        Returns:
            DIalogTurn: the parent dialog turn
        """
        return self._parent
    
    @property
    def children(self) -> typing.Iterator['DialogTurn']:
        """get the children to the dialog turn

        Returns:
            typing.List['DialogTurn']: The children
        """
        return iter(self._children)

    def list_messages(self) -> typing.List[Msg]:
        """List all messages up to this dialog turn

        Returns:
            typing.List[Msg]: the list of messages
        """
        return list(
            msg for msg in self
        )
    
    def list_turns(self) -> typing.List['DialogTurn']:
        """List all turns up to this dialog turn

        Returns:
            typing.List[Msg]: the list of messages
        """
        turns = []
        turn = self
        while True:
            turns.append(turn)
            turn = turn.parent
            if turn is None:
                break
        return list(reversed(turns))

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog
        up to the current position

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        node = self
        nodes = []
        while True:
            nodes.append(node)
            if node.parent is None:
                break
            node = node.parent
        for node in reversed(nodes):
            yield node.message

    def child(self, idx: int) -> Msg:

        return self._children[idx]

    def __add__(self, other: 'BaseDialog') -> 'DialogTurn':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        cur = self
        for node in other:
            cur = cur.append(node)
        return cur

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Msg: The message in the dialog
        """
        return self.list_messages()[idx]

    def __setitem__(self, idx, message) -> Self:
        """Set idx with a message

        Args:
            idx: The index to set
            message: The message to set

        Returns:
            Dialog: The updated dialog
        """
        turn = self.index(idx)
        turn.message = message
        return self
    
    def index(self, idx: int) -> 'DialogTurn':

        # if idx < 0:
        #     node = self
        #     while idx < 0:
        #         node = node._parent
        #         if node is None:
        #             raise IndexError(
        #                 f'Index {idx} is out of bounds for the Dialog'
        #             )
        #         idx += 1
        #     return node

        turns = self.list_turns()
        return turns[idx]
    
    def find_in_children(self, message: Msg) -> 'DialogTurn':

        if self._message == message:
            return self

        result = None
        for child in self._children:
            result = result or child.find_in_children(message)

        return result
    
    @property
    def message(self) -> Msg:
        return self._message
    
    def find(self, message: Msg) -> 'DialogTurn':

        return self.root().find_in_children(message)

    def pop(self, index: int, get_msg: bool=False) -> typing.Union['ListDialog', Msg]:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        turn = self.index(index)
        parent = turn._parent
        turn._parent = None
        for child in turn._children:
            child._parent = parent
        turn._children = []
        if parent is not None:
            parent._children = turn._children
        if get_msg:
            return self, turn.message
        return self

    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        turn = self.find(message)
        if turn is None:
            raise KeyError(
                f'There is no message {message} in'
            )

    def extend(self, dialog: typing.Union['BaseDialog', typing.List[Msg]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        if isinstance(dialog, Msg):
            dialog = [dialog]
        node = self
        for turn in dialog:
            node = node.append(turn)
        return node

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self.list_messages())
        
    def clone(self) -> 'DialogTurn':
        """Clones the entire tree including the root

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        root = self.root()
        return root.clone_sub()
    
    def clone_sub(self) -> 'DialogTurn':
        """clone the tree lying below this tree

        Returns:
            DialogTurn: The sub tree
        """
        cloned_self = DialogTurn(
            self.message
        )
        for child in self.children:

            cloned_child = child.clone_sub()
            cloned_self._children.append(cloned_child)
            cloned_child._parent = cloned_self

        return cloned_self

    def depth_iter(self) -> typing.Iterator['DialogTurn']:

        for child in self.children:
            yield child
            for turn in child.breadth_iter():
                yield child

    def append(self, message: Msg) -> 'DialogTurn':
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        turn = DialogTurn(
            message=message, 
            parent=self
        )
        self._children.append(turn)
        return turn

    def insert(self, idx: int, message: Msg) -> Self:
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        turn = self.index(idx)
        inserted = DialogTurn(
            message=message,
            parent=turn._parent,
            children=[turn]
        )
        if turn._parent is not None:
            turn._parent._children.remove(turn)
            turn._parent._children.append(inserted)
        turn._parent = inserted
        return inserted

    def replace(self, message: Msg, idx: int) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            idx (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        turn = self.index(idx)
        inserted = DialogTurn(
            message=message,
            parent=turn._parent,
            children=turn._children
        )
        for child in inserted._children:
            child._parent = inserted
        if turn._parent is not None:
            turn._parent._children.remove(turn)
            turn._parent._children.append(inserted)

        turn._children = []
        turn._parent = None
        return inserted

    def __add__(self, other: BaseDialog | Msg) -> 'ListDialog':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        if isinstance(other, typing.List):
            return ListDialog(
                self.list_messages() + other
            )
        return ListDialog(
            self.list_messages() + other.aslist()
        )


class MsgRenderer(Module):

    @abstractmethod
    def forward(self, msg: Msg | BaseDialog) -> str:
        pass


class FieldRenderer(MsgRenderer):

    def __init__(self, field: str='content'):
        """Renderer to render a specific field in the message

        Args:
            field (str, optional): The field name. Defaults to 'content'.
        """
        self.field = field

    def forward(self, msg: Msg | BaseDialog) -> str:
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


def exclude_messages(dialog: BaseDialog, val: typing.Union[typing.Any, typing.Set], field='role') -> ListDialog:

    if not isinstance(val, typing.Set):
        val = {val}

    return ListDialog(
        [msg for msg in dialog.list_messages() if msg[field] not in val]
    )

            
def include_messages(dialog: BaseDialog, val: typing.Union[typing.Any, typing.Set], field='role') -> ListDialog:

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


def exclude_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
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


def include_role(messages: typing.Iterable[Msg], *role: str) -> typing.List[Msg]:
    """Filter the iterable of messages by a particular role

    Args:
        messages (typing.Iterable[Msg]): 

    Returns:
        typing.List[Msg]: 
    """
    include = set(role)
    return [message for message in messages
        if message.role in include]

