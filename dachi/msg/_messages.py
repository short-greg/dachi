# 1st party
import typing
from abc import abstractmethod
from typing import Self
import typing

# 3rd party
import pydantic

# local
from ..base._core import Renderable


class _Final:
    """A unique object to mark the end of a streaming response."""
    def __repr__(self):
        return "<Final Token>"

END_TOK = _Final()


class Msg(dict):
    """A Msg used for a dialog
    """
    def __init__(
        self, role: str, type_: str='data', 
        meta: typing.Dict=None, delta: typing.Dict=None, 
        _include_role: bool=True, **kwargs
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

    def to_list_input(self) -> typing.List[typing.Dict]:
        """Convert to an input appropriate for a list

        Returns:
            typing.List[typing.Dict]: The message converted to a list
        """
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
    

class BaseDialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """

    msg_renderer: typing.Optional[typing.Callable[[Msg], str]] = None

    @abstractmethod
    def messages(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog"""
        pass

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self.messages():
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

    def add(self, role: str='user', type_: str='data', delta: typing.Dict=None, meta: typing.Dict=None, _ind: typing.Optional[int]=None, _replace: bool=False, _inplace: bool=False, **kwargs) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            type_ (str, optional): The type of message. Defaults to 'data'.
            delta (typing.Dict, optional): The change in the message. Defaults to None.
            meta (typing.Dict, optional): Any other information that is not a part of the message. Defaults to None.
            ind (typing.Optional[int], optional): The index to add it to. Defaults to None.
            replace (bool, optional): Whether to replace the value at the index or offset it. Defaults to False.

        Returns:
            Dialog: The dialog with the message appended
        """
        msg = Msg(
            role=role, type_=type_, meta=meta, 
            delta=delta, **kwargs
        )
        return self.insert(message=msg, ind=_ind, replace=_replace, inplace=_inplace)
    
    def append(
        self, message: Msg, _replace: bool=False, 
        _inplace: bool=False) -> 'BaseDialog':
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        return self.insert(message=message, ind=None, replace=_replace, inplace=_inplace)

    @abstractmethod
    def insert(self, message: Msg, ind: typing.Optional[int]=None, replace: bool=False, inplace: bool=False) -> 'BaseDialog':
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
            message.render() if self.msg_renderer is None else self.msg_renderer(message)
            for message in self.messages()
        )
    
    def to_input(self) -> typing.List[typing.Dict]:
        """Convert the dialog to an input to pass into an API

        Returns:
            typing.List[typing.Dict]: A list of inputs
        """
        return [msg.to_input() for msg in self]

    def to_list_input(self) -> typing.List[typing.Dict]:

        return self.to_input()

    def aslist(self) -> typing.List['Msg']:
        """Retrieve the message list

        Returns:
            typing.List[Msg]: the messages in the dialog
        """
        return list(self.messages())
    
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
        return inp.to_input()
    
    return {msg.to_input() for msg in inp}


class ListDialog(BaseDialog):
    """A Dialog that uses a list data structure.
    """
    _messages: typing.List[Msg] = pydantic.PrivateAttr(default_factory=list)

    def messages(self):
        return self._messages

    def __init__(
        self, messages: typing.Iterable[Msg]=None,
        msg_renderer: typing.Callable[[Msg], str]=None
    ):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(msg_renderer=msg_renderer)
        self._messages = messages or []

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Msg]]: Each message in the dialog
        """
        for message in self._messages:
            yield message

    def __add__(self, other: 'BaseDialog') -> 'BaseDialog':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        return BaseDialog(
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
        self._messages[idx] = message
        return self

    def pop(self, index: int) -> Msg:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        return self._messages.pop(index)

    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (Msg): The message to remove
        """
        self._messages.remove(message)

    def insert(self, message: Msg, ind: typing.Optional[int]=None, replace: bool=False, inplace: bool=False):
        """Add a message to the dialog

        Args:
            message (Msg): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if ind is not None and ind < 0:
            ind = max(len(self) + ind, 0)

        messages = [*self._messages]
        if ind is None or ind == len(self):
            messages.append(message)
            # if not replace or ind == len(self) or len(self) == 0:
            #     messages.append(message)
            # else:
            #     messages[-1] = message
        elif ind > len(self._messages):
            raise ValueError(
                f'The index {ind} is out of bounds '
                f'for size {len(self)}')
        elif replace:
            messages[ind] = message
        else:
            messages.insert(ind, message)
        if inplace:
            self._messages = messages
            return self
        return ListDialog(
            messages
        )

    def extend(self, dialog: typing.Union['BaseDialog', typing.List[Msg]], inplace: bool=False):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        if isinstance(dialog, BaseDialog):
            dialog = dialog.aslist()
        
        messages = [
            self._messages + dialog
        ]
        if inplace:
            self._messages = messages
            return self
        return ListDialog(messages)

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self._messages)
        
    def clone(self) -> typing.Self:
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        return ListDialog(
            messages=[message for message in self._messages]
        )


class RenderMsgField:

    def __init__(self, field: str='content'):
        """Renderer to render a specific field in the message

        Args:
            field (str, optional): The field name. Defaults to 'content'.
        """
        self.field = field

    def __call__(self, msg: Msg) -> str:
        """Render a message

        Args:
            msg (Msg): The message to render

        Returns:
            str: The result
        """
        return f'{msg['role']}: {msg[self.field]}'


def exclude_messages(dialog: BaseDialog, val: typing.Union[typing.Any, typing.Set], field='role') -> ListDialog:

    if not isinstance(val, typing.Set):
        val = {val}

    return ListDialog(
        [msg for msg in dialog.messages() if msg[field] not in val], msg_renderer=dialog.msg_renderer
    )

            
def include_messages(dialog: BaseDialog, val: typing.Union[typing.Any, typing.Set], field='role') -> ListDialog:

    if not isinstance(val, typing.Set):
        val = {val}

    return ListDialog(
        [msg for msg in dialog.messages() if msg[field] in val],
        msg_renderer=dialog.msg_renderer
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
