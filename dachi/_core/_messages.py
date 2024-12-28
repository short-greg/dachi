# 1st party
import typing
from abc import ABC, abstractmethod
from typing import Self
import typing

from ._core import (
    Reader, 
)
from ._core import Renderable

# 3rd party
import pydantic


class Msg(dict):

    def __init__(self, type_: str='data', meta: typing.Dict=None, delta: typing.Dict=None, **kwargs):
        super().__init__(
            type_=type_, meta=meta, delta=delta, **kwargs
        )

    @property
    def type(self) -> str:
        return self['type_']
    
    def to_dict(self) -> typing.Dict:

        d = {**self}
        del d['type_']
        if 'meta' in d:
            del d['meta']
        if 'delta' in d:
            del d['delta']
        return d


class Dialog(pydantic.BaseModel, Renderable):
    """A Dialog stores the interactions between the system/user and the assistant
    (i.e. the prompts and the responses)
    """

    @abstractmethod
    def messages(self) -> typing.Iterator[Msg]:
        pass

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Message]]: Each message in the dialog
        """
        for message in self.messages():
            yield message

    @abstractmethod
    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Message: The message in the dialog
        """
        pass
        # return self.messages[idx]

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
        # self.messages[idx] = message
        # return self

    @abstractmethod
    def pop(self, index: int) -> Msg:
        """Remove a value from the dialog

        Args:
            index (int): The index to pop
        """
        pass
        # return self.messages.pop(index)

    @abstractmethod
    def remove(self, message: Msg):
        """Remove a message from the dialog

        Args:
            message (ChatMessage): The message to remove
        """
        pass

    @abstractmethod
    def add(self, message: Msg, ind: typing.Optional[int]=None, replace: bool=False) -> 'Dialog':
        """Add a message to the dialog

        Args:
            message (Message): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        pass
        # if ind is not None and ind < 0:
        #     ind = max(len(self) + ind, 0)

        # if ind is None or ind == len(self):
        #     if not replace or ind == len(self):
        #         self.messages.append(message)
        #     else:
        #         self.messages[-1] = message
        # elif ind > len(self.messages):
        #     raise ValueError(
        #         f'The index {ind} is out of bounds '
        #         f'for size {len(self.messages)}')
        # elif replace:
        #     self.messages[ind] = message
        # else:
        #     self.messages.insert(ind, message)

    @abstractmethod
    def extend(self, dialog: typing.Union['Dialog', typing.Iterable[Msg]]) -> 'Dialog':
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Message]]): _description_
        """
        pass

    @abstractmethod
    def reader(self) -> 'Reader':
        """Get the "Reader" for the dialog. By default will use the last one
        that is available.

        Returns:
            Reader: The reader to retrieve
        """
        pass

    def render(self) -> str:
        """Render the dialog as a series of turns 
        <role>: <text>

        Returns:
            str: The dialog
        """
        return '\n'.join(
            message.render() for message in self.messages()
        )

    def aslist(self) -> typing.List['Msg']:
        """Retrieve the message list

        Returns:
            typing.List[Message]: the messages in the dialog
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
        
    def clone(self) -> 'Dialog':
        """Clones the dialog

        Returns:
            Dialog: A dialog cloned with shallow copying of the messages
        """
        pass


class ListDialog(Dialog):

    _messages: typing.List[Msg] = pydantic.PrivateAttr(default_factory=list)

    def __init__(self, messages: typing.Iterable[Msg]=None):
        """Create a dialog

        Args:
            messages: The messages
        """
        super().__init__(_messages=messages)

    def __iter__(self) -> typing.Iterator[Msg]:
        """Iterate over each message in the dialog

        Yields:
            Iterator[typing.Iterator[Message]]: Each message in the dialog
        """
        for message in self.messages():
            yield message

    def __add__(self, other: 'Dialog') -> 'Dialog':
        """Concatenate two dialogs together

        Args:
            other (Dialog): The other dialog to concatenate

        Returns:
            Dialog: The concatenated dialog
        """
        return Dialog(
            self._messages + other.aslist()
        )

    def __getitem__(self, idx) -> Msg:
        """Retrieve a value from the dialog

        Args:
            idx : The index to add at

        Returns:
            Message: The message in the dialog
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
            message (ChatMessage): The message to remove
        """
        self._messages.remove(message)

    def add(self, message: Msg, ind: typing.Optional[int]=None, replace: bool=False):
        """Add a message to the dialog

        Args:
            message (Message): The message to add
            ind (typing.Optional[int], optional): The index to add. Defaults to None.
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if ind is not None and ind < 0:
            ind = max(len(self) + ind, 0)

        messages = [*self._messages]
        if ind is None or ind == len(self):
            if not replace or ind == len(self):
                messages.append(message)
            else:
                messages[-1] = message
        elif ind > len(self._messages):
            raise ValueError(
                f'The index {ind} is out of bounds '
                f'for size {len(self)}')
        elif replace:
            messages[ind] = message
        else:
            messages.insert(ind, message)
        return ListDialog(
            messages
        )

    def extend(self, dialog: typing.Union['Dialog', typing.List[Msg]]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Message]]): _description_
        """
        if isinstance(dialog, Dialog):
            dialog = dialog.aslist()
        
        messages = [
            self._messages + dialog
        ]
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
