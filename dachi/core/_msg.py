# 1st party
from __future__ import annotations
import typing
import typing as t
from abc import abstractmethod
from typing import Self
import base64
from datetime import datetime, timezone

# 3rd party
from pydantic import BaseModel, PrivateAttr
import pydantic

# local
from . import Renderable
from ._tool import ToolUse
from ._base import BaseModule

try:
    # pydantic v2 preferred
    from pydantic import BaseModel, Field
    _PD_V2 = True
except Exception:  # pragma: no cover
    # pydantic v1 fallback
    from pydantic import BaseModel, Field  # type: ignore
    _PD_V2 = False



class _Final:
    """A unique object to mark the end of a streaming response."""
    def __repr__(self):
        return "<Final Token>"

END_TOK = _Final()
NULL_TOK = object()



class RespDelta(pydantic.BaseModel):
    """Delta information for streaming responses."""
    
    text: str | None = pydantic.Field(
        default=None, description="Incremental text content for streaming responses"
    )
    tool: str | None = pydantic.Field(
        default=None, description="Partial tool call arguments (JSON string) for streaming responses"
    )
    thinking: str | None = pydantic.Field(
        default=None, description="Incremental reasoning/thinking content for streaming responses"
    )
    citations: typing.List[typing.Dict] | None = pydantic.Field(
        default=None, description="Incremental citation information for streaming responses"
    )
    finish_reason: str | None = pydantic.Field(
        default=None, description="Reason generation stopped (e.g., 'stop', 'length', 'tool_calls')"
    )
    usage: typing.Dict[str, int] | None = pydantic.Field(
        default=None, description="Per-chunk token usage statistics (when stream_options.include_usage=true)"
    )
    proc_store: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=dict, description="Storage for RespProc processing data"
    )


class Resp(pydantic.BaseModel):
    """A response from the LLM or an API.
    
    This class represents a unified response format that contains both the message 
    content and metadata from API calls. It supports streaming responses through 
    delta accumulation and provides access to raw API responses.
    """

    # Core content
    msg: Msg | None = pydantic.Field(
        default=None, description="The main message content from the LLM response"
    )
    text: str | None = pydantic.Field(
        default=None, description="Generated text content from the LLM"
    )
    tool: typing.List[typing.Dict] | None = pydantic.Field(
        default=None, description="Complete tool call objects for non-streaming responses"
    )
    thinking: str | None = pydantic.Field(
        default=None, description="Model's reasoning or thought process content"
    )
    logprobs: typing.Dict | None = pydantic.Field(
        default=None, description="Log probabilities for generated tokens"
    )
    citations: typing.List[typing.Dict] | None = pydantic.Field(
        default=None, description="Source citations for generated content"
    )
    finish_reason: str | None = pydantic.Field(
        default=None, description="Reason generation stopped (e.g., 'stop', 'length', 'tool_calls')"
    )
    
    # Legacy/compatibility
    val: typing.Any = pydantic.Field(
        default=None, description="Legacy field for processed API outputs"
    )
    follow_up: typing.List[Msg] | None = pydantic.Field(
        default_factory=list, description="Follow-up messages (e.g., tool execution results)"
    )
    
    # Metadata
    response_id: str | None = pydantic.Field(
        default=None, description="Unique identifier for this response"
    )
    model: str | None = pydantic.Field(
        default=None, description="Model name/version that generated this response"
    )
    usage: typing.Dict[str, int] = pydantic.Field(
        default_factory=dict, description="Token usage statistics (prompt_tokens, completion_tokens, etc.)"
    )
    choices: typing.List[typing.Dict[str, typing.Any]] | None = pydantic.Field(
        default=None, description="Choice-level metadata for multiple completions (index, finish_reason, etc.)"
    )
    
    # Streaming support
    delta: RespDelta = pydantic.Field(
        default_factory=RespDelta, description="Delta information for streaming responses"
    )
    
    # Provider-specific metadata
    meta: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=dict, description="Provider-specific metadata and additional fields"
    )
    
    # Private attributes
    _data: typing.Dict = pydantic.PrivateAttr(
        default_factory=dict
    )
    _delta: typing.Dict = pydantic.PrivateAttr(
        default_factory=dict
    )
    _out: typing.Dict = pydantic.PrivateAttr(
        default_factory=dict
    )

    @property
    def data(self) -> typing.Any:
        """ Get the raw data from the response.

        Returns:
            typing.Any: The raw data from the response.
        """
        return self._data
        
    # Note: delta is now a proper RespDelta field, no property needed
    
    @property
    def out(self) -> typing.Dict:
        """Get the output values from the response.

        Returns:
            typing.Dict: The output values from the response.
        """
        return self._out
    
    def spawn(self, msg: Msg, data: typing.Dict=None, follow_up: bool=None) -> 'Resp':
        """Spawn a new response with the same delta and out but a different message and data

        Args:
            msg (Msg): The message to spawn

        Returns:
            Resp: A new response with the same data but a different message
        """
        data = data if data is not None else {}
        resp = Resp(
            msg=msg,
            val=self.val,
            follow_up=follow_up,
        )
        resp.data.update(data)
        resp.out.update(self._out)
        # Copy delta information including proc_store for streaming
        if self.delta:
            resp.delta.text = self.delta.text
            resp.delta.tool = self.delta.tool  
            resp.delta.thinking = self.delta.thinking
            resp.delta.citations = self.delta.citations
            resp.delta.finish_reason = self.delta.finish_reason
            resp.delta.proc_store.update(self.delta.proc_store)
        return resp


class Attachment(BaseModel):
    """
    Declarative reference to a non-text asset.

    Attributes:
        kind: 'image' | 'audio' | 'video' | 'file' | 'data'
        ref: Stable handle (file id, URL, object key). Adapters resolve/upload.
        mime: Optional MIME type, e.g. 'image/png'.
        name: Human-friendly label/filename.
        purpose: Hint for adapters, e.g. 'vision_input', 'context_doc'.
        info: Small, inspectable metadata. Large payloads should be external.
    """
    kind: t.Literal['image', 'audio', 'video', 'file', 'data']
    data: str = Field(
        description="The raw data in bytes or the url for the data."
    )
    mime: t.Optional[str] = Field(default=None, description="MIME type of the attachment, e.g. 'image/png'.")
    name: t.Optional[str] = Field(default=None, description="Human-friendly label/filename.")
    purpose: t.Optional[str] = Field(default=None, description="Hint for adapters, e.g. 'vision_input', 'context_doc'.")
    info: t.Dict[str, t.Any] = Field(default_factory=dict)


def to_b64(filepath) -> str:
    """ Convert a file to a base64 string.
    """
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class Msg(BaseModel):
    """
    Text-first message with attachments and tool events.

    Notes:
        - `content` may be plain text or an ordered mapping of channels (e.g., {"final": "...", "thinking": "..."}).
        - Attachments are declarative refs; adapters handle upload/encoding.
        - Tool events interleave calls, deltas, and results (including computer-use).
        - Provider/runtime metadata should live on Resp, not here.
    """
    role: str
    text: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = Field(default=None, description="Text content as a single element or a sequence of label text values.")
    alias: t.Optional[str] = None

    attachments: t.List[Attachment] = Field(default_factory=list, description="List of attachments included in the message.")
    tool_calls: t.List[ToolUse] = Field(default_factory=list, description="List of tool calls with their results stored in ToolUse.result")

    tags: t.List[str] = Field(default_factory=list, description="List of tags associated with the message.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the message creation in UTC.")
    id: t.Optional[str] = Field(default=None, description="Unique identifier for the message.")
    prev_id: t.Optional[str] = Field(default=None, description="ID of the previous message in the conversation.")
    
    # Fields for compatibility with existing tests
    filtered: bool = Field(default=False, description="Whether this message should be filtered from outputs")
    meta: t.Dict[str, t.Any] = Field(default_factory=dict, description="Metadata dictionary for additional information")

    class Config:
        extra = "allow"

    def render(self) -> str:
        """Compact, human-readable line for logs."""
        name = self.alias or self.role
        if isinstance(self.text, str):
            text = self.text
        elif isinstance(self.text, dict):
            text = self.text.get("final") or self.text.get("text") or ""
        else:
            text = ""
        extras: t.List[str] = []
        if self.attachments:
            extras.append(f"attachments={len(self.attachments)}")
        if self.tool_calls:
            extras.append(f"tool_calls={len(self.tool_calls)}")
        suffix = f"  [{' | '.join(extras)}]" if extras else ""
        return f"{name}: {text}{suffix}"
    
    def apply(self, func):
        """Apply a function to this message and return the result."""
        return func(self)
    
    def output(self, key: str = "tool_out", default=None):
        """Get a value from the meta dictionary."""
        return self.meta.get(key, default)

    # def to_input(self) -> t.Dict[str, t.Any]:
    #     """
    #     Provider-agnostic dict for adapters to expand into provider-specific shapes.

    #     Returns:
    #         Dict[str, Any]: Minimal neutral representation.
    #     """
    #     base: t.Dict[str, t.Any] = {"role": self.role}
    #     if isinstance(self.content, (str, dict)):
    #         base["content"] = self.content
    #     if self.alias:
    #         base["alias"] = self.alias
    #     if self.attachments:
    #         base["attachments"] = [a.model_dump() if _PD_V2 else a.dict()]
    #     if self.tools:
    #         base["tools"] = [e.model_dump() if _PD_V2 else e.dict()]
    #     if self.tags:
    #         base["tags"] = list(self.tags)
    #     return base


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
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        return self.insert(
            message=message, ind=None
        )

    @abstractmethod
    def replace(
        self, message: Msg, ind: int
    ) -> 'BaseDialog':
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
    
    # def to_input(self) -> typing.List[typing.Dict]:
    #     """Convert the dialog to an input to pass into an API

    #     Returns:
    #         typing.List[typing.Dict]: A list of inputs
    #     """
    #     return [msg.to_input() for msg in self if msg.filtered is False]

    # def to_list_input(self) -> typing.List[typing.Dict]:

    #     return self.to_input()

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

    
# def to_input(
#     inp: typing.Union[typing.Iterable[Msg], Msg]
#     ) -> typing.Union[typing.List[Msg], Msg]:
#     """Convert a list of messages or a single message to an input

#     Args:
#         inp (typing.Union[typing.Iterable[Msg], Msg]): The inputs to convert

#     Returns:
#         typing.Union[typing.List[Msg], Msg]: The input
#     """
#     if isinstance(inp, Msg):
#         return inp.to_input() if inp.filtered is False else None
    
#     return {msg.to_input() for msg in inp if msg.filtered is False}


class ListDialog(BaseDialog):
    """A Dialog that uses a list data structure.
    """

    messages: typing.List[Msg] = pydantic.Field(default_factory=list)

    def __iter__(self) -> typing.Iterator[Msg]:
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

    def pop(self, index: int=-1, get_msg: bool=False) -> typing.Union['ListDialog', Msg]:
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

    def extend(self, dialog: typing.Iterable[Msg]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        if isinstance(dialog, BaseDialog):
            dialog = dialog.aslist()
        
        validated = True
        for msg in dialog:
            validated = validated and isinstance(msg, Msg)
            if not validated:
                raise ValueError(
                    "List dialog must only consist of messages."
                )
        self.messages.extend(dialog)
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
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if not isinstance(message, Msg):
            raise ValueError(
                "List dialog must only consist of messages."
            )
        self.messages.append(message)
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
        self.messages.insert(ind, message)
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
        self.messages[idx] = message
        return self


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


def to_list_input(
    msg: typing.List | typing.Tuple | BaseDialog | Msg
) -> typing.List:
    """Convert a message or a list of messages to an input
    Args:
        msg (typing.List | typing.Tuple | BaseDialog | Msg): The message or messages to convert
    Returns:
        typing.List: A list of inputs
    """
    
    if isinstance(msg, BaseDialog):
        return msg.to_input()
    elif isinstance(msg, Msg):
        return [msg.to_input()]
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


class DialogTurn(pydantic.BaseModel):
    """A single tuurn in the dialog.
    """

    message: Msg
    children: typing.Union[typing.List['DialogTurn']] = pydantic.Field(default_factory=list)
    _parent: typing.Union['DialogTurn', None] = pydantic.PrivateAttr(default=None)

    def model_post_init(self, __context):
        for child in self.children:
            child._parent = self
            child.model_post_init(__context)

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
        if len(self.children) == 0:
            return self
        return self.children[0].leaf()
    
    @property
    def ancestors(self) -> typing.Iterator['DialogTurn']:

        turn = self
        while True:
            if turn._parent is None:
                return
            turn = turn._parent
            yield turn
    
    def prepend(self, message: 'Msg') -> 'DialogTurn':
        """

        Args:
            turn (DialogTurn): 

        Returns:
            DialogTurn: 
        """
        turn = DialogTurn(message=message)
        turn._parent = self._parent
        if turn._parent is not None:
            turn._parent.children.remove(self)
            turn._parent.children.append(turn)
        turn.children.append(self)
        self._parent = turn
        return turn
    
    def append(self, message: 'Msg') -> 'DialogTurn':
        """

        Args:
            message (Msg): _description_

        Returns:
            DialogTurn: _description_
        """
        turn = DialogTurn(message=message)
        turn._parent = self
        self.children.append(turn)
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
        return self.children[idx]
    
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

        for child in self.children:
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

        for child in self.children:
            result = child.find(turn)
        return result

    def prune(self, idx: int) -> 'DialogTurn':
        """Remove the subtree specified by idx.

        Args:
            idx (int): The index of the child to prune.

        Returns:
            DialogTurn: The pruned subtree.
        """
        if idx < 0 or idx >= len(self.children):
            raise IndexError("Index out of range for pruning.")

        pruned_subtree = self.children.pop(idx)
        pruned_subtree._parent = None
        return pruned_subtree
    
    def index(self, turn: 'DialogTurn') -> int:

        return self.children.index(turn)
    
    def my_index(self) -> int | None:

        if self._parent is None:
            return None
        idx = self._parent.index(self)
        return idx

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
        my_idx = self.my_index()
        if my_idx is None:
            raise IndexError(
                f'Requesting sibling but no parent.'
            )
        sib_idx = idx + my_idx
        if sib_idx < 0:
            raise IndexError(
                f"{idx} is invalid sibling index"
            )
        return self._parent.child(sib_idx)
    
    def ancestor(self, count: int) -> 'DialogTurn':
        """

        Args:
            count (int): 

        Returns:
            DialogTurn: 
        """
        if not isinstance(count, int):
            raise TypeError(f"Count must be of type int not {type(count)}")
        turn = self
        i = 0
        while True:
            if i == count:
                return turn
            turn = turn._parent
            if turn is None:
                raise IndexError(
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

    def n_children(self) -> int:
        """
        Returns:
            int: The number of children
        """
        return len(self.children)
    
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
    _leaf: DialogTurn | None = PrivateAttr(
        default=None
        )
    _indices: typing.List[int] = PrivateAttr(default_factory=list)
    _counts: typing.List[int] = PrivateAttr(default_factory=list)
    root: DialogTurn | None = None

    def model_post_init(self, __context) -> None:
        # Runs after __init__ and validation
        self._update()

    def rise(self, count: int):
        
        self._leaf = self._leaf.ancestor(count)
        self._update()

    def leaf_child(self, idx: int):
        """Descend 

        Args:
            idx (int): The index of the child
        """
        self._leaf = self._leaf.child(idx)
        self._update()

    def leaf_sibling(self, idx: int):
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
        while turn.parent is not None:
            idx = turn.my_index()
            indices.insert(0, idx)
            turn = turn.parent

        indices.insert(
            0, 0
        )
        self._indices = indices

    def _update_counts(self):

        if self._leaf is None:
            self._counts = []
            return
        result = []
        for ancestor in self._leaf.ancestors:
            result.insert(
                0, ancestor.n_children()
            )
        result.insert(0, 1)

        self._counts = result

    @property
    def leaf(self) -> DialogTurn | None:
        return self._leaf

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
        turns = list(reversed(list(self._leaf.ancestors)))
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
        if self.root is None:
            return TreeDialog()

        # Perform a BFS to clone the tree
        # root_clone = DialogTurn(self._root._message)

        from collections import deque
        # turns = deque()
        counts = deque()
        dialog = TreeDialog()
        
        dialog.append(self.root.message)
        turn = self.root
        count = 0
        counts.append(0)
        leaf = None
        if turn is self._leaf:
            leaf = dialog._leaf
        while True:
            if turn.n_children() > count:
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
                dialog.rise(1)
                count += 1

        dialog._leaf = leaf
        dialog._update()
        return dialog

    def pop(
        self, idx: int=-1, get_msg: bool=False
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
        for turn in target.children:
            turn._parent = parent
            if parent is not None:
                parent.children.append(turn)
        
        if target._parent is not None:
            target._parent.children.remove(target)
        
        target._parent = None
        target.children = []
        if self._leaf is target:
            if turn.parent is not None:
                self._leaf = turn
            elif len(turn.children) > 0:
                self._leaf = turn.children[0]
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
            queue.extend(current.children)
        return None

    def remove(self, message: Msg):
        """Remove a message from the dialog

        If turn is the current "leaf" then
        it will set the turn to the parent if that is not None
        Or the first child or None

        Args:
            message (Msg): The message to remove
        """
        turn = self.root.find_val(message)
        if turn is self.root:
            raise ValueError(
                'Cannot remove root node in tree dialog.'
            )
        if turn is None:
            raise ValueError(
                f'Message {message} does not exist so cannot remove.'
            )
        for child in turn.children:
            child.parent = turn.parent
            if turn.parent is not None:
                turn.parent.append(child)
        if turn.parent is not None:
            turn.parent.children.remove(turn)
        if self._leaf is turn:
            if turn.parent is not None:
                self._leaf = turn
            elif len(turn.children) > 0:
                self._leaf = turn.children[0]
            else:
                self._leaf = None
        turn._parent = None
        turn.children.clear()
        self._update()
        return self

    def extend(self, dialog: typing.Iterable[Msg]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[&#39;Dialog&#39;, typing.List[Msg]]): _description_
        """
        for msg in dialog:
            if self._leaf is None:
                self._leaf = DialogTurn(message=msg)
                self.root = self._leaf
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
            self._leaf = DialogTurn(message=message)
            self.root = self._leaf
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
            self._leaf = DialogTurn(message=message)
            self.root = self._leaf
        else:
            turn_list = self._turn_list()
            turn = turn_list[ind]
            if ind == len(turn_list):
                self._leaf = self._leaf.append(message)
            else:
                inserted = turn.prepend(message)
                if ind == 0:
                    self.root = inserted
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




# class Msg(BaseModel):
#     """
#     """

#     role: str
#     alias: Optional[str] = None
#     type: str = "data"
#     content: Union[str, Dict[str, Any], None] = None
#     tools: typing.List[ToolCall] | typing.List[AsyncToolCall] | None = None

#     def __post_init__(self):
#         self.tools = (
#             self.tools 
#             or typing.List[ToolCall | AsyncToolCall]([])
#         )

#     def apply(self, func):
#         return func(self)

#     def output(self, key: str = "tool_out", default=None) -> Any:
#         return self.meta.get(key, default)

#     def render(self) -> str:
#         return f"{self.alias or self.role}: {self.content}"

#     def to_input(self) -> Dict[str, Any]:
#         if self.filtered:
#             return {}
#         return {"role": self.role, "content": self.content}

#     class Config:
#         extra = "allow"


# class ToolEvent(BaseModel):
#     """
#     A single tool lifecycle event (generic or computer-use).

#     Lifecycle:
#         call  →  delta*  →  result | error

#     Attributes:
#         id: Unique id for this event (optional but useful).
#         call_id: Correlates all events for one invocation.
#         name: Tool name, e.g. 'search', 'get_weather', 'computer.use'.
#         kind: 'generic' or 'computer'. Purely informative for adapters.
#         type: 'call' | 'delta' | 'result' | 'error'.
#         args: Arguments for the call (for 'call').
#         data: Small, structured telemetry (for 'delta').
#         result: Final structured output (for 'result').
#         error: Terminal error descriptor (for 'error').
#         artifacts: Large outputs as references (screenshots/files). No raw bytes.
#         ts: Event timestamp (UTC).
#         parent_id: Optional parent event id.
#         stream_id: Optional grouping key for multi-stream deltas.
#         computer: Optional computer-use info (when kind == 'computer').
#     """
#     id: t.Optional[str] = None
#     call_id: str
#     name: str
#     kind: str = "generic"  # 'generic' | 'computer'
#     type: str  # 'call' | 'delta' | 'result' | 'error'

#     args: t.Optional[t.Dict[str, t.Any]] = None
#     data: t.Optional[t.Dict[str, t.Any]] = None
#     result: t.Optional[t.Dict[str, t.Any]] = None
#     error: t.Optional[ToolError] = None

#     artifacts: t.List[Attachment] = Field(default_factory=list)

#     ts: datetime = Field(default_factory=lambda: datetime.utcnow())
#     parent_id: t.Optional[str] = None
#     stream_id: t.Optional[str] = None

#     computer: t.Optional[ComputerUseInfo] = None

#     class Config:
#         extra = "allow"

#     @property
#     def is_terminal(self) -> bool:
#         """Whether the event is terminal for its call_id."""
#         return self.type in ("result", "error")
