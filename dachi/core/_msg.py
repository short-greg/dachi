"""
Defines core Msg, Resp, and Dialog classes for conversation handling.




"""

# 1st party
from __future__ import annotations
import typing
import typing as t
from abc import abstractmethod
from typing import Self
import base64
from datetime import datetime, timezone

# 3rd party
from pydantic import BaseModel, PrivateAttr, ValidationError
import pydantic

# local
from . import Renderable
from ._tool import ToolUse

"""
Streaming Response Flow:

The response architecture handles both complete and streaming responses with clear separation:

1. First streaming chunk:
   resp = Resp(msg=Msg(text="Hello"))
   resp.delta = RespDelta(text="Hello")  # Same as msg.text for first chunk
   resp.out_store = {}  # Empty processing state
   
2. Second chunk arrives:
   resp.msg.text = "Hello world"         # Accumulated complete text
   resp.delta.text = " world"            # Just the new chunk
   resp.out_store["processor"] = {...}   # Updated processing state
   resp.out = current_processed_result   # Current processed output
   
3. Processing with output processors:
   # Processors use resp.out_store to maintain state across chunks
   # resp.out gets updated with current processed result
   
4. spawn() for next chunk:
   new_resp = resp.spawn(new_msg, chunk_data)
   # Copies out_store state and out value for continuity
   # Creates fresh delta for new chunk
   
Key principles:
- Resp.msg: Always contains complete accumulated state
- RespDelta: Only contains current chunk changes  
- out_store: Accumulates processor state across streaming
- out: Current processed result (any type)
"""

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
    """Single streaming chunk data - NOT accumulated values.
    
    Contains only the incremental changes for the current streaming chunk.
    All fields represent deltas/changes, not accumulated state.
    
    Field Meanings:
        text: New text content in this chunk only
        tool: Partial tool call JSON fragment for this chunk  
        thinking: New reasoning content in this chunk only
        citations: New citation information in this chunk only
        finish_reason: Set only when streaming completes
        usage: Token usage for this specific chunk
        
    Important: These are chunk deltas, not accumulated values.
    For accumulated values, use the parent Resp object fields.
    """
    
    text: str | None = pydantic.Field(
        default=None, description="Incremental text content for this streaming chunk only"
    )
    tool: str | None = pydantic.Field(
        default=None, description="Partial tool call arguments (JSON fragment) for this streaming chunk"
    )
    thinking: str | typing.Dict[str, typing.Any] | None = pydantic.Field(
        default=None, description="Incremental reasoning/thinking content for this streaming chunk only. Can be string or structured dict."
    )
    citations: typing.List[typing.Dict] | None = pydantic.Field(
        default=None, description="Incremental citation information for this streaming chunk only"
    )
    finish_reason: str | None = pydantic.Field(
        default=None, description="Reason generation stopped (e.g., 'stop', 'length', 'tool_calls') - set only when streaming completes"
    )
    usage: typing.Dict[str, typing.Any] | None = pydantic.Field(
        default=None, description="Per-chunk token usage statistics for this specific chunk. Can include nested structures."
    )


class Resp(pydantic.BaseModel):
    """Complete response from an LLM with streaming accumulation support.
    
    This class represents a unified response format that separates complete message 
    content from processing metadata and streaming chunks. It supports both complete
    and streaming response patterns with clear separation of concerns.
    
    Core Architecture:
        - Accumulates streaming chunks into complete response
        - Separates message content from processing metadata
        - Handles both complete and streaming response patterns
    
    Field Purposes:
        msg: Complete accumulated message ready for LLM consumption
             - msg.text contains full accumulated text (not deltas)
             - Always represents final/current complete state
        
        out: Processed output value from response processors
             - Can be any type: str, int, BaseModel, dict, tuple, None
             - Ephemeral processing result (not serialized)
             
        delta: Current streaming chunk information only
               - Contains incremental changes for this chunk
               - Reset/updated for each streaming iteration
               
        out_store: State storage for output processors during streaming
                   - Accumulates processing state across chunks
                   - Used by ToOut processors for stateful streaming
        
        data: Internal storage for raw API responses and processing state
        
    Usage Patterns:
        Complete response:
            resp.msg contains final message, resp.out contains final result
            
        Streaming response:
            resp.msg accumulates over chunks, resp.delta shows current chunk
            resp.out_store maintains processing state across chunks
            
        Access patterns:
            text = resp.msg.text     # Complete accumulated text
            result = resp.out        # Processed output (any type)
            chunk = resp.delta.text  # Current streaming chunk only
    """

    # Core content
    msg: Msg | None = pydantic.Field(
        default=None, description="The complete, accumulated message content (not delta). Contains final text for completed responses, accumulated text for streaming."
    )
    tool: typing.List[typing.Dict] | None = pydantic.Field(
        default=None, description="Complete tool call objects with results for non-streaming responses"
    )
    thinking: str | typing.Dict[str, typing.Any] | None = pydantic.Field(
        default=None, description="Complete model reasoning/thought process content (accumulated, not delta). Can be string or structured dict."
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
    usage: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=dict, description="Token usage statistics (prompt_tokens, completion_tokens, etc.). Can include nested structures for detailed token breakdowns."
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
    
    # Output processing state
    out_store: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=dict, description="State storage for output processors during streaming"
    )
    
    # Private attributes
    _data: typing.Dict = pydantic.PrivateAttr(
        default_factory=dict
    )
    _delta: typing.Dict = pydantic.PrivateAttr(
        default_factory=dict
    )
    _out: typing.Union[typing.Dict, typing.Any, typing.Tuple, None] = pydantic.PrivateAttr(
        default=None
    )

    @property
    def data(self) -> typing.Any:
        """ Get the raw data from the response.

        Returns:
            typing.Any: The raw data from the response.
        """
        return self._data
    
    @data.setter
    def data(self, value: typing.Any) -> None:
        """Set the raw data for the response."""
        self._data = value
        
    # Note: delta is now a proper RespDelta field, no property needed
    
    @property
    def out(self) -> typing.Union[typing.Dict, typing.Any, typing.Tuple, None]:
        """Get the output values from the response.
        
        The out attribute can contain processed output from response processors:
        - dict: Key-value pairs for structured outputs
        - single value: For simple outputs (str, int, bool, etc.)
        - tuple: For multiple outputs
        - None: When no processing has been done

        Returns:
            typing.Union[typing.Dict, typing.Any, typing.Tuple, None]: The processed output values
        """
        return self._out
    
    @out.setter
    def out(self, value: typing.Union[typing.Dict, typing.Any, typing.Tuple, None]) -> None:
        """Set the output values for the response."""
        self._out = value
    
    def spawn(self, msg: Msg, data: typing.Dict=None, follow_up: bool=None) -> 'Resp':
        """Create new response for next streaming chunk.
        
        Preserves accumulation state (out_store, out) while allowing new message 
        content and chunk data. Used to maintain processing continuity across 
        streaming chunks without losing processor state.
        
        Args:
            msg: New accumulated message state (complete text so far)
            data: Raw chunk data for this iteration  
            follow_up: Optional follow-up message flag
            
        Returns:
            New Resp with preserved processing state, fresh delta ready for new chunk
            
        Usage:
            # During streaming - maintain processor state across chunks
            new_resp = prev_resp.spawn(
                msg=Msg(text=accumulated_text),
                data=api_chunk_data  
            )
            # new_resp.out_store contains accumulated processor state
            # new_resp.delta will be updated with new chunk deltas
        """
        data = data if data is not None else {}
        resp = Resp(
            msg=msg,
            val=self.val,
            follow_up=follow_up,
        )
        resp.data.update(data)
        resp.out = self._out
        # Copy processing state for continuity
        resp.out_store.update(self.out_store)
        # Copy current delta state (will be updated with new chunk)
        if self.delta:
            resp.delta.text = self.delta.text
            resp.delta.tool = self.delta.tool  
            resp.delta.thinking = self.delta.thinking
            resp.delta.citations = self.delta.citations
            resp.delta.finish_reason = self.delta.finish_reason
        return resp
    
    def tool(self) -> typing.Dict[str, typing.Any]:
        """Execute the tool calls in the response and return the new response(s).

        Returns:
            None Resp | typing.List[typing.Dict]: List of tool call dictionaries matching the name.
        """
        result = {}
        if self.msg is not None and self.msg.tool_calls:
            for tool_call in self.msg.tool_calls:
                res = tool_call()

                # Need to append the result to the dictionary
                result[tool_call.name] = res
        return result


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
    Represents a complete, accumulated message in a conversation.
    
    This class contains the full, final content of a message - not incremental deltas.
    For streaming responses, this represents the accumulated text so far, not just 
    the latest chunk. Delta information is stored separately in RespDelta.
    
    Key concepts:
        - Role: Indicates who sent the message ('user', 'assistant', 'system', etc.)
        - Text: The complete accumulated content, not just the latest delta
        - Tool calls: Complete tool/function calls with their results
        - Attachments: Files, images, or other media associated with the message
        
    Usage:
        Complete message:
            msg = Msg(role='user', text='Hello, how are you?')
            
        Accumulated streaming message:
            # During streaming, text contains all content received so far
            msg = Msg(role='assistant', text='The answer is 42 and here is why...')
            
        Message with tool calls:
            msg = Msg(role='assistant', text='I found the information')
            msg.tool_calls.append(ToolUse(name='search', arguments={'query': 'python'}))

    Notes:
        - Text may be plain text or structured dict with channels (e.g., {"final": "...", "thinking": "..."})
        - Attachments are declarative refs; adapters handle upload/encoding
        - Tool events include complete calls, not partial deltas
        - Provider/runtime metadata should live on Resp, not here
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

    def add(self, message: Msg | Resp) -> Msg:
        """Alias for append to add a message to the end of the current path."""
        if isinstance(message, Resp):
            message = message.msg
        self.append(message)
        return message

    @abstractmethod
    def replace(
        self, message: Msg | Resp, ind: int
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
        self, dialog: typing.Union['BaseDialog', typing.Iterable[Msg | Resp]], 
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

    def extend(self, dialog: typing.Iterable[Msg | Resp]):
        """Extend the dialog with another dialog or a list of messages

        Args:
            dialog (typing.Union[Dialog;, typing.List[Msg]]): The dialog or list of messages to extend with
        """
        if isinstance(dialog, BaseDialog):
            dialog = dialog.aslist()
        
        validated = []
        for msg in dialog:
            if isinstance(msg, Resp):
                msg = msg.msg
            if not isinstance(msg, Msg):
                raise ValueError(
                    "List dialog must only consist of messages."
                )
            validated.append(msg)

        self.messages.extend(validated)
        return self

    def __len__(self) -> int:
        """Get the size of the dialog

        Returns:
            int: the number of turns in the dialog
        """
        return len(self.messages)
        
    def append(self, message: Msg | Resp) -> Self:
        """Add a message to the end of the dialog

        Args:
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if isinstance(message, Resp):
            message = message.msg
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

    def replace(self, idx: int, message: Msg | Resp) -> 'BaseDialog':
        """Add a message to the dialog

        Args:
            idx (typing.Optional[int], optional): The index to add. Defaults to None.
            message (Msg): The message to add
            replace (bool, optional): Whether to replace at the index. Defaults to False.

        Raises:
            ValueError: If the index is not correct
        """
        if isinstance(message, Resp):
            message = message.msg

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
    _root: str | None = PrivateAttr(default=None)
    _messages: typing.Dict[str, Msg] = PrivateAttr(default_factory=dict)
    _leaf: str | None = PrivateAttr(default=None)
    _parent: typing.Dict[str, str] = PrivateAttr(default_factory=dict)
    _children: typing.Dict[str, typing.List[str]] = PrivateAttr(default_factory=dict)
    _indices: typing.List[int] = PrivateAttr(default_factory=list)
    _counts: typing.List[int] = PrivateAttr(default_factory=list)
    _next_id: int = PrivateAttr(default=0)
        
    def model_post_init(self, __context) -> None:
        """Initialize after model validation."""
        self._update()
        
    def _generate_id(self) -> str:
        """Generate a unique node ID."""
        node_id = str(self._next_id)
        self._next_id += 1
        return node_id
        
    def _get_path_to_leaf(self) -> typing.List[str]:
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
    def indices(self) -> typing.List[int]:
        """Get a copy of the current path indices."""
        return [*self._indices]
        
    @property
    def counts(self) -> typing.List[int]:
        """Get a copy of the current level counts."""
        return [*self._counts]
        
    @property
    def root(self) -> Msg | None:
        """Get the root message."""
        if self._root is None:
            return None
        return self._messages[self._root]
        
    @property
    def leaf(self) -> Msg | None:
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

    def __iter__(self) -> typing.Iterator[Msg]:
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
        dialog: typing.Union['BaseDialog', typing.Iterable[Msg | Resp]], 
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

    def append(self, message: Msg | Resp) -> Self:
        """Add a message to the end of the current path."""
        if isinstance(message, Resp):
            message = message.msg
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
        return message

    def insert(self, ind: int, message: Msg) -> Self:
        """Insert a message at the specified index in the current path."""
        if not isinstance(message, Msg):
            raise ValueError("Message must be a Msg instance")
            
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

    def replace(self, idx: int, message: Msg | Resp) -> 'BaseDialog':
        """Replace the message at the specified index."""
        if isinstance(message, Resp):
            message = message.msg
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


# class DialogTurn(pydantic.BaseModel):
#     """A single tuurn in the dialog.
#     """

#     message: Msg
#     children: typing.Union[typing.List['DialogTurn']] = pydantic.Field(default_factory=list)
#     _parent: typing.Union['DialogTurn', None] = pydantic.PrivateAttr(default=None)

#     def model_post_init(self, __context):
#         for child in self.children:
#             child._parent = self
#             child.model_post_init(__context)

#     def root(self) -> 'DialogTurn':

#         node = self
#         while node.parent is not None:
#             node = node._parent
#         return node

#     def leaf(self) -> 'DialogTurn':
#         """Get the leftmost leaf node below the current node

#         Returns:
#             DialogTurn: The leaf node
#         """
#         if len(self.children) == 0:
#             return self
#         return self.children[0].leaf()
    
#     @property
#     def ancestors(self) -> typing.Iterator['DialogTurn']:

#         turn = self
#         while True:
#             if turn._parent is None:
#                 return
#             turn = turn._parent
#             yield turn
    
#     def prepend(self, message: 'Msg') -> 'DialogTurn':
#         """

#         Args:
#             turn (DialogTurn): 

#         Returns:
#             DialogTurn: 
#         """
#         turn = DialogTurn(message=message)
#         turn._parent = self._parent
#         if turn._parent is not None:
#             turn._parent.children.remove(self)
#             turn._parent.children.append(turn)
#         turn.children.append(self)
#         self._parent = turn
#         return turn
    
#     def append(self, message: 'Msg') -> 'DialogTurn':
#         """

#         Args:
#             message (Msg): _description_

#         Returns:
#             DialogTurn: _description_
#         """
#         turn = DialogTurn(message=message)
#         turn._parent = self
#         self.children.append(turn)
#         return turn

#     def ancestor(self, count: int) -> 'DialogTurn':
#         """Get an an answer

#         Args:
#             count (int): The number to ascend

#         Returns:
#             DialogTurn: The ancestor
#         """

#         i = 0
#         turn = self
#         while True:
#             if i == count:
#                 return turn.message
#             if turn._parent is None:
#                 raise RuntimeError(
#                     "There are only "
#                     f"{i} ancestors yet"
#                     f"you passed {count}"
#                 )
#             turn = turn._parent
#             i += 1

#     def depth(self) -> int:
#         """Calculate the depth of the current turn in the dialog tree.

#         Returns:
#             int: The number of turns to the root turn.
#         """
#         depth = 1
#         turn = self
#         while turn._parent is not None:
#             depth += 1
#             turn = turn._parent
#         return depth
    
#     def child(self, idx: int) -> 'DialogTurn':
#         """Get the child specified by index

#         Args:
#             idx (int): The index to retrieve

#         Returns:
#             DialogTurn: 
#         """
#         return self.children[idx]
    
#     def find_val(self, message: Msg) -> typing.Optional['DialogTurn']:
#         """Search through all children to find the message.

#         Args:
#         message (Msg): The message to find.

#         Returns:
#         typing.Optional[DialogTurn]: The DialogTurn containing the message, or None if not found.
#         """
#         result = None
#         if self.message == message:
#             return self

#         for child in self.children:
#             result = child.find_val(message)

#         return result

#     def find(self, turn: 'DialogTurn') -> typing.Optional['DialogTurn']:
#         """Search through all children to find the message.

#         Args:
#         message (Msg): The message to find.

#         Returns:
#         typing.Optional[DialogTurn]: The DialogTurn containing the message, or None if not found.
#         """
#         result = None
#         if self == turn:
#             return self

#         for child in self.children:
#             result = child.find(turn)
#         return result

#     def prune(self, idx: int) -> 'DialogTurn':
#         """Remove the subtree specified by idx.

#         Args:
#             idx (int): The index of the child to prune.

#         Returns:
#             DialogTurn: The pruned subtree.
#         """
#         if idx < 0 or idx >= len(self.children):
#             raise IndexError("Index out of range for pruning.")

#         pruned_subtree = self.children.pop(idx)
#         pruned_subtree._parent = None
#         return pruned_subtree
    
#     def index(self, turn: 'DialogTurn') -> int:

#         return self.children.index(turn)
    
#     def my_index(self) -> int | None:

#         if self._parent is None:
#             return None
#         idx = self._parent.index(self)
#         return idx

#     def sibling(self, idx: int) -> int:
#         """
#         Returns:
#             int: The index for the sibling
#         """
#         if self._parent is None:
#             if idx != 0:
#                 raise RuntimeError(
#                     "There is no parent so must be 1."
#                 )
#             return self
#         my_idx = self.my_index()
#         if my_idx is None:
#             raise IndexError(
#                 f'Requesting sibling but no parent.'
#             )
#         sib_idx = idx + my_idx
#         if sib_idx < 0:
#             raise IndexError(
#                 f"{idx} is invalid sibling index"
#             )
#         return self._parent.child(sib_idx)
    
#     def ancestor(self, count: int) -> 'DialogTurn':
#         """

#         Args:
#             count (int): 

#         Returns:
#             DialogTurn: 
#         """
#         if not isinstance(count, int):
#             raise TypeError(f"Count must be of type int not {type(count)}")
#         turn = self
#         i = 0
#         while True:
#             if i == count:
#                 return turn
#             turn = turn._parent
#             if turn is None:
#                 raise IndexError(
#                     f"Cannot ascend {count}."
#                     f"Only {i} parents."
#                 )
#             i += 1
    
#     @property
#     def parent(self) -> typing.Union['DialogTurn', None]:
#         """get the parent to this dialgo turn

#         Returns:
#             DIalogTurn: the parent dialog turn
#         """
#         return self._parent

#     def n_children(self) -> int:
#         """
#         Returns:
#             int: The number of children
#         """
#         return len(self.children)
    
#     def n_siblings(self) -> int:
#         """The number of siblings for the turn

#         Returns:
#             int: 
#         """
#         if self._parent is None:
#             return 1
        
#         return self._parent.n_children
