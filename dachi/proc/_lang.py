import typing as t
from abc import abstractmethod, ABC
import pydantic
from ._process import (
    Process,
    AsyncProcess,
    StreamProcess,
    AsyncStreamProcess,
)
from dachi.core import Inp, Registry, Runtime, Module
from dachi.core._base import ToolResult, ToolMsg

from ._resp import ToOut

# =============================================================================
# Tool Use Types
# =============================================================================

class BaseToolCall(pydantic.BaseModel):
    """Base class for tool invocations.

    Stores a callable and its arguments for deferred execution.
    Not serializable - these are ephemeral objects created and
    discarded within a single LLM interaction.

    Attributes:
        id: Optional tool call ID from LLM API (e.g., "call_abc123", "toolu_01A09...")
        func: The callable to invoke (function, method, or callable object)
        args: Named arguments to pass to the callable
    """
    id: str | None = None
    func: t.Callable
    args: t.Dict[str, t.Any] = pydantic.Field(default_factory=dict)


class Call(BaseToolCall, Process):
    """Synchronous tool invocation."""

    def forward(self) -> t.Any:
        return self.func(**self.args)


class AsyncCall(BaseToolCall, AsyncProcess):
    """Asynchronous tool invocation."""

    async def aforward(self) -> t.Any:
        return await self.func(**self.args)


class StreamCall(BaseToolCall, StreamProcess):
    """Streaming synchronous tool invocation."""

    def stream(self) -> t.Iterator[t.Any]:
        yield from self.func(**self.args)


class AsyncStreamCall(BaseToolCall, AsyncStreamProcess):
    """Streaming asynchronous tool invocation."""

    async def astream(self) -> t.AsyncIterator[t.Any]:
        async for item in self.func(**self.args):
            yield item


ToolCall = Call | AsyncCall | StreamCall | AsyncStreamCall


class ToolUse(Module):
    """Container for LLM tool use decision.

    Returned by LangModel when the LLM decides to call tools.
    Ephemeral object - execute tools and discard, not part of message history.

    Attributes:
        text: Optional accompanying text from the LLM (e.g., "Let me search...")
        calls: List of tool invocations to execute

    Example:
        result, messages, raw = model.forward(prompt, tools={"add": add_fn})

        if isinstance(result, ToolUse):
            print(result.text)
            for call in result.calls:
                if isinstance(call, Call):
                    output = call.forward()
                elif isinstance(call, AsyncCall):
                    output = await call.aforward()
    """
    text: str = ""
    calls: t.List[BaseToolCall] = pydantic.Field(default_factory=list)

    def forward(self) -> t.List[ToolResult]:
        """Execute all tool calls synchronously and return a ToolMsg."""
        tool_results = []
        for idx, call in enumerate(self.calls):
            if isinstance(call, Call):
                output = call.forward()
            elif isinstance(call, StreamCall):
                output = "".join(list(call.stream()))
            elif isinstance(call, AsyncCall):
                raise ValueError(f"Cannot execute async call {call} in synchronous forward.")
            elif isinstance(call, AsyncStreamCall):
                raise ValueError(f"Cannot execute async streaming call {call} in synchronous forward.")
            else:
                raise ValueError(f"Cannot execute async or streaming call {call} in synchronous forward.")
            tool_result = ToolResult(
                id=call.id or f"call_{idx}",
                output=str(output)
            )
            tool_results.append(tool_result)
        
        return tool_results
    
    async def aforward(self) -> t.List[ToolResult]:
        """Execute all tool calls asynchronously and return a ToolMsg."""
        tool_results = []
        for idx, call in enumerate(self.calls):
            if isinstance(call, Call):
                output = call.forward()
            elif isinstance(call, AsyncCall):
                output = await call.aforward()
            elif isinstance(call, StreamCall):
                output = "".join(list(call.stream()))
            elif isinstance(call, AsyncStreamCall):
                output = "".join([chunk async for chunk in call.astream()])
            else:
                raise ValueError(f"Unknown call type: {call}")
            tool_result = ToolResult(
                id=call.id or f"call_{idx}",
                output=str(output)
            )

            tool_results.append(tool_result)
        return tool_results
    
    def merge(self, other: "ToolUse"):
        """Merge another ToolUse into this one."""
        self.calls.extend(other.calls)
        if other.text:
            if self.text:
                self.text += "\n" + other.text
            else:
                self.text = other.text

    def empty(self) -> bool:
        """Check if there are no tool calls."""
        return len(self.calls) == 0
    
    def __len__(self) -> int:
        return len(self.calls)


def lang_forward(
    prompt: list[Inp] | Inp,
    structure: t.Dict | None | pydantic.BaseModel = None,
    tools: t.Dict | None | pydantic.BaseModel = None,
    _model: t.Optional["LangModel"] = None,
    _out: t.Optional[ToOut] = None,
    **kwargs
) -> t.Tuple[str | ToolUse, t.List[Inp], t.Any]:
    """Helper function to call LLM forward method."""
    if _model is None:
        raise ValueError("Model must be provided for llm_forward.")
    res, msgs, raw = _model.forward(
        prompt,
        structure=structure,
        tools=tools,
        **kwargs
    )
    if _out is not None and isinstance(res, str):
        res = _out.process(res)
    return res, msgs, raw


async def lang_aforward(
    prompt: list[Inp] | Inp,
    structure: t.Dict | None | pydantic.BaseModel = None,
    tools: t.Dict | None | pydantic.BaseModel = None,
    _model: t.Optional["LangModel"] = None,
    _out: t.Optional[ToOut] = None,
    **kwargs
) -> t.Tuple[str | ToolUse, t.List[Inp], t.Any]:
    """Helper function to call LLM aforward method."""
    if _model is None:
        raise ValueError("Model must be provided for llm_aforward.")
    res, msgs, raw = await _model.aforward(
        prompt,
        structure=structure,
        tools=tools,
        **kwargs
    )
    if _out is not None and isinstance(res, str):
        res = _out.process(res)
    return res, msgs, raw


def lang_stream(
    prompt: list[Inp] | Inp,
    structure: t.Dict | None | pydantic.BaseModel = None,
    tools: t.Dict | None | pydantic.BaseModel = None,
    _model: t.Optional["LangModel"] = None,
    _out: t.Optional[ToOut] = None,
    **kwargs
) -> t.Iterator[t.Tuple[str | ToolUse, t.List[Inp], t.Any]]:
    """Helper function to call LLM stream method."""
    if _model is None:
        raise ValueError("Model must be provided for llm_stream.")
    delta_store = {}
    for res, msgs, raw in _model.stream(
        prompt,
        structure=structure,
        tools=tools,
        **kwargs
    ):
        if _out is not None and isinstance(res, str):
            res = _out.delta(res, delta_store)
        yield res, msgs, raw
    if _out is not None and isinstance(res, str):
        res = _out.delta(res, delta_store, True)
    else:
        res = ''
    yield res, msgs, raw


async def lang_astream(
    prompt: list[Inp] | Inp,
    structure: t.Dict | None | pydantic.BaseModel = None,
    tools: t.Dict | None | pydantic.BaseModel = None,
    _model: t.Optional["LangModel"] = None,
    _out: t.Optional[ToOut] = None,
    **kwargs
) -> t.AsyncIterator[t.Tuple[str | ToolUse, t.List[Inp], t.Any]]:
    """Helper function to call LLM astream method."""
    if _model is None:
        raise ValueError("Model must be provided for llm_astream.")
    delta_store = {}
    async for res, msgs, raw in _model.astream(
        prompt,
        structure=structure,
        tools=tools,
        **kwargs
    ):
        if _out is not None and isinstance(res, str):
            res = _out.delta(res, delta_store)
        yield res, msgs, raw
    if _out is not None and isinstance(res, str):
        res = _out.delta(res, delta_store, True)
    else:
        res = ''
    yield res, msgs, raw


class LangModel(
    Process, 
    AsyncProcess, 
    StreamProcess, 
    AsyncStreamProcess
):
    """A simple LLM process that echoes input with a prefix.
    """

    @abstractmethod
    def forward(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str | ToolUse, t.List[Inp], t.Any]:
        """Synchronous LLM response.

        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the LLM's response (str | ToolUse, List[Inp], Any). The first element is the response (text str or ToolUse),
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), and the third element is the raw response from the LLM

        """
        pass

    @abstractmethod
    async def aforward(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str | ToolUse, t.List[Inp], t.Any]:
        """Asynchronous LLM response.

        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            A tuple containing the LLM's response (str | ToolUse, List[Inp], Any). The first element is the response (text str or ToolUse),
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), and the third element is the raw response from the LLM

        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[t.Any, t.List[Inp], t.Any]]:
        """Streaming synchronous LLM response.

        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            An iterator yielding tuples containing the LLM's response (Any, List[Inp], Any). The first element is the response (could be text chunks, ToolUse, or other streaming data),
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), the message for the current call will not be added until the stream is complete, and the third element is the raw response from the LLM, could be a chunk object
        """
        pass

    @abstractmethod
    async def astream(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[t.Any, t.List[Inp], t.Any]]:
        """Streaming asynchronous LLM response.

        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            An async iterator yielding tuples containing the LLM's response (Any, List[Inp], Any). The first element is the response (could be text chunks, ToolUse, or other streaming data),
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), the message for the current call will not be added until the stream is complete, and the third element is the raw response from the LLM, could be a chunk object
        """
        pass


LANG_MODEL = t.TypeVar("LANG_MODEL", bound=LangModel)

Engines = Registry[LangModel]()


class LangEngine(
    Process, 
    AsyncProcess, 
    StreamProcess, 
    AsyncStreamProcess
):
    """An operation that uses a language model to process input and generate output.
    """
    def model_post_init(self, __context):
        """Post init to set up model property.
        """
        super().model_post_init(__context)
        self._model = Runtime[LangModel](
            data=None
        )

    @property
    def model(self) -> LangModel | str | None:
        """The language model used by this operation.
        """
        return self._model.data
    
    @model.setter
    def model(
        self, 
        value: LangModel | str | None
    ) -> LangModel | str | None:
        self._model.data = value
        return self._model.data

    def get_model(self, override: t.Optional[LangModel | str]=None) -> LangModel:
        """Get the language model, raising an error if it is not set.
        """
        if override is None:
            override = self._model.data
        if override is None:
            raise ValueError("Model is not set.")

        if isinstance(override, str):
            return Engines[override].obj

        return override
    
    def forward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return lang_forward(prompt, structure=structure, tools=tools, _model=model, **kwargs)
    
    async def aforward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return await lang_aforward(prompt, structure=structure, tools=tools, _model=model, **kwargs)
    
    def stream(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp], t.Any]]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return lang_stream(prompt, structure=structure, tools=tools, _model=model, **kwargs)
    
    async def astream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        _model: LangModel | None = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp], t.Any]]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return await lang_astream(prompt, structure=structure, tools=tools, _model=model, **kwargs)


class LangOp(Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """An operation that uses a language model to process input and generate output.
    """

    def forward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, 
        _out: ToOut | None = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        if _model is None:
            raise ValueError("Model must be provided for LangOp operations.")
        res, _, _ = _model.forward(prompt, structure=structure, tools=tools, **kwargs)
        if _out is not None and isinstance(res, str):
            res = _out.process(res)
        
        return res
    
    async def aforward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, 
        _out: ToOut | None = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        if _model is None:
            raise ValueError("Model must be provided for LangOp operations.")
        res, _, _ = await _model.aforward(prompt, structure=structure, tools=tools, **kwargs)
        if _out is not None and isinstance(res, str):
            res = _out.process(res)
        
        return res
    
    def stream(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, 
        _out: ToOut | None = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp], t.Any]]:
        if _model is None:
            raise ValueError("Model must be provided for LangOp operations.")
        delta_store = {}
        for res, msgs, raw in _model.stream(prompt, structure=structure, tools=tools, **kwargs):
            if _out is not None and isinstance(res, str):
                res = _out.delta(res, delta_store) 
            yield res
        if _out is not None and isinstance(res, str):
            res = _out.delta(res, delta_store, True)
        yield res

    async def astream(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, 
        _out: ToOut | None = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp], t.Any]]:
        if _model is None:
            raise ValueError("Model must be provided for LangOp operations.")
        delta_store = {}
        async for res, msgs, raw in _model.astream(prompt, structure=structure, tools=tools, **kwargs):
            if _out is not None and isinstance(res, str):
                res = _out.delta(res, delta_store) 
            yield res
        if _out is not None and isinstance(res, str):
            res = _out.delta(res, delta_store, True)
        yield res


class ToolUser(LangEngine):
    """An engine that uses tools with a language model.
    """

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._max_iterations = 10

    def forward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, 
        _out: ToOut | None = None,
        _callback: t.Callable | None = None,
        **kwargs
    ) -> t.Tuple[t.Any, t.List[Inp], t.Any]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        
        msgs = [*prompt] if isinstance(prompt, list) else [prompt]
        for iteration in range(self._max_iterations):
            res, msgs, raw = lang_forward(
                prompt=msgs,
                structure=structure,
                tools=tools,
                _model=model,
                _out=_out,
                **kwargs
            )
            if not isinstance(res, ToolUse):
                return res, msgs, raw

            tool_results = res.forward()
            if _callback is not None:
                _callback(res, tool_results, iteration, msgs)
            msg = ToolMsg(
                role="tool",
                text='',
                tool_calls=tool_results
            )
            msgs = msgs + [msg]

        raise RuntimeError(f"Max tool use iterations ({self._max_iterations}) exceeded.")

    async def aforward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, 
        _out: ToOut | None = None,
        _callback: t.Callable | None = None,
        **kwargs
    ) -> t.Tuple[t.Any, t.List[Inp], t.Any]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        
        msgs = [*prompt] if isinstance(prompt, list) else [prompt]
        for iteration in range(self._max_iterations):
            res, msgs, raw = await lang_aforward(
                prompt=msgs,
                structure=structure,
                tools=tools,
                _model=model,
                _out=_out,
                **kwargs
            )
            if not isinstance(res, ToolUse):
                return res, msgs, raw

            tool_results = await res.aforward()
            if _callback is not None:
                _callback(res, tool_results, iteration, msgs)
            msg = ToolMsg(
                role="tool",
                text='',
                tool_calls=tool_results
            )
            msgs = msgs + [msg]

        raise RuntimeError(f"Max tool use iterations ({self._max_iterations}) exceeded.")
    
    def stream(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        _model: LangModel | None = None,
        _out: ToOut | None = None,
        _callback: t.Callable | None = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[t.Any, t.List[Inp], t.Any]]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")

        msgs = [*prompt] if isinstance(prompt, list) else [prompt]
        for iteration in range(self._max_iterations):
            tool_use = ToolUse()
            for res, msgs, raw in lang_stream(
                prompt=msgs,
                structure=structure,
                tools=tools,
                _model=model,
                _out=_out,
                **kwargs
            ):
                
                if isinstance(res, ToolUse):
                    tool_use.merge(res)
                yield res, msgs, raw

            if tool_use.empty():
                return
            
            tool_results = tool_use.forward()
            if _callback is not None:
                _callback(tool_use, tool_results, iteration, msgs)
            msg = ToolMsg(
                role="tool",
                text='',
                tool_calls=tool_results
            )
            msgs = msgs + [msg]

        raise RuntimeError(f"Max tool use iterations ({self._max_iterations}) exceeded.")

    async def astream(
        self,
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        _model: LangModel | None = None,
        _out: ToOut | None = None,
        _callback: t.Callable | None = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[t.Any, t.List[Inp], t.Any]]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")

        msgs = [*prompt] if isinstance(prompt, list) else [prompt]
        for iteration in range(self._max_iterations):

            tool_use = ToolUse()
            async for res, msgs, raw in lang_astream(
                prompt=msgs,
                structure=structure,
                tools=tools,
                _model=model,
                _out=_out,
                **kwargs
            ):
                
                if isinstance(res, ToolUse):
                    tool_use.merge(res)
                yield res, msgs, raw

            if tool_use.empty():
                return

            tool_results = await tool_use.aforward()
            if _callback is not None:
                _callback(tool_use, tool_results, iteration, msgs)
            msg = ToolMsg(
                role="tool",
                text='',
                tool_calls=tool_results
            )
            msgs = msgs + [msg]

        raise RuntimeError(f"Max tool use iterations ({self._max_iterations}) exceeded.")


OP = t.TypeVar("OP", bound=LangEngine)
