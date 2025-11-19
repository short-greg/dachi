# 1st party
import typing
import json
from abc import abstractmethod
from typing import (
    Callable, Type, Optional,
    Any, Dict, get_type_hints
)
import typing as t
from inspect import signature, Parameter
import inspect

# 3rd party
import pydantic
from pydantic import create_model, BaseModel
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from dataclasses import InitVar

# local
from ..utils import is_async_function

# 1) How can the tool be used by the LLM
# 2) So I would simply pass the tool to the
#    resp conv.
#    Then I need some different ways to define
#    tools

# Tool
#   - How to define the tool
#   - 1) use text to parse (?)
#   - 2) use the LLM to parse [if the LLM does not support tools]
#   - 3) just create the tool from a decorator
#   - 4) create the tool from a Pydantic BaseModel
#   - 5) 
# ToolResp(
#   tools=f, pydantic etc
# )
# I'd want a way to convert the
# function or the pydantic to
# the format necessary for a tool

# The "RespConv" has access to the tools
# If I keep it controlled by the RespConv

# The RespConv outputs the ToolCall
# tool


# I want to define a tool registry 
# for tools. The user should define a tool
# by using the @tool decoractor
# so let's define the registry


# then let's define the decorator, that will create
# the ToolDef and register it

_tool_registry: Dict[str, Callable] = {}


def get_tool_function(name: str) -> Callable:
    """Get a registered tool function by name."""
    if name not in _tool_registry:
        raise KeyError(f"Tool function '{name}' not found in registry")
    return _tool_registry[name]


def list_tool_functions() -> Dict[str, Callable]:
    """Get all registered tool functions."""
    return _tool_registry.copy()


def tool(func: Callable) -> 'BaseTool':
    """Decorator that replaces function with Tool/AsyncTool instance."""
    # Create Tool/AsyncTool instance with metadata (this will register the function)
    tool_instance = register_tool(func)
    
    return tool_instance


def register_tool(
    func: Callable,
    name_override: str | None = None,
    description_override: str | None = None
) -> 'BaseTool':
    """Register a tool function and create a tool definition from it.
    """
    sig = signature(func)
    hints = get_type_hints(func)
    fields: Dict[str, tuple[type, Any]] = {}

    name = name_override or func.__qualname__
    description = description_override or (func.__doc__ or "").strip() or f"Tool for {name}" + f"\n\n {signature}"
    
    # Register the function in the private registry
    _tool_registry[name] = func

    for param_name, param in sig.parameters.items():
        if param.kind == Parameter.VAR_POSITIONAL:
            raise TypeError(f"Function {func.__name__} has *args (VAR_POSITIONAL), which is not supported for tool use.")
        if param.kind == Parameter.VAR_KEYWORD:
            raise TypeError(f"Function {func.__name__} has **kwargs (VAR_KEYWORD), which is not supported for tool use.")

        typ = hints.get(param_name) or (
            type(param.default) if param.default is not Parameter.empty else Any
        )
        default = param.default if param.default is not Parameter.empty else ...
        fields[param_name] = (typ, default)

    InputModel = create_model(
        f"{name}Inputs", **fields, __base__=BaseModel
    )

    if is_async_function(func):
        return AsyncTool(
            name=name,
            description=description,
            # fn=func,
            input_model=InputModel,
            return_type=hints.get("return", None)
        )
    return Tool(
        name=name,
        description=description,
        # fn=func,
        input_model=InputModel,
        return_type=hints.get("return", None)
    )


class BaseTool(pydantic.BaseModel):
    """A definition of a tool that can be called by the LLM
    Pass this to the LLM to invoke the tool.
    """
    name: str
    description: str
    input_model: Type[pydantic.BaseModel]  # pydantic model for inputs
    # fn: Callable
    return_type: Optional[Type[Any]] = None  # for future use
    version: Optional[str] = None            # optional metadata
    
    @property
    @abstractmethod
    def is_async(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass


class AsyncTool(BaseTool):
    """A definition of an async tool that can be called by the LLM."""
    async def __call__(self, *args, **kwargs):
        fn = get_tool_function(self.name)
        sig = inspect.signature(fn)
        param_names = list(sig.parameters)
        bound_kwargs = {}
        for i, arg in enumerate(args):
            if i >= len(param_names):
                raise TypeError(f"Too many positional arguments for tool '{self.name}'")
            bound_kwargs[param_names[i]] = arg
        bound_kwargs.update(kwargs)
        input_data = self.input_model(**bound_kwargs)
        return await fn(**input_data.model_dump())

    @property
    def is_async(self) -> bool:
        return True

    def to_tool_call(self, *args, tool_id: str, **kwargs) -> 'ToolUse':
        """Create a ToolUse instance with the given arguments."""
        fn = get_tool_function(self.name)
        sig = inspect.signature(fn)
        param_names = list(sig.parameters)
        
        # Bind positional args to parameter names
        bound_kwargs = {}
        for i, arg in enumerate(args):
            if i >= len(param_names):
                raise TypeError(f"Too many positional arguments for tool '{self.name}'")
            bound_kwargs[param_names[i]] = arg
        
        # Add keyword args (allowing override)
        bound_kwargs.update(kwargs)
        
        # Validate with input_model
        input_data = self.input_model(**bound_kwargs)
        
        return ToolUse(
            tool_id=tool_id,
            option=self,
            inputs=input_data
        )


class Tool(BaseTool):
    """A definition of a sync tool that can be called by the LLM."""
    def __call__(self, *args, **kwargs):
        fn = get_tool_function(self.name)
        sig = inspect.signature(fn)
        param_names = list(sig.parameters)
        bound_kwargs = {}
        for i, arg in enumerate(args):
            if i >= len(param_names):
                raise TypeError(f"Too many positional arguments for tool '{self.name}'")
            bound_kwargs[param_names[i]] = arg
        bound_kwargs.update(kwargs)
        input_data = self.input_model(**bound_kwargs)
        return fn(**input_data.model_dump())

    @property
    def is_async(self) -> bool:
        return False

    def to_tool_call(self, *args, tool_id: str, **kwargs) -> 'ToolUse':
        """Create a ToolUse instance with the given arguments."""
        fn = get_tool_function(self.name)
        sig = inspect.signature(fn)
        param_names = list(sig.parameters)
        
        # Bind positional args to parameter names
        bound_kwargs = {}
        for i, arg in enumerate(args):
            if i >= len(param_names):
                raise TypeError(f"Too many positional arguments for tool '{self.name}'")
            bound_kwargs[param_names[i]] = arg
        
        # Add keyword args (allowing override)
        bound_kwargs.update(kwargs)
        
        # Validate with input_model
        input_data = self.input_model(**bound_kwargs)
        
        return ToolUse(
            tool_id=tool_id,
            option=self,
            inputs=input_data
        )


def register_tools(*tools) -> typing.List[BaseTool]:
    """Register multiple tool functions and create tool definitions.

    Args:
        *tools: Functions to register as tools

    Returns:
        typing.List[BaseTool]: List of tool definitions
    """
    return [register_tool(tool) for tool in tools]


class ToolUse(pydantic.BaseModel):
    """A base class for ToolUse and AsyncToolCall
    """
    tool_id: str
    id: str | None = pydantic.Field(
        default=None, description="The id of the tool specified by the LLM"
    )
    option: BaseTool = pydantic.Field(
        description="The tool that was chosen."
    )
    inputs: pydantic.BaseModel = pydantic.Field(
        description="The inputs to use for calling the ToolDef"
    )
    result: typing.Any = None
    option_text: str | None = None
    executed: bool = pydantic.Field(
        default=False, 
        description="Set True when the tool has been executed"
    )

    @property
    def is_async(self) -> bool:
        return self.option.is_async

    def __call__(self) -> typing.Any:
        """Execute the tool call."""
        return self.forward()

    async def aforward(self) -> typing.Any:
        fn = get_tool_function(self.option.name)
        data = self.inputs.model_dump()
        if self.option.is_async is False:
            result = fn(**data)
        else:
            result = await fn(**data)
        
        self.result = result
        self.executed = True
        return result

    def forward(self) -> typing.Any:
        data = self.inputs.model_dump()

        if self.is_async:
            raise RuntimeError(
                "ToolCall.forward() called on an async tool. Use aforward() instead."
            )
        fn = get_tool_function(self.option.name)
        result = fn(**data)
        self.result = result
        self.executed = True
        return result


class ToolError(BaseModel):
    """
    Normalized tool error information. Preserve a compact provider blob for audits.

    Attributes:
        kind: Short category ('timeout', 'navigation', 'validation', 'permission', 'provider', ...).
        message: Human-readable summary.
        provider_raw: Optional compact provider payload.
    """
    kind: str
    message: str
    provider_raw: t.Optional[t.Dict[str, t.Any]] = None


class ComputerUse(BaseModel):
    """
    Small, structured telemetry for computer-use tools (not sent directly to providers).

    Attributes:
        action: Optional action, e.g. 'open_url', 'click', 'type', 'scroll'.
        log: Short progress note.
        ocr_text: Optional OCR snippet tied to the latest screenshot.
        focus: Optional element/region description.
        extras: Additional small fields specific to the action.
    """
    action: t.Optional[str] = None
    log: t.Optional[str] = None
    ocr_text: t.Optional[str] = None
    focus: t.Optional[t.Dict[str, t.Any]] = None
    extras: t.Dict[str, t.Any] = Field(default_factory=dict)


@pydantic_dataclass
class ToolChunk:
    """
    A single partial update for one tool call.

    The adapter supplies *whatever it currently knows*.
    You don't have to fill everything each time.
    Call with done=True exactly once for that call.
    """
    # routing/position (let your adapter choose a stable combo)
    id: Optional[str] = Field(default=None, description="Optional provider-specific ID for this tool call")
    turn_index: Optional[int] = Field(default=None, description="Assistant message/content-block index")
    call_index: Optional[int] = Field(default=None, description="Per-message index (for streaming deltas)")

    # core info
    name: Optional[str] = Field(default=None, description="Tool/function name")
    args_text_delta: Optional[str] = Field(default=None, description="Append-only JSON text fragment")
    args_kv_patch: Optional[dict] = Field(default=None, description="Optional: shallow-merge patch")

    # lifecycle
    done: bool = Field(default=False, description="Set True when the provider signals completion")
    
    # error handling
    error: Optional[str] = Field(default=None, description="Error message if tool call failed")
    error_type: Optional[str] = Field(default=None, description="Error category (e.g., 'json_invalid', 'timeout', 'unknown_tool')")
    
    # extensibility 
    metadata: Dict[str, typing.Any] = Field(default_factory=dict, description="Provider-specific metadata")


# ToolBuffer implementation complete with:
# - Parallel tool call support via key-based routing
# - JSON text fragment and key-value patch accumulation  
# - Error handling and recovery
# - Proper lifecycle management

@pydantic_dataclass
class ToolBuffer:
    """
    Buffer for accumulating tool call chunks and managing their lifecycle.
    """

    tools: InitVar[typing.List[BaseTool]] = None
    _calls: typing.List[ToolUse] = Field(default_factory=list)
    _tool_map: Dict[str, BaseTool] = Field(default_factory=dict)
    _chunks: typing.List[ToolChunk] = Field(default_factory=list)

    def __post_init__(self, tools: typing.List[BaseTool]):
        self._acc: Dict[typing.Tuple[Optional[str], Optional[int], Optional[int]], dict] = {}
        self._calls = []
        self._tool_map = {
            tool.name: tool
            for tool in tools or []
        }
        self._chunks = []

    def append(self, chunk: ToolChunk) -> bool:
        """
        Add one chunk. Returns True if a tool call has just completed.

        Contract:
          - if you have a provider id, pass it every time for that call
          - else use (turn_index, call_index) as a stable pair
          - call once with done=True for that call (after the last delta)
        """
        key = self._make_key(chunk)
        acc = self._acc.get(key)
        if acc is None:
            acc = {
                "id": None,
                "name": None,
                "text": [],     # list[str] of JSON text fragments
                "kv": {},       # shallow key->value patches (optional)
                "turn": chunk.turn_index,
                "call": chunk.call_index,
                "done": False,
            }
            self._acc[key] = acc

        # update core fields (first non-empty wins)
        if chunk.id:
            acc["id"] = chunk.id
        if chunk.name:
            acc["name"] = chunk.name

        # accumulate arguments
        if chunk.args_text_delta:
            acc["text"].append(chunk.args_text_delta)
        if chunk.args_kv_patch:
            for k, v in chunk.args_kv_patch.items():
                acc["kv"][k] = v

        # handle error chunks
        if chunk.error:
            acc["error"] = chunk.error
            acc["error_type"] = chunk.error_type
            acc["done"] = True  # Errors complete the tool call
        
        # mark done if this is the final chunk for that call
        if chunk.done:
            acc["done"] = True

        # finalize if complete - id is optional for index-based routing
        if acc["done"] and acc["name"]:
            tool_name = acc["name"]
            tool_id = acc["id"] or f"tool_{key[1]}_{key[2]}" if key[1] is not None and key[2] is not None else "tool_call"

            # Check for errors first
            if acc.get("error"):
                # Tool call failed - still create ToolUse but mark it as failed
                # For now, we'll raise an exception. Could be configurable behavior.
                error_msg = acc.get("error", "Unknown error")
                error_type = acc.get("error_type", "unknown")
                raise RuntimeError(f"Tool call '{tool_name}' failed ({error_type}): {error_msg}")

            # prefer text fragments if any; otherwise serialize kv
            if acc["text"]:
                args_json = "".join(acc["text"])
                args, parse_success = self._try_parse_json(args_json)
                if not parse_success:
                    # Handle JSON parse failure - could log warning or raise error
                    # For now, we'll use empty dict and continue
                    args = {}
            else:
                args = acc["kv"]

            # Find tool definition in tool map
            tool_def = self._tool_map.get(tool_name)
            if tool_def is None:
                raise KeyError(f"ToolBuffer: unknown tool '{tool_name}'")

            # Create ToolUse instance
            try:
                tool_call = ToolUse(
                    tool_id=tool_id,
                    option=tool_def,
                    inputs=tool_def.input_model(**args)
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create ToolUse for '{tool_name}': {e}") from e

            # Clear only this call's chunks and add completed tool call
            # Note: Don't clear all chunks, just remove this accumulator
            key = self._make_key(chunk)
            if key in self._acc:
                del self._acc[key]
            
            self._calls.append(tool_call)
            # record completion and clear accumulator
            # self._done.append(call)
            return True

        return False

    def _make_key(self, c: ToolChunk) -> typing.Tuple[Optional[str], Optional[int], Optional[int]]:
        """
        Prefer provider id if present; otherwise (turn_index, call_index).
        This keeps keys stable across deltas without adding new types.
        """
        if c.id:
            return (c.id, None, None)
        return (None, c.turn_index, c.call_index)
    
    def _try_parse_json(self, json_text: str) -> typing.Tuple[dict, bool]:
        """Try to parse JSON, with basic repair if needed."""
        try:
            return json.loads(json_text), True
        except json.JSONDecodeError:
            # Try basic JSON repair - add missing closing braces/brackets
            repaired = self._repair_json(json_text)
            try:
                return json.loads(repaired), True
            except json.JSONDecodeError:
                return {}, False
    
    def _repair_json(self, json_text: str) -> str:
        """Basic JSON repair for common streaming issues."""
        if not json_text.strip():
            return "{}"
        
        # Count braces and brackets to add missing closers
        text = json_text.strip()
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # Add missing closers
        repaired = text
        repaired += '}' * open_braces
        repaired += ']' * open_brackets
        
        return repaired




### TODO: Archive the following

    # def to_tool_call(self, *args, tool_id: str, **kwargs):
    #     sig = inspect.signature(self.fn)
    #     param_names = list(sig.parameters)

    #     # Bind positional args to parameter names
    #     bound_kwargs = {}
    #     for i, arg in enumerate(args):
    #         if i >= len(param_names):
    #             raise TypeError(f"Too many positional arguments for tool '{self.name}'")
    #         bound_kwargs[param_names[i]] = arg

    #     # Add keyword args (allowing override, like normal Python function call)
    #     bound_kwargs.update(kwargs)

    #     # Validate with input_model
    #     input_data = self.input_model(**bound_kwargs)

    #     if self.is_async():
    #         return AsyncToolCall(
    #             tool_id=tool_id,
    #             option=self, 
    #             inputs=input_data
    #         )
    #     return ToolCall(
    #         tool_id=tool_id,
    #         option=self, 
    #         inputs=input_data
    #     )

    # convenience: tool_def(...)
    # __call__ = to_tool_call



# class ToolCall(pydantic.BaseModel):
#     """A response from the LLM that a tool was called
#     """

#     def __call__(self, store: bool=False):
#         data = self.inputs.model_dump()
#         # remaining keys are normal named parameters
#         result = self.option.fn(**data)
#         if store:
#             self.result = result
#         return result


# class AsyncToolCall(pydantic.BaseModel):
#     """A response from the LLM that a tool was called
#     """
#     tool_id: str
#     id: str | None = pydantic.Field(
#         default=None, description="Additional identifier for tool location when ToolDef is unknown"
#     )
#     option: BaseTool = pydantic.Field(
#         description="The tool that was chosen."
#     )
#     inputs: pydantic.BaseModel = pydantic.Field(
#         description="The inputs to use for calling the ToolDef"
#     )
#     result: typing.Any = None
#     option_text: str | None = None
#     executed: bool = pydantic.Field(default=False, description="Set True when the tool has been executed")
    
#     async def __call__(self, store: bool=False) -> typing.Any:
#         """Call the tool 

#         Raises:
#             NotImplementedError: If the function is a generator

#         Returns:
#             typing.Any: The result of the call
#         """
#         data = self.inputs.model_dump()
#         result = await self.option.fn(**data)
#         if store:
#             self.result = result
#         return result



# def make_tool_defs(*tools) -> typing.List[BaseToolDef]:
#     """Make multiple tools

#     Returns:
#         typing.List[ToolDef]: 
#     """
#     return list(
#         make_tool_def(tool)
#         for tool in tools
#     )


# llm(..., tool)
# LLM(tools)
# LLM(x.tool, y.tool, 

# Option 1) 

# class Tool(pydantic.BaseModel):
#     """A model representing a synchronous tool."""

#     name: str

#     @property
#     def tool_def(self) -> BaseToolDef | None:
#         return tool_registry.get(self.name)

#     @property
#     def is_async(self) -> bool:
#         return False

#     def __call__(self, *args, **kwargs):
#         """Call the synchronous tool.

#         Raises:
#             ValueError: If the tool is unknown.

#         Returns:
#             The result of the tool call.
#         """

#         tool_def = tool_registry.get(self.name)
#         if not tool_def:
#             raise ValueError(f"Unknown tool: {self.name}")

#         return tool_def(*args, **kwargs)


# class AsyncTool(pydantic.BaseModel):
#     """A model representing an asynchronous tool."""

#     name: str

#     @property
#     def tool_def(self) -> BaseToolDef | None:
#         return tool_registry.get(self.name)

#     @property
#     def is_async(self) -> bool:
#         return True

#     async def __call__(self, *args, **kwargs):
#         """Call the asynchronous tool.

#         Raises:
#             ValueError: If the tool is unknown.
#         Returns:
#             The result of the tool call.
#         """

#         tool_def = tool_registry.get(self.name)
#         if not tool_def:
#             raise ValueError(f"Unknown tool: {self.name}")

#         return await tool_def(*args, **kwargs)



# class ToolOut(pydantic.BaseModel):
#     """A response from the LLM that a tool was called
#     """
#     tool_call_id: str = pydantic.Field(
#         description="The ID linking this output to the original tool call"
#     )
#     option: BaseTool = pydantic.Field(
#         description="The tool that was chosen."
#     )
#     result: typing.Any = pydantic.Field(
#         description="The output of the tool"
#     )


# class ToolCall(pydantic.BaseModel):
#     """A response from the LLM that a tool was called
#     """
#     tool_id: str
#     id: str | None = pydantic.Field(
#         default=None, description="Additional identifier for tool location when ToolDef is unknown"
#     )
#     option: ToolDef = pydantic.Field(
#         description="The tool that was chosen."
#     )
#     inputs: pydantic.BaseModel = pydantic.Field(
#         description="The inputs to use for calling the ToolDef"
#     )
#     result: typing.Any = None
#     option_text: str | None = None

#     def __call__(self, store: bool=False):
#         data = self.inputs.model_dump()
#         # remaining keys are normal named parameters
#         result = self.option.fn(**data)
#         if store:
#             self.result = result
#         return result


# def to_openai_tool(tool: ToolDef | list[ToolDef]) -> list[dict]:
#     if not isinstance(tool, list):
#         tool = [tool]

#     tools = []
#     for t in tool:
#         schema = (
#             t.input_model.model_json_schema()
#             if IS_V2 else
#             t.input_model.schema()
#         )

#         tools.append({
#             "type": "function",
#             "function": {
#                 "name": t.name,
#                 "description": t.description,
#                 "parameters": schema
#             }
#         })
#     return tools

# class ToolOption(pydantic.BaseModel):
#     """
#     Represents an option for a tool, encapsulating the tool's name, 
#     the function to be executed, and any additional keyword arguments.
#     Attributes:
#         name (str): The name of the tool.
#         f (typing.Callable[[typing.Any], typing.Any]): The function to be executed by the tool.
#         kwargs (typing.Dict): A dictionary of additional keyword arguments to be passed to the function.
#     """
#     name: str
#     f: typing.Callable[
#         [typing.Any], typing.Any
#     ]
#     kwargs: typing.Dict

#     def to_input(self) -> typing.Dict:
#         """
#         Converts the instance's keyword arguments into a dictionary of arguments.
#         Returns:
#             dict: A dictionary containing the keyword arguments of the instance.
#         """
#         return {
#             **self.kwargs
#         }

# class ToolBuilder(object):
#     """Use to build up a tool from the 
#     delta values
#     """
#     def __init__(self):
#         """
#         """
#         self._index = None
#         self._name = ''
#         self._args = ''
#         self._tools = []

#     def update(self, id, index, name, args, **kwargs) -> 'ToolCall':        
#         """

#         Args:
#             id : The id of the 
#             index : 
#             name : 
#             args : 

#         Returns:
#             ToolCall: 
#         """
        
#         if index != self._index:
#             if self._index is not None:
#                 result = ToolCall(
#                     tool_id=id,
#                     option=self.tools[self._name],
#                     args=json.loads(self._args)
#                 )
#                 self._tools.append(result)
#                 return result
#             self._index = index
#             self._name = name
#             self._args = args
#             return None
#             # return {
#             #     'name': self._name,
#             #     'args': self._args
#             # }
#         self._args += args
#         return None

