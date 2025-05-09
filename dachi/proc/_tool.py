# 1st party
import typing
import json
from dataclasses import dataclass
from typing import Callable, Type, Any, Optional
from typing import Any, Dict, get_type_hints
from inspect import signature, Parameter
import inspect
# 3rd party
import pydantic
from pydantic import create_model
from pydantic import BaseModel
from ..proc import AsyncModule, Module, StreamModule, AsyncStreamModule

# local
from ..utils import (
    is_async_function,
    is_generator_function,
)

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


class ToolDef(pydantic.BaseModel):

    name: str
    description: str
    fn: Callable
    input_model: Type[BaseModel]
    return_type: Optional[Type[Any]] = None  # for future use
    version: Optional[str] = None            # optional metadata
    
    def to_tool_call(self, *args, **kwargs):
        sig = inspect.signature(self.fn)
        param_names = list(sig.parameters)

        # Bind positional args to parameter names
        bound_kwargs = {}
        for i, arg in enumerate(args):
            if i >= len(param_names):
                raise TypeError(f"Too many positional arguments for tool '{self.name}'")
            bound_kwargs[param_names[i]] = arg

        # Add keyword args (allowing override, like normal Python function call)
        bound_kwargs.update(kwargs)

        # Validate with input_model
        input_data = self.input_model(**bound_kwargs)

        if self.is_async():
            return AsyncToolCall(option=self, inputs=input_data)
        return ToolCall(option=self, inputs=input_data)

    # convenience: tool_def(...)
    __call__ = to_tool_call
    
    def is_async(self) -> bool:

        return is_async_function(self.fn)
    
    # def to_tool_call(self, *args, **kwargs) -> typing.Union['ToolCall', 'AsyncToolCall']:

    #     print(args, kwargs)
    #     bound_args = signature(self.fn).bind_partial(*args, **kwargs)
    #     bound_args.apply_defaults()
    #     input_data = self.input_model(**bound_args.arguments)
    #     print(input_data)
    #     if self.is_async():
    #         return AsyncToolCall(
    #             option=self,
    #             inputs=input_data
    #         )
    #     return ToolCall(
    #         option=self, inputs=input_data
    #     )

    # def to_tool_call(self, *args, **kwargs):
    #     fn_sig = signature(self.fn)
    #     params = list(fn_sig.parameters.values())

    #     input_dict = {}
    #     named_param_names = [p.name for p in params if p.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)]

    #     # First, match named params with args or kwargs
    #     for i, name in enumerate(named_param_names):
    #         if i < len(args):
    #             input_dict[name] = args[i]
    #         elif name in kwargs:
    #             input_dict[name] = kwargs[name]

    #     # Remaining positional args go to _args
    #     num_consumed = len(input_dict)
    #     if any(p.kind == Parameter.VAR_POSITIONAL for p in params):
    #         input_dict["_args"] = list(args[num_consumed:])

    #     # Extra kwargs go to _kwargs
    #     if any(p.kind == Parameter.VAR_KEYWORD for p in params):
    #         remaining_keys = set(kwargs.keys()) - set(named_param_names)
    #         input_dict["_kwargs"] = {k: kwargs[k] for k in remaining_keys}

    #     input_data = self.input_model(**input_dict)

    #     if self.is_async():
    #         return AsyncToolCall(option=self, inputs=input_data)
    #     return ToolCall(option=self, inputs=input_data)

IS_V2 = int(pydantic.__version__.split(".")[0]) >= 2


# def make_tool_def(func: Callable) -> ToolDef:
#     sig = signature(func)
#     hints = get_type_hints(func)
#     fields: Dict[str, tuple[type, Any]] = {}

#     for name, param in sig.parameters.items():
#         if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
#             continue
#         typ = hints.get(name) or (
#             type(param.default) if param.default is not Parameter.empty else Any
#         )
#         default = param.default if param.default is not Parameter.empty else ...
#         fields[name] = (typ, default)

#     input_model = create_model(
#         f"{func.__name__.title()}Inputs",
#         **fields,
#         __base__=BaseModel
#     )

#     return ToolDef(
#         name=func.__name__,
#         description=(func.__doc__ or "").strip() or f"Tool for {func.__name__}",
#         fn=func,
#         input_model=input_model,
#         return_type=hints.get("return", None)
#     )

from inspect import signature, Parameter
from typing import Any, Dict, Callable, get_type_hints
from pydantic import create_model, BaseModel
import pydantic

IS_V2 = int(pydantic.__version__.split(".")[0]) >= 2

def make_tool_def(func: Callable) -> ToolDef:
    sig = signature(func)
    hints = get_type_hints(func)
    fields: Dict[str, tuple[type, Any]] = {}

    for name, param in sig.parameters.items():
        if param.kind == Parameter.VAR_POSITIONAL:
            raise TypeError(f"Function {func.__name__} has *args (VAR_POSITIONAL), which is not supported for tool use.")
        if param.kind == Parameter.VAR_KEYWORD:
            raise TypeError(f"Function {func.__name__} has **kwargs (VAR_KEYWORD), which is not supported for tool use.")

        typ = hints.get(name) or (
            type(param.default) if param.default is not Parameter.empty else Any
        )
        default = param.default if param.default is not Parameter.empty else ...
        fields[name] = (typ, default)

    InputModel = create_model(
        f"{func.__name__.title()}Inputs", **fields, __base__=BaseModel
    )

    return ToolDef(
        name=func.__name__,
        description=(func.__doc__ or "").strip() or f"Tool for {func.__name__}",
        fn=func,
        input_model=InputModel,
        return_type=hints.get("return", None)
    )


class ToolCall(
    pydantic.BaseModel,
    Module
):
    """A response from the LLM that a tool was called
    """
    option: ToolDef = pydantic.Field(
        description="The tool that was chosen."
    )
    inputs: pydantic.BaseModel = pydantic.Field(
        description="The inputs to use for calling the ToolDef"
    )

    def forward(self):
        data = self.inputs.model_dump() if IS_V2 else self.inputs.dict()
        # remaining keys are normal named parameters
        return self.option.fn(**data)


class AsyncToolCall(
    pydantic.BaseModel,
    AsyncModule
):
    """A response from the LLM that a tool was called
    """
    option: ToolDef = pydantic.Field(
        description="The tool that was chosen."
    )
    inputs: pydantic.BaseModel = pydantic.Field(
        description="The inputs to use for calling the ToolDef"
    )

    async def aforward(self) -> typing.Any:
        """Call the tool 

        Raises:
            NotImplementedError: If the function is a generator

        Returns:
            typing.Any: The result of the call
        """
        data = self.inputs.model_dump() if IS_V2 else self.inputs.dict()
        return await self.option.fn(**data)


class ToolBuilder(object):

    def __init__(self):
        
        self._index = None
        self._name = ''
        self._args = ''
        self._tools = []

    def update(self, index, name, args):        
        
        if index != self._index:
            if self._index is not None:
                result = ToolCall(
                    option=self.tools[self._name],
                    args=json.loads(self._args)
                )
                self._tools.append(result)
            self._index = index
            self._name = name
            self._args = args
            return {
                'name': self._name,
                'args': self._args
            }
        self._args += args
        return None



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
