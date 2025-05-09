# 1st party
import typing
import json
from dataclasses import dataclass
from typing import Callable, Type, Any, Optional
from typing import Any, Dict, get_type_hints
from inspect import signature, Parameter

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


@dataclass
class ToolDef:
    name: str
    description: str
    fn: Callable
    input_model: Type[BaseModel]
    return_type: Optional[Type[Any]] = None  # for future use
    version: Optional[str] = None            # optional metadata


IS_V2 = int(pydantic.__version__.split(".")[0]) >= 2

def make_tool_def(func: Callable) -> ToolDef:
    sig = signature(func)
    hints = get_type_hints(func)
    fields: Dict[str, tuple[type, Any]] = {}

    for name, param in sig.parameters.items():
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        typ = hints.get(name) or (
            type(param.default) if param.default is not Parameter.empty else Any
        )
        default = param.default if param.default is not Parameter.empty else ...
        fields[name] = (typ, default)

    input_model = create_model(
        f"{func.__name__.title()}Inputs",
        **fields,
        __base__=BaseModel
    )

    return ToolDef(
        name=func.__name__,
        description=(func.__doc__ or "").strip() or f"Tool for {func.__name__}",
        fn=func,
        input_model=input_model,
        return_type=hints.get("return", None)
    )


class ToolOption(pydantic.BaseModel):
    """
    Represents an option for a tool, encapsulating the tool's name, 
    the function to be executed, and any additional keyword arguments.
    Attributes:
        name (str): The name of the tool.
        f (typing.Callable[[typing.Any], typing.Any]): The function to be executed by the tool.
        kwargs (typing.Dict): A dictionary of additional keyword arguments to be passed to the function.
    """
    name: str
    f: typing.Callable[
        [typing.Any], typing.Any
    ]
    kwargs: typing.Dict

    def to_input(self) -> typing.Dict:
        """
        Converts the instance's keyword arguments into a dictionary of arguments.
        Returns:
            dict: A dictionary containing the keyword arguments of the instance.
        """
        return {
            **self.kwargs
        }


class ToolCall(
    AsyncModule, Module,
    pydantic.BaseModel
):
    """A response from the LLM that a tool was called
    """
    option: ToolOption = pydantic.Field(
        description="The tool that was chosen."
    )
    args: typing.Dict[str, typing.Any] = pydantic.Field(
        description="The arguments to the tool."
    )

    def forward(self) -> typing.Any:
        """Call the tool

        Raises:
            NotImplementedError: If the function is async
            NotImplementedError: If the function is a generator function

        Returns:
            typing.Any: The result of the call
        """
        # Check if valid to use with forward
        if is_async_function(self.option.f):
            raise NotImplementedError
        # if is_generator_function(self.option.f):
        #     raise NotImplementedError
        return self.option.f(**self.args)

    async def aforward(self) -> typing.Any:
        """Call the tool 

        Raises:
            NotImplementedError: If the function is a generator

        Returns:
            typing.Any: The result of the call
        """
        if not is_async_function(self.option.f):
            raise NotImplementedError
            # return await self.option.f(**self.args)
        if is_generator_function(self.option.f):
            raise NotImplementedError
        return await self.option.f(**self.args)
    
    @property
    def is_async(self) -> bool:
        return is_async_function(self.option.f)

    # def stream(self) -> typing.Iterator:
    #     """Stream the tool

    #     Raises:
    #         NotImplementedError: The result

    #     Yields:
    #         Iterator[typing.Iterator]: The result of the call
    #     """
    #     if is_async_function(self.option.f):
    #         raise NotImplementedError
    #     elif is_generator_function(self.option.f):
    #         for k in self.option.f(**self.args):
    #             yield k
    #     else:
    #         yield self.option.f(**self.args)
        
    # async def astream(self):
    #     """Stream the tool

    #     Yields:
    #         Iterator[typing.Iterator]: The result of the call
    #     """
    #     if is_generator_function(
    #         self.option.f
    #     ) and is_async_function(self.option.f):
    #         async for k in await self.option.f(**self.args):
    #             yield k
    #     elif is_generator_function(self.option.f):
    #         for k in await self.option.f(**self.args):
    #             yield k
    #     elif is_async_function(self.option.f):
    #         yield await self.option.f(**self.args)
    #     else:
    #         yield self.option.f(**self.args)


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


# class ToolSet(object):
#     """A set of tools that the LLM can use
#     """
#     def __init__(self, tools: typing.List[ToolOption], **kwargs):
#         """The set of tools

#         Args:
#             tools (typing.List[ToolOption]): The list of tools
#         """
#         self.tools = {
#             tool.name: tool
#             for tool in tools
#         }
#         self.kwargs = kwargs

#     def add(self, option: ToolOption):
#         """Add a tool to the set

#         Args:
#             option (ToolOption): The option to add
#         """
#         self.tools[option.name] = option

#     def remove(self, option: ToolOption):
#         """Remove a tool from the tool set

#         Args:
#             option (ToolOption): The option to add
#         """
#         del self.tools[option.name]

#     def to_input(self):
#         return list(
#             tool.to_input() for _, tool in self.tools.items()
#         )
    
#     def __len__(self) -> int:
#         return len(self.tools)

#     def __iter__(self) -> typing.Iterator:
#         """
#         Returns an iterator over the tools in the collection.
#         Yields:
#             tool: Each tool in the collection.
#         """

#         for _, tool in self.tools.items():
#             yield tool

#     def __getitem__(self, name):
#         """
#         Retrieve a tool by its name.
#         Args:
#             name (str): The name of the tool to retrieve.
#         Returns:
#             object: The tool associated with the given name.
#         Raises:
#             KeyError: If the tool with the specified name does not exist.
#         """
#         return self.tools[name]

