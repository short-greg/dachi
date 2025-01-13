import typing
from abc import abstractmethod

import pydantic
from .. import _core as core
import pydantic

# class ToolParam(pydantic.BaseModel, core.Renderable):
#     name: str
#     type_: str
#     descr: str = ''
#     required: typing.List[str] = Field(default_factory=list)
#     enum: typing.Optional[typing.List[typing.Any]] = None
#     minimum: typing.Optional[float] = None
#     maximum: typing.Optional[float] = None
#     minLength: typing.Optional[int] = None
#     maxLength: typing.Optional[int] = None
#     default: typing.Optional[typing.Any] = None
#     format: typing.Optional[str] = None

#     def render(self) -> str:
#         return core.render(self)
    
#     # def clone(self) -> 'ToolParam':
#     #     pass

# class ToolObjParam(ToolParam):
#     params: typing.List[ToolParam] = Field(default_factory=list)


# class ToolArrayParam(ToolParam):
#     items: typing.List[ToolParam]

#     def __init__(self, name: str, items: typing.List[ToolParam], descr: str='', 
#             **kwargs):
#         """Create an array of tools

#         Args:
#             name (str): The name of the array
#             items (typing.List[ToolParam]): The items in the array
#             descr (str, optional): The description. Defaults to ''.
#         """
#         super().__init__(
#             name=name, type_="array", items=items, descr=descr, **kwargs
#         )
# 'name': name,
# 'type_': type_,
# 'param': param,
# 'descr': descr,
# 'required': required,
# 'strict': strict


# class Tool(core.Module, pydantic.BaseModel):
#     """Tool for the agent to excute. It stores a function and
#     allows the 
#     """
#     name: str

#     @abstractmethod
#     def forward(self, *args, **kwargs):
#         # Execute the tool
#         pass

#     @abstractmethod
#     def todict(self) -> typing.Dict:
#         pass


# class FuncTool(core.Module, pydantic.BaseModel):
#     """Tool for the agent to execute. It stores a function for an 
#     agent to execute along with all of the information on the tool.
#     """
#     name: str
#     descr: str = None
#     parameters: typing.Dict[str, typing.Any] = None
#     function: typing.Callable
#     options: typing.Dict = None

#     def __init__(self, **data):
#         """
#         Initialize the tool with the provided data.
#         Args:
#             **data: Arbitrary keyword arguments containing the tool's attributes.
#         Keyword Args:
#             name (str): The name of the tool.
#             description (str, optional): A brief description of the tool.
#             parameters (dict, optional): Parameters required by the tool.
#             function (callable): The function associated with the tool.
#         """
#         super().__init__(**data)
#         self.name = data['name']
#         self.descr = data.get('description')
#         self.parameters = data.get('parameters')
#         self.function = data['function']

#     def forward(self, *args, **kwargs):
#         """
#         Executes the stored function with the provided arguments and keyword arguments.
#         Args:
#             *args: Variable length argument list to pass to the function.
#             **kwargs: Arbitrary keyword arguments to pass to the function.
#         Returns:
#             The result of the function execution.
#         """
#         return self.function(*args, **kwargs)
    
#     @abstractmethod
#     def todict(self) -> typing.Dict:
#         """
#         Converts the tool to a dictionary.
#         Returns:
#             dict: A dictionary representation of the tool.
#         """
#         pass
#         # return {
#         #     'name': self.name,
#         #     'description': self.descr,
#         #     'parameters': self.parameters,
#         #     'function': self.function,
#         #     **self.options
#         # }


# class ToolSet(pydantic.BaseModel, core.Module):
#     """A set of tools for an agent to use"""
    
#     name: str
#     tools: typing.Dict[str, Tool] = pydantic.Field(
#         default_factory=dict, description="A dictionary of tools to use for an agent."
#     )
#     options: typing.Dict = pydantic.Field(default_factory=dict, description="Options for the tool set.")

#     def __init__(self, tools: typing.List[Tool] = None):
#         """

#         Args:
#             tools (typing.List[Tool], optional): . Defaults to None.
#         """
#         super().__init__()
#         self.tools = {
#             tool.name: tool for tool in tools
#         } if tools else {}

#     def add(self, tool: Tool):
#         """
#         Adds a tool to the ToolSet.
#         Args:
#             tool (Tool): The tool to add to the ToolSet.
#         """
#         self.tools[tool.name] = tool

#     def remove(self, name: str):
#         """
#         Removes a tool from the ToolSet.
#         Args:
#             name (str): The name of the tool to remove.
#         """
#         self.tools.pop(name)

#     def todict(self) -> typing.Dict:
#         """
#         Converts the ToolSet to a dictionary.
#         Returns:
#             dict: A dictionary representation of the ToolSet.
#         """
#         return {
#             'name': self.name,
#             'tools': {
#                 name: tool.todict() for name, tool in self.tools.items()
#             },
#             **self.options
#         }
    
    
#     def forward(self, name: str, *args, **kwargs):
#         """
#         Executes the forward method of a tool by its name.
#         Args:
#             name (str): The name of the tool to execute.
#             *args: Variable length argument list to pass to the tool's forward method.
#             **kwargs: Arbitrary keyword arguments to pass to the tool's forward method.
#         Returns:
#             The result of the tool's forward method.
#         Raises:
#             ValueError: If the tool with the specified name is not found in the ToolSet.
#         """
#         if name in self.tools:
#             return self.tools[name].forward(*args, **kwargs)
#         else:
#             raise ValueError(
#                 f"Tool with name '{name}' not found in ToolSet."
#             )
    
#     def __iter__(self) -> typing.Iterator[typing.Tuple[str, Tool]]:
#         """
#         Returns an iterator over the tools in the collection.
#         Yields:
#             Iterator[Tuple[str, Tool]]: An iterator of tuples where each tuple contains
#             a string (the tool name) and a Tool object.
#         """
#         return iter(self.tools.items())
