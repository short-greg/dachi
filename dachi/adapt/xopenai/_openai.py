# 1st party 
from abc import ABC
import pkg_resources
import typing
import json
import pydantic
from dataclasses import InitVar
import openai
import inspect
from abc import abstractmethod

# 3rd party
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

# Local
from dachi.core import (
    Msg, to_list_input, Resp,
    BaseTool, BaseModule, AIAdapt, OpenAIChat
)
from dachi.proc import (
    llm_aforward, llm_astream, 
    llm_forward, llm_stream,
    Sequential, RespProc
)
from dachi.proc._resp import TextConv, StructConv, ParsedConv, ToolConv
from dachi.proc._out import ToolExecConv
from dachi.proc import Process, AsyncProcess, StreamProcess, AsyncStreamProcess
from dachi.core import ModuleList
from dachi.utils import UNDEFINED, coalesce

# TODO: add utility for this


if len(missing) > 0:
    raise RuntimeError(f'To use this module openai must be installed.')


class OpenAIAdapter(Process):
    """Use to convert response to OpenAI format
    """

    streamed: bool = False

    @abstractmethod
    def forward(self, resp: typing.Dict) -> typing.Dict:
        pass



def to_openai_tool(tool: BaseTool | list[BaseTool]) -> list[dict]:
    """Converts a tool definition or a list of tool definitions to OpenAI format.

    Args:
        tool (ToolDef | list[ToolDef]): The tool definition(s) to convert.

    Returns:
        list[dict]: A list of dictionaries representing the OpenAI tools.
    """
    if isinstance(tool, BaseTool):
        tool = [tool]

    tools = []
    for t in tool:
        schema = t.input_model.model_json_schema()

        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": schema
            }
        })
    return tools











class LLM(BaseModule):
    """LLM is a base class for language model adapters.
    It provides a structure for implementing various language model interactions,
    including synchronous and asynchronous message forwarding, streaming, and spawning new instances.
    """

    tools: typing.List[BaseTool] | None = None
    json_output: bool | pydantic.BaseModel | typing.Dict = False
    api_key: InitVar[str | None] = None
    client_kwargs: InitVar[typing.Dict | None] = None
    kwargs: InitVar[typing.Dict] = None
    message_arg: str = 'messages'
    adapt: AIAdapt | None = None

    def __post_init__(
        self, api_key, client_kwargs, kwargs
    ):
        """
        Args:
            tools (ToolSet, optional): . Defaults to None.
            json_output (bool | pydantic.BaseMode | typing.Dict, optional): . Defaults to False.
            api_key (str, optional): . Defaults to None.
            client_kwargs (typing.Dict, optional): . Defaults to None.
        """
        if client_kwargs is None:
            client_kwargs = {}

        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.convs = ModuleList(items=[])
        if self.json_output is False:
            self.convs.append(TextConv())
        elif isinstance(self.json_output, pydantic.BaseModel) or (
            inspect.isclass(self.json_output) and 
            issubclass(self.json_output, pydantic.BaseModel)
        ):
            self.convs.append(ParsedConv(struct=self.json_output))
        else:
            self.convs.append(StructConv(struct=self.json_output))
        if self.tools is not None:
            self.convs.append(ToolConv(tools=self.tools))
        self._client = openai.Client(
            api_key=api_key, **client_kwargs
        )
        self._aclient = openai.AsyncClient(
            api_key=api_key, **client_kwargs
        )
        
        if self.adapt is None:
            self.adapt = OpenAIChat()
        
    def spawn(self, 
        tools: typing.Iterable[BaseTool]=UNDEFINED,
        json_output: bool | pydantic.BaseModel | typing.Dict=UNDEFINED,
        procs: typing.List[RespProc]=UNDEFINED
    ):
        # TODO: Implement
        pass




