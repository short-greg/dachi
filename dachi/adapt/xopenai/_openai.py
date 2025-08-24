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
    ToolDef, BaseModule, AIAdapt, OpenAIChat
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



def to_openai_tool(tool: ToolDef | list[ToolDef]) -> list[dict]:
    """Converts a tool definition or a list of tool definitions to OpenAI format.

    Args:
        tool (ToolDef | list[ToolDef]): The tool definition(s) to convert.

    Returns:
        list[dict]: A list of dictionaries representing the OpenAI tools.
    """
    if isinstance(tool, ToolDef):
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

    tools: typing.List[ToolDef] | None = None
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
        tools: typing.Iterable[ToolDef]=UNDEFINED,
        json_output: bool | pydantic.BaseModel | typing.Dict=UNDEFINED,
        procs: typing.List[RespProc]=UNDEFINED
    ):
        # TODO: Implement
        pass


class ChatCompletion(LLM, Process, AsyncProcess, StreamProcess, AsyncStreamProcess):
    """
    OpenAIChatComp is an adapter for the OpenAI Chat Completions API. It provides methods for 
    interacting with the API, including synchronous and asynchronous message forwarding, 
    streaming, and spawning new instances with modified configurations.
    """
    proc: InitVar[Sequential | RespProc | None] = None

    def __post_init__(self, api_key, client_kwargs, kwargs, proc: typing.List[RespProc]):
        """
        Initializes the OpenAIChatComp instance with the provided tools and JSON output configuration.
        Args:
            procs (typing.List[RespProc], optional): A list of response processors to handle the output from the API.
                If not provided, defaults to an empty list.
        """
        super().__post_init__(api_key, client_kwargs, kwargs)
        if isinstance(proc, RespProc):
            proc = Sequential(items=[proc])
        if proc is None:
            proc = Sequential(items=[])
        self.proc = proc

    def spawn(
        self, 
        tools: typing.Iterable[ToolDef]=UNDEFINED,
        json_output: bool | pydantic.BaseModel | typing.Dict=UNDEFINED
    ):
        """
        Spawns a new OpenAIChatComp instance, updating all values that are defined.
        Args:
            tools (ToolSet, optional): The set of tools to be used. If not provided, 
                defaults to the instance's `_tools` attribute.
            json_output (bool | pydantic.BaseMode | typing.Dict, optional): Specifies 
                the JSON output configuration. If not provided, defaults to the 
                instance's `_json_output` attribute.
        Returns:
            OpenAIChatComp: A new instance of OpenAIChatComp with the updated values.
        """
        tools = coalesce(
            tools, self._tools
        )
        json_output = coalesce(json_output, self._json_output)
        return ChatCompletion(
            tools, json
        )
    
    def forward(
        self, 
        msg, 
        **kwargs
    )-> Resp:
        """
        Processes a message by forwarding it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            Resp: The response from the LLM.
        Notes:
            - This method uses the `llm_forward` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """
        
        kwargs = {
            **self.kwargs,
            **kwargs
        }
        return llm_forward(
            self._client.chat.completions.create, 
            msg,
            _adapt=self.adapt,
            _proc=self.proc,
            **kwargs
        )

    async def aforward(
        self, 
        msg, 
        **kwargs
    ) -> Resp:
        """
        Processes a message by asynchronously forwarding it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            Resp: The response from the LLM.
        Notes:
            - This method uses the `llm_aforward` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """

        kwargs = {
            **self.kwargs, 
            **kwargs
        }
        return await llm_aforward(
            self._aclient.chat.completions.create, 
            msg,
            _adapt=self.adapt,
            _proc=self.proc, 
            **kwargs
        )

    def stream(
        self, 
        msg, 
        **kwargs
    ) -> typing.Iterator[Resp]:
        """
        Processes a message by streaming it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            typing.Iterator[Resp]: An iterator of response objects from the LLM.
        Notes:
            - This method uses the `llm_stream` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """

        kwargs = {
            **self.kwargs, 
            **kwargs
        }
        for r in llm_stream(
            self._client.chat.completions.create, 
            msg,
            _adapt=self.adapt,
            _proc=self.proc, 
            stream=True,
            **kwargs
        ):
            yield r
    
    async def astream(
        self, 
        msg, 
        *args, 
        **kwargs
    ) -> typing.AsyncIterator[Resp]:
        """
        Processes a message by streaming it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            typing.AsyncIterator[Resp]: An async iterator of response objects from the LLM.
        Notes:
            - This method uses the `llm_astream` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """
        kwargs = {
            **self.kwargs, 
            **kwargs
        }
        async for r in await llm_astream(
            self._aclient.chat.completions.create, 
            msg,
            _adapt=self.adapt,
            _proc=self.proc,
            stream=True,
            **kwargs
        ):
            yield r


