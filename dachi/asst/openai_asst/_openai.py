# 1st party 
from abc import ABC
import pkg_resources
import typing
import json
import pydantic

# 3rd party
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

# Local
from ...msg import (
    Msg, END_TOK, to_list_input,
    ToolDef, ToolBuilder, ToolCall
)
from ... import utils
from ...inst import render
from .._ai import (
    LLM, llm_aforward, llm_astream, llm_forward, llm_stream
)
from ...msg import RespConv, OutConv
from ...utils import UNDEFINED, coalesce
from ... import store

# TODO: add utility for this

import openai

if len(missing) > 0:
    raise RuntimeError(f'To use this module openai must be installed.')


def to_openai_tool(tool: ToolDef | list[ToolDef]) -> list[dict]:
    """

    Args:
        tool (ToolDef | list[ToolDef]): 

    Returns:
        list[dict]: 
    """
    if isinstance(tool, ToolDef):
        tool = [tool]

    tools = []
    for t in tool:
        schema = (
            t.input_model.model_json_schema()
            if utils.pydantic_v2() else
            t.input_model.schema()
        )

        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": schema
            }
        })
    return tools


class TextConv(RespConv):
    """
    OpenAITextProc is a class that processes an OpenAI response and extracts text outputs from it.
    """

    def __init__(
        self, 
        name: str='content', 
        from_ = 'response'
    ):
        super().__init__(name, from_)

    def post(
        self, 
        msg, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = False
    ):
        """

        Args:
            msg: The message to process
            result: 
            delta_store: 
            streamed (bool, optional): whether streamed or not. Defaults to False.
            is_last (bool, optional): the last. Defaults to False.
        """
        content = '' if msg.m['content'] in (None, UNDEFINED) else msg.m['content']
        store.acc(
            delta_store, 'all_content', content
        )
        msg['content'] = delta_store['all_content']
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        if streamed and is_last:
            return ''
        elif streamed:
            content = resp.choices[0].delta.content
        else:
            content = resp.choices[0].message.content

        return content

    def prep(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        return {}


class StructConv(RespConv):
    """
    OpenAITextProc is a class that processes an OpenAI response and extracts text outputs from it.
    """
    def __init__(
        self, 
        struct: pydantic.BaseModel | typing.Dict=None, name: str='content', 
        from_: str='response'
    ):
        super().__init__(name, from_)
        self._struct = struct

    def post(
        self, 
        msg, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = False
    ):
        if is_last:
            msg['content'] = delta_store['content']
        msg['content'] = ''

    def delta(
        self, resp, delta_store: typing.Dict, 
        streamed: bool=False, is_last: bool=False
    ):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        if streamed and resp is not END_TOK:
            delta = resp.choices[0].delta.content
            store.acc(
                delta_store, 'content', delta
            )
        elif not streamed:
            content = resp.choices[0].message.content
            store.acc(
                delta_store, 'content', content
            )

        if is_last:
            struct = json.loads(delta_store['content'])
            return struct

        return ''

    def prep(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: 
        """
        if isinstance(
            self._struct, 
            typing.Dict
        ):
            return {'response_format': {
                "type": "json_schema",
                "json_schema": self._struct
            }}
        elif isinstance(
            self._struct, pydantic.BaseModel
        ):
            return {
                'response_format': self._struct
            }
        return {
            'response_format': "json_object"
        }


# TODO: Update these converters

class StructStreamConv(StructConv):

    def __init__(self, struct = None, name = 'content', from_ = 'response'):
        super().__init__(struct, name, from_)
    
    def delta(
        self, resp, delta_store: typing.Dict, 
        streamed: bool=False, is_last: bool=False
    ):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        if resp.type == "content.delta":
            if resp.parsed is not None:
                print("Chunk received:", resp.parsed)
        elif resp.type == "content.done":
            return None
        elif resp.type == "error":
            raise RuntimeError(resp.error)
        
    def prep(self):
        return {
            "response_format": "json_object"
        }


class ParsedConv(StructConv):
    """For use with the "parse" API
    """

    def __init__(
        self, 
        struct: typing.Type[pydantic.BaseModel] = None, 
        name = 'content', from_ = 'response'
    ):
        super().__init__(struct=struct, name=name, from_=from_)
    
    def delta(
        self, resp, delta_store: typing.Dict, 
        streamed: bool=False, is_last: bool=False
    ):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        # if (message.refusal):
        #     raise RuntimeError(
        #         f"Refusal: {message.refusal}"
        #     )
        # return message.parsed
        if not is_last:
            return UNDEFINED
        print(resp.choices[0].message.content)
        delta_store["content"] = self._struct.model_validate_json(
            resp.choices[0].message.content
        )
        return delta_store["content"]
    
    def prep(self):
        
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self._struct.__name__,  # or a custom string identifier
                    "schema": self._struct.model_json_schema()
                }
            }
        }


class ToolConv(RespConv):
    """
    Process OpenAI LLM responses to extract tool calls.
    This class extends the RespProc class and is designed to handle responses from OpenAI's language model,
    specifically to extract and manage tool calls embedded within the responses.
    """
    def __init__(
        self, 
        tools: typing.Iterable[ToolDef], 
        name: str='tools', 
        from_: str='response',
        run_call: bool=False
    ):
        """
        Args:
            tools (typing.Dict[str, ToolOption]): 
        """
        super().__init__(name, from_)
        self.tools = {
            tool.name: tool
            for tool in tools
        }
        self.run_call = run_call

    def prep(self):
        """
        Returns:
            : 
        """
        return {
            'tools': to_openai_tool(
                self.tools.values()
            )
        }
    
    def post(
        self, 
        msg: Msg, 
        result, 
        delta_store, 
        streamed = False, 
        is_last = False
    ):
        if len(delta_store['tools']) != 0 and self.run_call: 
            
            msg['tool_calls'] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc.to_dict()
                for tc in    
                delta_store['tool_calls']
            ]

            for tool in delta_store['tools']:
                msg.follow_up.append(Msg(
                    role="tool",
                    content=render(tool.result),
                    tool_call_id=tool.tool_id,
                    name=tool.option.name
                ))

    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        streamed: bool=False, 
        is_last: bool=False
    ) -> typing.Any:
        """Convert the response to an output

        Args:
            resp : Update the tool call
            delta_store (typing.Dict): The state of the conversion
            streamed (bool, optional): Whether streamed or not. Defaults to False.
            is_last (bool, optional): Whether it is the last item in the stream. Defaults to False.

        Returns:
            typing.Any
        """
        store.get_or_set(
            delta_store, 'tool_calls', []
        )
        store.get_or_set(delta_store, 'tools', [])
        if streamed and resp is END_TOK:
            if 'delta' not in delta_store:
                return utils.UNDEFINED
            return delta_store[
                'delta'
            ]['builder'].tools
        
        if streamed:
            # TODO: Finish implementing streaming
            if resp.choices[0].delta.tool_calls is not None:

                tool_call = resp.choices[0].delta.tool_calls[0]
                index = tool_call.index
                name = tool_call.function.name
                args = tool_call.function.arguments

                # store.get_or_setf(
                #     delta_store, 'builder', ToolBuilder
                # )

                builder = store.get_or_set(
                    delta_store, 'builder', ToolBuilder
                )
                tool = builder.update(tool_call.id, index, name, args)

                delta_store['tool'] = tool
                if isinstance(tool, ToolCall):
                    delta_store['tools'].append(tool)
                    if self.run_call:
                        tool.forward(True)
                return [tool]
        else:
            tools = []
            if (
                resp.choices[0].finish_reason == 'tool_calls'
            ):
                delta_store['tool_calls'] = resp.choices[0].message.tool_calls
                for tool_call in resp.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    argstr = tool_call.function.arguments
                    args = json.loads(argstr)
                    
                    tool_def = self.tools[name]
                    
                    # TODO: Add in Async
                    cur_call = ToolCall(
                        tool_id=tool_call.id,
                        option=tool_def,
                        inputs=tool_def.input_model(**args)
                    )
                    if self.run:
                        cur_call.forward(True)
                    tools.append(cur_call)

                delta_store['tools'] = tools
                return tools
                
        return []
    

class LLM(LLM, ABC):
    """An adapter for the OpenAILLM
    """
    def __init__(
        self, 
        tools: typing.Iterable[ToolDef]=None,
        json_output: bool | pydantic.BaseModel | typing.Dict=False,
        api_key: str=None,
        **client_kwargs
    ):
        """
        Args:
            tools (ToolSet, optional): . Defaults to None.
            json_output (bool | pydantic.BaseMode | typing.Dict, optional): . Defaults to False.
            api_key (str, optional): . Defaults to None.
            client_kwargs (typing.Dict, optional): . Defaults to None.
        """
        super().__init__()
        convs = []
        if json_output is False:
            convs.append(TextConv())
        elif isinstance(json_output, pydantic.BaseModel):
            convs.append(ParsedConv(json_output))
        else:
            convs.append(StructConv(json_output))
        if tools is not None:
            convs.append(ToolConv(tools))
        super().__init__(
            procs=convs, 
        )
        self._client_kwargs = client_kwargs
        self._tools = tools
        self._json_output = json_output
        self._client = openai.Client(
            api_key=api_key, **client_kwargs
        )
        self._aclient = openai.AsyncClient(
            api_key=api_key, **client_kwargs
        )
        
    def spawn(self, 
        tools: typing.Iterable[ToolDef]=UNDEFINED,
        json_output: bool | pydantic.BaseModel | typing.Dict=UNDEFINED,
        procs: typing.List[RespConv]=UNDEFINED
    ):
        tools = coalesce(
            tools, self._tools
        )
        procs = coalesce(
            procs, self.procs
        )
        json_output = coalesce(json_output, self._json_output)
        return LLM(
            tools, json, 
        )


class ChatCompletion(LLM, ABC):
    """
    OpenAIChatComp is an adapter for the OpenAI Chat Completions API. It provides methods for 
    interacting with the API, including synchronous and asynchronous message forwarding, 
    streaming, and spawning new instances with modified configurations.
    """
    def spawn(self, 
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
    
    def forward(self, msg, **kwargs)-> typing.Tuple[Msg, typing.Any]:
        """
        Processes a message by forwarding it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            typing.Tuple[Msg, typing.Any]: A tuple containing the processed message and the 
            response from the LLM.
        Notes:
            - This method uses the `llm_forward` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """
        kwargs = {
            **self._kwargs, 
            **kwargs, 
            self._message_arg:to_list_input(msg)
        }

        return llm_forward(
            self._client.chat.completions.create, 
            _proc=self.procs, 
            **kwargs
        )

    async def aforward(self, msg, **kwargs) -> typing.Tuple[Msg, typing.Any]:
        """
        Processes a message by asynchronously forwarding it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            typing.Tuple[Msg, typing.Any]: A tuple containing the processed message and the 
            response from the LLM.
        Notes:
            - This method uses the `llm_forward` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """

        kwargs = {
            **self._kwargs, 
            **kwargs, 
            self._message_arg: to_list_input(msg)
        }
        return await llm_aforward(
            self._aclient.chat.completions.create, 
            _proc=self.procs, 
            **kwargs
        )

    def stream(self, msg, **kwargs) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        """
        Processes a message by streaming it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            typing.Tuple[Msg, typing.Any]: A tuple containing the processed message and the 
            response from the LLM.
        Notes:
            - This method uses the `llm_forward` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """

        kwargs = {
            **self._kwargs, 
            **kwargs, 
            self._message_arg:to_list_input(msg)
        }
        for r in llm_stream(
            self._client.chat.completions.create, 
            _proc=self.procs, 
            stream=True,
            **kwargs
        ):
            yield r
    
    async def astream(self, msg, *args, **kwargs) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:
        """
        Processes a message by streaming it to the language model (LLM) and returns the response.
        Args:
            msg (Msg): The message object to be processed by the LLM.
            **kwargs: Additional keyword arguments to customize the LLM request. These arguments
                      are merged with the default arguments stored in `self._kwargs`.
        Returns:
            typing.Tuple[Msg, typing.Any]: A tuple containing the processed message and the 
            response from the LLM.
        Notes:
            - This method uses the `llm_forward` function to handle the interaction with the LLM.
            - The `_resp_proc` parameter is used to process the response from the LLM.
        """
        kwargs = {
            **self._kwargs, 
            **kwargs, 
            self._message_arg:to_list_input(msg)
        }
        async for r in await llm_astream(
            self._aclient.chat.completions.create, 
            _proc=self.procs,
            stream=True,
            **kwargs
        ):
            yield r


class ToolExecConv(OutConv):
    """Use for converting a tool from a response
    """
    def __init__(self, name: str, from_: str='content'):
        """Create a reader for Primitive values

        Args:
            out (typing.Type): The type of data
        """
        super().__init__(
            name=name, from_=from_
        )
        self._name = name
    
    def delta(
        self, 
        resp, 
        delta_store: typing.Dict, 
        is_streamed: bool=False, 
        is_last: bool=True
    ) -> typing.Any:
        """Read in the output

        Args:
            message (str): The message to read

        Returns:
            typing.Any: The output of the reader
        """
        if resp is None or resp is utils.UNDEFINED:
            return utils.UNDEFINED
        
        if isinstance(resp, ToolCall):
            return resp()
        
        if isinstance(resp, typing.List):
            return [r() for r in resp]
        
        return utils.UNDEFINED
