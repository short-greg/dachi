from abc import ABC
import pkg_resources
import typing
import json

import pydantic

from . import Msg
from ._ai import ToolSet, ToolCall, ToolBuilder
from ._resp import RespConv
from . import END_TOK
from ._ai import LLM, llm_aforward, llm_astream, llm_forward, llm_stream
from ..utils import UNDEFINED, coalesce
from .. import utils

# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

import openai

if len(missing) > 0:

    raise RuntimeError(f'To use this module openai must be installed.')

# stream
# parsed
# create

class OpenAITextConv(RespConv):
    """
    OpenAITextProc is a class that processes an OpenAI response and extracts text outputs from it.
    """

    def __init__(self):
        super().__init__(True)

    def __call__(self, response, msg):
        """
        Processes a response from OpenAI and extracts the text content.
        Args:
            response (object): The response object from OpenAI containing choices and messages.
            msg (dict): A dictionary to be updated with the extracted content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted content string.
        """

        content = response.choices[0].message.content
        msg['content'] = content if content is not None else None
        return content
    
    def delta(self, response, msg, delta_store: typing.Dict):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        delta = response.choices[0].delta.content

        msg['delta']['delta'] = delta

        utils.call_or_set(
            msg['delta'], 'content', delta, 
            lambda x, delta: x + delta
        )
        msg['content'] = msg['delta']['content']
        return delta

    def prep(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        return {}


class OpenAIStructConv(RespConv):
    """
    OpenAITextProc is a class that processes an OpenAI response and extracts text outputs from it.
    """

    def __init__(self, struct: pydantic.BaseModel | typing.Dict=None):
        super().__init__(False)
        self._struct = struct

    def __call__(self, response, msg):
        """
        Processes a response from OpenAI and extracts the text content.
        Args:
            response (object): The response object from OpenAI containing choices and messages.
            msg (dict): A dictionary to be updated with the extracted content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted content string.
        """
        json_data = response.choices[0].message.content
        if response.type == 'error':
            raise RuntimeError(response.error)
        msg['content'] = json_data
        msg['meta']['struct'] = json.loads(json_data)
        return msg['meta']['struct']

    def delta(self, response, msg, delta_store: typing.Dict):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        delta = response.choices[0].delta.conten

        if response == END_TOK:
            struct = json.loads(delta_store['content'])
            return struct

        msg['delta']['content'] = delta
        _, delta = delta_store.adv(
            delta, 'content'
        )
        return delta
        
    def prep(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        if isinstance(self._struct, typing.Dict):
            return {'response_format': {
                "type": "json_schema",
                "json_schema": self._struct
            }}
        elif isinstance(self._struct, pydantic.BaseModel):
            return {'response_format': self._struct}
        return {
            'response_format': "json_object"
        }


class OpenAIStreamStructConv(OpenAIStructConv):
    
    def delta(self, response, msg, delta_store: typing.Dict):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        if response.type == "content.delta":
            if response.parsed is not None:
                print("Chunk received:", response.parsed)
        elif response.type == "content.done":
            return None
        elif response.type == "error":
            raise RuntimeError(response.error)


class OpenAIParsedStructConv(OpenAIStructConv):
    """For use with the "parse" API
    """
    
    def __call__(self, response, msg):
        """
        Processes a response from OpenAI and extracts the text content.
        Args:
            response (object): The response object from OpenAI containing choices and messages.
            msg (dict): A dictionary to be updated with the extracted content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted content string.
        """
        msg['content'] = str(response)
        msg['meta']['struct'] = response
        return response
    
    def delta(self, response, msg, delta_store: typing.Dict):
        """
        Processes a delta response and extracts text.
        Args:
            response: The response object containing the delta.
            msg: A dictionary to store the message content.
            delta_store (typing.Dict): A dictionary to store the accumulated delta content.
        Returns:
            tuple: A tuple containing the updated msg dictionary and the extracted delta content.
        """
        raise RuntimeError('Cannot use ParseConv with a stream')


class OpenAIToolConv(RespConv):
    """
    Process OpenAI LLM responses to extract tool calls.
    This class extends the RespProc class and is designed to handle responses from OpenAI's language model,
    specifically to extract and manage tool calls embedded within the responses.
    """
    def __init__(self, tools: ToolSet):
        """

        Args:
            tools (typing.Dict[str, ToolOption]): 
        """
        super().__init__(True)
        self.tools = tools

    def __call__(self, response, msg: Msg):
        """

        Args:
            response (_type_): 
            msg (_type_): 

        Returns:
            : 
        """
        
        result = None
        if response.choices[0].finish_reason == 'tool_calls':
            result = []
            for tool_call in response.choices[0].message.tool_calls:
                name = tool_call.function.name
                argstr = tool_call.function.arguments
                args = json.loads(argstr)
                
                cur_call = ToolCall(
                    option=self.tools[name], 
                    args=args
                )
                msg['meta']['tools'].append(cur_call)
                result.append(cur_call)
            
        return result

    def delta(self, response, msg, delta_store: typing.Dict):

        if response is END_TOK:
            msg['meta']['tools'] = self.delta_store['delta']['builder'].tools
            msg['delta']['tools'] = None
            return None

        delta = response.choices[0].delta.content
        msg['delta']['content'] = delta
        if response.choices[0].delta.tool_calls is not None:
            if 'tools' not in msg['meta']:
                msg['meta']['tools'] = []

            tool_call = response.choices[0].delta.tool_calls[0]
            index = tool_call.index
            name = tool_call.function.name
            args = tool_call.function.arguments

            utils.get_or_setf(
                delta_store, 'builder', ToolBuilder
            )

            builder = utils.get_or_set(delta_store, 'builder', ToolBuilder)
            tool = builder.update(index, name, args)
            msg['delta']['tool'] = tool
            if tool is not None:
                msg['meta']['tools'].append(tool)
            return delta
    
            # tools, delta = delta_store.adv(
            #     'tools', 
            #     {'index': index, 'name': name, 'args': args}, ToolBuilder()
            # )
        
        # 1) 
        # if 'idx' not in delta_store:
        #     delta_store.update({
        #         'idx': None,
        #         'calls': [],
        #         'prep': [],
        #         'tools': []
        #     })
        # if response is END_TOK:
        #     msg['meta']['tools'] = [*delta_store['tools']]
        #     msg['delta']['tools'] = None
        #     return None
        
        # result = None
        # if response.choices[0].delta.tool_calls is not None:
        #     tool_call = response.choices[0].delta.tool_calls[0]
        #     idx = delta_store['idx']
        #     if tool_call.index != idx:
        #         if idx is not None:
        #             cur = delta_store['prep'][-1]
        #             result = ToolCall(
        #                 option=self.tools[cur['name']],
        #                 args=json.loads(cur['argstr'])
        #             )
        #             delta_store['tools'].append(result)
        #         delta_store['idx'] = tool_call.index
        #         cur_tool = {
        #             'name': tool_call.function.name,
        #             'argstr': tool_call.function.arguments,
        #         }
        #         delta_store['prep'].append(cur_tool)
        #         msg['delta']['tool'] = cur_tool
        #     else:
        #         cur_tool = delta_store['prep'][idx]
        #         cur_tool['argstr'] += tool_call.function.arguments
        #         result = None
        #         msg['delta']['tool'] = None

        # elif delta_store['idx'] is not None:
        #     cur = delta_store['prep'][-1]

        #     result = ToolCall(
        #         option=self.tools[cur['name']],
        #         args=json.loads(cur['argstr'])
        #     )
        #     delta_store['tools'].append(result)
        #     delta_store['idx'] = None
        #     msg['delta']['tool'] = cur_tool

        # msg['meta']['tools'] = [*delta_store['tools']]

        # return result

    def prep(self):
        
        return {
            'tools': [tool.to_input() for tool in self.tools] if len(self.tools) != 0 else None
        }


class OpenAILLM(LLM, ABC):

    def __init__(
        self, 
        tools: ToolSet=None,
        json_output: bool | pydantic.BaseMode | typing.Dict=False,
        api_key: str=None,
        client_kwargs: typing.Dict=None
    ):
        convs = []
        if json_output is False:
            convs.append(OpenAITextConv())
        else:
            convs.append(OpenAIStructConv(json_output))
        if tools is not None:
            convs.append(OpenAIToolConv(tools))
        super().__init__(
            resp_procs=convs, 
        )
        self._client_kwargs = client_kwargs
        self._tools = tools
        self._json_output = json_output
        self._client = openai.Client(
            api_key, **client_kwargs
        )
        self._aclient = openai.AsyncClient(api_key, **client_kwargs)
        
    def spawn(self, 
        tools: ToolSet=UNDEFINED,
        json_output: bool | pydantic.BaseMode | typing.Dict=UNDEFINED
    ):
        tools = coalesce(
            tools, self._tools
        )
        json_output = coalesce(json_output, self._json_output)
        return OpenAILLM(
            tools, json
        )

class OpenAICreateLLM(OpenAILLM, ABC):

    def spawn(self, 
        tools: ToolSet=UNDEFINED,
        json_output: bool | pydantic.BaseMode | typing.Dict=UNDEFINED
    ):
        tools = coalesce(
            tools, self._tools
        )
        json_output = coalesce(json_output, self._json_output)
        return OpenAILLM(
            tools, json
        )
    
    def forward(self, msg, **kwargs)-> typing.Tuple[Msg, typing.Any]:
        
        kwargs = {**self._kwargs, **kwargs}

        return llm_forward(
            self._client.chat.completions.create, 
            _resp_proc=self.resp_procs, 
            msg=msg,
            **kwargs
        )

    async def aforward(self, msg, **kwargs) -> typing.Tuple[Msg, typing.Any]:
        kwargs = {**self._kwargs, **kwargs}

        return await llm_aforward(
            self._aclient.chat.completions.create, 
            _resp_proc=self.resp_procs, 
            msg=msg,
            **kwargs
        )

    def stream(self, msg, **kwargs) -> typing.Iterator[typing.Tuple[Msg, typing.Any]]:
        kwargs = {**self._kwargs, **kwargs}

        for r, c in llm_stream(
            self._client.chat.completions.create, 
            _resp_proc=self.resp_procs, 
            msg=msg,
            stream=True,
            **kwargs
        ):
            yield r, c
    
    async def astream(self, msg, *args, **kwargs) -> typing.AsyncIterator[typing.Tuple[Msg, typing.Any]]:

        async for r, c in await llm_astream(
            self._aclient.chat.completions.create, 
            _resp_proc=self.resp_procs, 
            msg=msg,
            stream=True,
            **kwargs
        ):
            yield r, c
