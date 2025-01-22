import pkg_resources
import typing

from .._core import Msg
from ..ai import ToolSet, ToolCall
from ..ai._ai import RespProc
import json

# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if len(missing) > 0:

    raise RuntimeError(f'To use this module openai must be installed.')


class OpenAITextProc(RespProc):
    """
    OpenAITextProc is a class that processes an OpenAI response and extracts text outputs from it.
    Methods:
        __init__():
            Initializes the OpenAITextProc instance.
        __call__(response, msg):
            Processes the OpenAI response and extracts the main content.
            Args:
                response: The response object from OpenAI.
                msg: A dictionary to store the extracted content.
                A tuple containing the updated msg dictionary and the extracted content.
        delta(response, msg, delta_store):
            Processes the OpenAI response to extract incremental content updates (deltas).
            Args:
                response: The response object from OpenAI.
                msg: A dictionary to store the extracted content.
                delta_store: A dictionary to store the accumulated deltas.
                A tuple containing the updated msg dictionary and the extracted delta.
        prep():
            Prepares and returns an empty dictionary.
                An empty dictionary.
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
        return msg, content
    
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

        if 'content' not in delta_store:
            delta_store['content'] = ''
        
        if delta_store is None:
            msg['content'] = None
            return msg, None
        
        if delta is not None:
            delta_store['content'] += delta
        msg['content'] = delta
        return msg, delta

    def prep(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        return {}


class OpenAIToolProc(RespProc):
    """
    Process OpenAI LLM responses to extract tool calls.
    This class extends the RespProc class and is designed to handle responses from OpenAI's language model,
    specifically to extract and manage tool calls embedded within the responses.
    Attributes:
        tools (ToolSet): A set of tools available for processing.
    Methods:
        __call__(response, msg):
            Processes the response to extract tool calls and updates the message metadata.
        delta(response, msg, delta_store):
            Handles incremental updates to the tool calls from the response and updates the delta store.
        prep():
            Prepares the tools for input.
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
            response (_type_): _description_
            msg (_type_): _description_

        Returns:
            : _description_
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

    def delta(self, response, msg, delta_store):

        if 'idx' not in delta_store:
            delta_store.update({
                'idx': None,
                'calls': [],
                'prep': [],
                'tools': []
            })
        
        result = None
        if response.choices[0].delta.tool_calls is not None:
            tool_call = response.choices[0].delta.tool_calls[0]
            idx = delta_store['idx']
            print('Idx: ', delta_store['idx'], tool_call.index)
            if tool_call.index != idx:
                print('Updating index')
                if idx is not None:
                    cur = delta_store['prep'][-1]
                    result = ToolCall(
                        option=self.tools[cur['name']],
                        args=json.loads(cur['argstr'])
                    )
                    delta_store['tools'].append(result)
                delta_store['idx'] = tool_call.index
                print(delta_store['idx'])
                cur_tool = {
                    'name': tool_call.function.name,
                    'argstr': tool_call.function.arguments,
                }
                delta_store['prep'].append(cur_tool)
            else:
                cur_tool = delta_store['prep'][idx]
                cur_tool['argstr'] += tool_call.function.arguments
                result = None
        elif delta_store['idx'] is not None:
            cur = delta_store['prep'][-1]

            result = ToolCall(
                option=self.tools[cur['name']],
                args=json.loads(cur['argstr'])
            )
            delta_store['tools'].append(result)
            delta_store['idx'] = None
            # msg['meta']['tools']['calls'].append(cur_tool)

        msg['meta']['tools'] = [*delta_store['tools']]

        return result

    def prep(self):
        
        return {
            'tools': [tool.to_input() for tool in self.tools] if len(self.tools) != 0 else None
        }

