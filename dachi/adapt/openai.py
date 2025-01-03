import pkg_resources
import openai
import typing

from functools import singledispatch
from .._core import (
    Module, Dialog,  Msg
)
from ..utils import UNDEFINED
from ..ai import LLM, LLM_PROMPT, Get

from typing import Dict, List

# TODO: add utility for this
required = {'openai'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed


if len(missing) > 0:

    raise RuntimeError(f'To use this module openai must be installed.')


class OpenAILLM(LLM):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    def __init__(self, model: str, client_kwargs: typing.Dict=None, **kwargs) -> None:
        """Create an OpenAIChat model

        Args:
            model (str): The name of the model
        """
        super().__init__()
        self.client_kwargs = client_kwargs or {}
        self.model = model
        self.kwargs = kwargs

    def process_response(self, response) -> Msg:
        """
        Processes the response from the OpenAI API and returns a Msg object.
        Args:
            response: The response object from the OpenAI API.
        Returns:
            Msg: A message object containing the role, type, tool, and meta information
                 based on the response. If the response includes a function call, the
                 message type is set to 'function' and includes the function name and
                 arguments. Otherwise, the message content is set to the response content.
        """
        if response.choices[0].message.function_call:
            return Msg(
                role='assistant',
                type_='function',
                tool={
                    'name': response.choices[0].message.function_call.name,
                    'arguments': response.choices[0].message.function_call.arguments
                },
                meta={'response': response}
            )
        else:
            return Msg(
                role='assistant', 
                content=response.choices[0].message.content,
                meta={'response': response}
            )

    def process_delta_response(self, delta_response, response: Msg=None) -> Msg:
        """
        Processes a delta response and updates or creates a Msg object accordingly. It aggregates the message sent from the AI and sets the delta in the Delta dictionary of Msg
        Args:
            delta_response (object): The delta response object containing choices with delta information.
            response (Msg, optional): An existing Msg object to update. Defaults to None.
        Returns:
            Msg: The updated or newly created Msg object.
        """ 
        if delta_response.choices[0].delta.function_call:
            if response is None:
                response = Msg(
                    role='assistant',
                    type_='function',
                    tool={
                    'name': delta_response.choices[0].delta.function_call.name,
                    'arguments': delta_response.choices[0].delta.function_call.arguments
                    },
                    meta={'delta': delta_response}
                )
            else:
                if response.tool is None:
                    response.tool = {
                        'name': delta_response.choices[0].delta.function_call.name,
                        'arguments': delta_response.choices[0].delta.function_call.arguments
                    }
                else:
                    response.tool['arguments'] += delta_response.choices[0].delta.function_call.arguments
            response.meta['delta'] = delta_response

        elif delta_response.choices[0].delta.content:
            if response is None:
                response = Msg(
                    role='assistant',
                    content=delta_response.choices[0].delta.content,
                    meta={'delta': delta_response}
                )
            else:
                response.content = (response.content or '') + delta_response.choices[0].delta.content
                response.meta['delta'] = delta_response

        return response

    def forward(self, prompt: LLM_PROMPT, **kwarg_override) -> Msg:
        """Execute the model

        Args:
            prompt (AIPrompt): The message to send the model

        Returns:
            AIResponse: the response from the model
        """
        kwargs = {**self.kwargs, **kwarg_override}
        dialog = self.prepare_dialog(
            prompt
        )
        messages = list(dialog.messages())
        client = openai.OpenAI(**self.client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return self.process_response(response)

    async def aforward(self, prompt: LLM_PROMPT, **kwarg_override) -> Msg:
        """Execute the model

        Args:
            prompt (AIPrompt): The message to send the model

        Returns:
            AIResponse: the response from the model
        """
        kwargs = {**self.kwargs, **kwarg_override}
        dialog = self.prepare_dialog(
            prompt
        )
        messages = list(dialog.messages())
        client = openai.AsyncOpenAI(**self.client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return self.process_response(response)

    def stream(self, prompt: LLM_PROMPT, **kwarg_override) -> typing.Iterator[Msg]:
        """Execute the model

        Args:
            prompt (AIPrompt): The message to send the model

        Returns:
            AIResponse: the response from the model
        """
        kwargs = {**self.kwargs, **kwarg_override}
        dialog = self.prepare_dialog(
            prompt
        )
        messages = list(dialog.messages())
        client = openai.OpenAI(**self.client_kwargs)
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        response = None
        for chunk in stream:
            yield self.process_delta_response(chunk, response)

    async def astream(self, prompt: LLM_PROMPT, **kwarg_override) -> typing.AsyncIterator[Msg]:
        """Execute the model

        Args:
            prompt (AIPrompt): The message to send the model

        Returns:
            AIResponse: the response from the model
        """
        kwargs = {**self.kwargs, **kwarg_override}
        dialog = self.prepare_dialog(
            prompt
        )
        messages = list(dialog.messages())
        client = openai.AsyncOpenAI(**self.client_kwargs)
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        response = None
        async for chunk in stream:
            yield self.process_delta_response(chunk, response)

    def get_text(self, x: str='content', dx: str=lambda msg: msg['dx']['content']):
        return Get(
            self, x, dx
        )

        # client = openai.OpenAI(**self.client_kwargs)
        # response = client.chat.completions.create(
        #     model=self.model,
        #     messages=self.to_prompt(dialog),
        #     **kwargs
        # )
        # message = process_api_response(
        #     response, dialog.reader(), dialog.schema
        # )
        # # text = response.choices[0].message.content
        # # parsed = prompt.reader().read(text)

        # # message = AssistantMsg(text=text, parsed=parsed)
        # return message, dialog.add(message)

    # def stream(
    #     self, prompt: LLM_PROMPT, 
    #     **kwarg_override
    # ) -> typing.Iterator[LLM_RESPONSE]:
    #     """Stream the model

    #     Args:
    #         prompt (AIPrompt): the model prmopt

    #     Yields:
    #         Iterator[typing.Iterator[typing.Tuple[AIResponse, AIResponse]]]: the responses from the model
    #     """
    #     kwargs = {**self.kwargs, **kwarg_override}
    #     dialog, kwargs = self.prepare_dialog(
    #         prompt, kwargs
    #     )
        
    #     client = openai.OpenAI(**self.client_kwargs)
    #     query = client.chat.completions.create(
    #         model=self.model,
    #         messages=self.convert_messages(prompt.aslist()),
    #         stream=True,
    #         **kwargs
    #     )
    #     cur_message = ''

    #     #p = dialog.reader()
    #     msg = AssistantMsg('')
    #     for chunk in query:
    #         delta = chunk.choices[0].delta.content

    #         if delta is None:
    #             delta = ''
            
    #         cur_message = cur_message + delta
    #         message = process_delta_api_response(
    #             chunk, msg, dialog.reader(), dialog.schema
    #         )
    #         # message = AssistantMsg(
    #         #     cur_message, delta=DeltaMsg(text=delta)
    #         # )
    #         # dx_val = p.read(delta)

    #         yield message, dialog.add(message)
    #         # dx_val = p.read(delta)

    # async def aforward(
    #     self, prompt: LLM_PROMPT, **kwarg_override
    # ) -> LLM_RESPONSE:
    #     """Run this query for asynchronous operations
    #     The default behavior is simply to call the query

    #     Args:
    #         data: Data to pass to the API

    #     Returns:
    #         typing.Any: 
    #     """
    #     kwargs = {**self.kwargs, **kwarg_override}
    #     dialog, kwargs = self.prepare_dialog(
    #         prompt, kwargs
    #     )

    #     client = openai.AsyncOpenAI(**self.client_kwargs)
    #     response = await client.chat.completions.create(
    #         model=self.model,
    #         messages=self.to_prompt(dialog),
    #         **kwargs
    #     )
    #     # text = response.choices[0].message.content
    #     # parsed = prompt.reader().read(text)

    #     message = process_api_response(
    #         response, dialog.reader(), dialog.schema
    #     )
    #     # message = AssistantMsg(text=text, parsed=parsed)
    #     return message, dialog.add(message)

    # async def astream(
    #     self, prompt: LLM_PROMPT, **kwarg_override
    # ) -> typing.AsyncIterator[LLM_RESPONSE]:
    #     """Run this query for asynchronous streaming operations
    #     The default behavior is simply to call the query

    #     Args:
    #         prompt (AIPrompt): The data to pass to the API

    #     Yields:
    #         typing.Dict: The data returned from the API
    #     """
    #     kwargs = {**self.kwargs, **kwarg_override}
    #     dialog, kwargs = self.prepare_dialog(
    #         prompt, kwargs
    #     )
        
    #     client = openai.AsyncOpenAI(**self.client_kwargs)
    #     query = await client.chat.completions.create(
    #         model=self.model,
    #         messages=self.convert_messages(prompt.aslist()),
    #         stream=True,
    #         **kwargs
    #     )
    #     cur_message = ''
    #     #p = dialog.reader()
    #     async for chunk in query:
    #         # delta = chunk.choices[0].delta.content
    #         message = process_delta_api_response(
    #             chunk, dialog.reader(), dialog.schema
    #         )
    #         yield message, dialog.add(message)
            

    #         # if delta is None:
    #         #     delta = ''
            
    #         # cur_message = cur_message + delta
            
    #         # message = AssistantMsg(
    #         #     cur_message, delta=DeltaMsg(text=delta)
    #         # )
    #         # dx_val = p.read(delta)

    # def __call__(self, prompt: LLM_PROMPT, **kwarg_override) -> LLM_RESPONSE:
    #     """Execute the AIModel

    #     Args:
    #         prompt (AIPrompt): The prompt

    #     Returns:
    #         AIResponse: Get the response from the AI
    #     """
    #     return self.forward(prompt, **kwarg_override)






# def process_api_response(api_response, read: Reader, scheme: Schema):
#     """
#     Processes the API response to determine if a tool (function) was called, and if so, which one.

#     Parameters:
#         api_response (object): The response object from the OpenAI API.

#     Returns:
#         AssistantMsg: An object containing structured information about the API response.
#     """
#     # Initialize the result fields
#     tool_info = None
#     parsed_info = None

#     # Access the 'choices' attribute in the response object
#     choices = api_response.choices
#     if choices and len(choices) > 0:
#         choice = choices[0]  # Assuming single choice processing
#         message = choice.message

#         if choice.finish_reason == 'funtion' and message.function_call:
#             tool_info = {
#                 'name': message.function_call.name,
#                 'arguments': message.function_call.arguments
#             }
#         else:
#             tool_info = None
        
#         text = message.content
#         # parsed_info = {
#         #     'finish_reason': choice.finish_reason,
#         #     'content': message.content
#         # }

#     # Create an AssistantMsg object
#     return AssistantMsg(
#         text=message.content,
#         response=api_response,
#         parsed=parsed_info if tool_info is None else None,
#         tool=tool_info
#     )

# def process_delta_api_response(delta_response, assistant_msg):
#     """
#     Processes a delta response from OpenAI's streaming API and updates the AssistantMsg object.

#     Parameters:
#         delta_response (object): The delta response object from the OpenAI API.
#         assistant_msg (AssistantMsg): The AssistantMsg object to update.

#     Returns:
#         DeltaMsg: An object representing the current delta.
#     """
#     # Extract delta information
#     delta_text = delta_response.text
#     delta_tool = delta_response.function_call

#     # Update the AssistantMsg object
#     if delta_tool:
#         if not assistant_msg.tool:
#             assistant_msg.tool = {
#                 'name': delta_tool.name,
#                 'arguments': delta_tool.arguments
#             }
#     if delta_text:
#         assistant_msg.text = (assistant_msg.text or '') + delta_text

#     # Create and set the DeltaMsg object
#     delta_msg = DeltaMsg(
#         text=delta_text,
#         response=delta_response,
#         parsed=None  # Delta is not parsed further
#     )
#     assistant_msg.delta = delta_msg

#     return delta_msg


# class OpenAILLM(LLM):
#     """APIAdapter allows one to adapt various WebAPI or otehr
#     API for a consistent interface
#     """

#     @singledispatch
#     def process_message(self, message, kwargs: typing.Dict) -> typing.Dict:
#         return {
#             'role': 'user',
#             'content': message.get_text()
#         }

#     @process_message.register
#     def process_message(self, message: SystemMsg, kwargs: typing.Dict) -> typing.Dict:
#         schema = message.get_scheme()
#         if schema is not None:
#             kwargs['structure'] = schema
#         return {
#             'role': 'system',
#             'content': message.get_text(),
#             'images': message.get_images(),
#         }

#     @process_message.register
#     def process_message(self, message: UserMsg, kwargs: typing.Dict) -> typing.Dict:
#         return {
#             'role': 'user',
#             'content': message.get_text(),
#             'images': message.get_images(),
#         }


#     @process_message
#     def process_message(self, message: AssistantMsg, kwargs: typing.Dict) -> typing.Dict:
#         return {
#             'role': 'assistant',
#             'content': message.get_text()
#         }

#     @process_message
#     def process_message(self, message: ToolMsg, kwargs: typing.Dict) -> typing.Dict:
#         return {
#             'role': 'function',
#             'name': message.name,
#             'content': message.get_text()
#         }

#     def __init__(self, model: str, client_kwargs: typing.Dict=None, **kwargs) -> None:
#         """Create an OpenAIChat model

#         Args:
#             model (str): The name of the model
#         """
#         super().__init__()
#         self.client_kwargs = client_kwargs or {}
#         self.model = model
#         self.kwargs = kwargs

#     def prepare_dialog(self, prompt: typing.Union[Dialog, ChatMsg], kwargs) -> Dialog:
#         if isinstance(prompt, ChatMsg):
#             prompt = ListDialog([prompt])

#         messages = []
#         for p_i in prompt:
#             messages.append(self.process_message(p_i, kwargs))
#         return messages

#     def forward(self, prompt: LLM_PROMPT, **kwarg_override) -> LLM_RESPONSE:
#         """Execute the model

#         Args:
#             prompt (AIPrompt): The message to send the model

#         Returns:
#             AIResponse: the response from the model
#         """
#         kwargs = {**self.kwargs, **kwarg_override}
#         dialog, kwargs = self.prepare_dialog(
#             prompt, kwargs
#         )

#         client = openai.OpenAI(**self.client_kwargs)
#         response = client.chat.completions.create(
#             model=self.model,
#             messages=self.to_prompt(dialog),
#             **kwargs
#         )
#         message = process_api_response(
#             response, dialog.reader(), dialog.schema
#         )
#         # text = response.choices[0].message.content
#         # parsed = prompt.reader().read(text)

#         # message = AssistantMsg(text=text, parsed=parsed)
#         return message, dialog.add(message)

#     def stream(
#         self, prompt: LLM_PROMPT, 
#         **kwarg_override
#     ) -> typing.Iterator[LLM_RESPONSE]:
#         """Stream the model

#         Args:
#             prompt (AIPrompt): the model prmopt

#         Yields:
#             Iterator[typing.Iterator[typing.Tuple[AIResponse, AIResponse]]]: the responses from the model
#         """
#         kwargs = {**self.kwargs, **kwarg_override}
#         dialog, kwargs = self.prepare_dialog(
#             prompt, kwargs
#         )
        
#         client = openai.OpenAI(**self.client_kwargs)
#         query = client.chat.completions.create(
#             model=self.model,
#             messages=self.convert_messages(prompt.aslist()),
#             stream=True,
#             **kwargs
#         )
#         cur_message = ''

#         #p = dialog.reader()
#         msg = AssistantMsg('')
#         for chunk in query:
#             delta = chunk.choices[0].delta.content

#             if delta is None:
#                 delta = ''
            
#             cur_message = cur_message + delta
#             message = process_delta_api_response(
#                 chunk, msg, dialog.reader(), dialog.schema
#             )
#             # message = AssistantMsg(
#             #     cur_message, delta=DeltaMsg(text=delta)
#             # )
#             # dx_val = p.read(delta)

#             yield message, dialog.add(message)
#             # dx_val = p.read(delta)

#     async def aforward(
#         self, prompt: LLM_PROMPT, **kwarg_override
#     ) -> LLM_RESPONSE:
#         """Run this query for asynchronous operations
#         The default behavior is simply to call the query

#         Args:
#             data: Data to pass to the API

#         Returns:
#             typing.Any: 
#         """
#         kwargs = {**self.kwargs, **kwarg_override}
#         dialog, kwargs = self.prepare_dialog(
#             prompt, kwargs
#         )

#         client = openai.AsyncOpenAI(**self.client_kwargs)
#         response = await client.chat.completions.create(
#             model=self.model,
#             messages=self.to_prompt(dialog),
#             **kwargs
#         )
#         # text = response.choices[0].message.content
#         # parsed = prompt.reader().read(text)

#         message = process_api_response(
#             response, dialog.reader(), dialog.schema
#         )
#         # message = AssistantMsg(text=text, parsed=parsed)
#         return message, dialog.add(message)

#     async def astream(
#         self, prompt: LLM_PROMPT, **kwarg_override
#     ) -> typing.AsyncIterator[LLM_RESPONSE]:
#         """Run this query for asynchronous streaming operations
#         The default behavior is simply to call the query

#         Args:
#             prompt (AIPrompt): The data to pass to the API

#         Yields:
#             typing.Dict: The data returned from the API
#         """
#         kwargs = {**self.kwargs, **kwarg_override}
#         dialog, kwargs = self.prepare_dialog(
#             prompt, kwargs
#         )
        
#         client = openai.AsyncOpenAI(**self.client_kwargs)
#         query = await client.chat.completions.create(
#             model=self.model,
#             messages=self.convert_messages(prompt.aslist()),
#             stream=True,
#             **kwargs
#         )
#         cur_message = ''
#         #p = dialog.reader()
#         async for chunk in query:
#             # delta = chunk.choices[0].delta.content
#             message = process_delta_api_response(
#                 chunk, dialog.reader(), dialog.schema
#             )
#             yield message, dialog.add(message)
            

#             # if delta is None:
#             #     delta = ''
            
#             # cur_message = cur_message + delta
            
#             # message = AssistantMsg(
#             #     cur_message, delta=DeltaMsg(text=delta)
#             # )
#             # dx_val = p.read(delta)

#     def __call__(self, prompt: LLM_PROMPT, **kwarg_override) -> LLM_RESPONSE:
#         """Execute the AIModel

#         Args:
#             prompt (AIPrompt): The prompt

#         Returns:
#             AIResponse: Get the response from the AI
#         """
#         return self.forward(prompt, **kwarg_override)


# create tool

# 1) message contains a cue
# 2) I need some way to "stream" with the cue
# 3) the assistant returns an image
# 4) the assistant returns tool usage
# 5) openai.update(delta)
#    dx_response.update()
#    response.update(dx)
# 6) 


# response = OpenAIResponse()
# 


# message = self.process_response()

# class OpenAIChatModel(LLM):
#     """A model that uses OpenAI's Chat API
#     """
#     def __init__(self, model: str, client_kwargs: typing.Dict=None, **kwargs) -> None:
#         """Create an OpenAIChat model

#         Args:
#             model (str): The name of the model
#         """
#         super().__init__()
#         self.client_kwargs = client_kwargs or {}
#         self.model = model
#         self.kwargs = kwargs

#     def create_prompt(self, prompt: typing.Union[Message, Dialog], **kwarg_override):

#         tools = self.get_tools(prompt)
#         messages = self.get_messages(prompt)
        
#         kwargs = {
#             **self.kwargs,
#             **kwarg_override
#         }

#         # add tools
#         if tools is not None:
#             pass

#         return 

#     def forward(self, prompt: typing.Union[Message, Dialog], **kwarg_override) -> Message:
#         """Execute the model

#         Args:
#             prompt (AIPrompt): The message to send the model

#         Returns:
#             AIResponse: the response from the model
#         """
#         # Need to extract the tools


#         prompt = self.create_prompt(prompt)

#         client = openai.OpenAI(**self.client_kwargs)

#         response = client.chat.completions.create(
#             model=self.model,
#             messages=self.convert_messages(prompt.aslist()),
#             **kwargs
#         )
#         p = prompt.reader()
#         text = response.choices[0].message.content
#         message = TextMessage('assistant', text)
        
#         val = p.read(text)
        
#         return AIResponse(
#             message,
#             response,
#             val
#         )

#     def stream(
#         self, prompt: AIPrompt, 
#         **kwarg_override
#     ) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:
#         """Stream the model

#         Args:
#             prompt (AIPrompt): the model prmopt

#         Yields:
#             Iterator[typing.Iterator[typing.Tuple[AIResponse, AIResponse]]]: the responses from the model
#         """
#         kwargs = {
#             **self.kwargs,
#             **kwarg_override
#         }
        
#         client = openai.OpenAI(**self.client_kwargs)
#         query = client.chat.completions.create(
#             model=self.model,
#             messages=self.convert_messages(prompt.aslist()),
#             stream=True,
#             **kwargs
#         )
#         cur_message = ''
#         p = prompt.reader()
#         for chunk in query:
#             delta = chunk.choices[0].delta.content

#             if delta is None:
#                 delta = ''
            
#             cur_message = cur_message + delta
            
#             message = TextMessage('assistant', cur_message)
#             dx = TextMessage('assistant', delta)

#             dx_val = p.read(delta)
#             yield AIResponse(
#                 message, chunk, p.read(cur_message)
#             ), AIResponse(
#                 dx, chunk, dx_val
#             )
    
#     async def aforward(
#         self, prompt: AIPrompt, 
#         **kwarg_override
#     ) -> typing.Tuple[str, typing.Dict]:
#         """

#         Args:
#             prompt (AIPrompt): The 

#         Returns:
#             typing.Tuple[str, typing.Dict]: _description_
#         """
#         client = openai.AsyncOpenAI(**self.client_kwargs)

#         kwargs = {
#             **self.kwargs,
#             **kwarg_override
#         }
#         p = prompt.reader()

#         response = await client.chat.completions.create(
#             model=self.model,
#             messages=self.convert(prompt),
#             **kwargs
#         )
#         text = response.choices[0].message.content
#         return AIResponse(
#             TextMessage('assistant', text),
#             response,
#             p.read(text)
#         )

#     async def astream(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[typing.Tuple[AIResponse, AIResponse]]:
#         """Stream the model asyncrhonously

#         Args:
#             prompt (AIPrompt): The prompt for the model

#         Yields:
#             Iterator[AsyncIterator[typing.Tuple[AIResponse, AIResponse]]]: The response from the model
#         """

#         kwargs = {
#             **self.kwargs,
#             **kwarg_override
#         }
#         client = openai.AsyncOpenAI(**self.client_kwargs)
#         query = await client.chat.completions.create(
#             model=self.model,
#             messages=self.convert(prompt.aslist()),
#             stream=True,
#             **kwargs
#         )
#         p = prompt.reader()

#         cur_message = ''
#         async for chunk in query:
#             delta = chunk.choices[0].delta.content
#             if delta is None:
#                 delta = ''
#             cur_message = cur_message + delta

#             yield AIResponse(
#                 TextMessage('assistant', cur_message),
#                 chunk,
#                 p.stream_read(cur_message),
#             ), AIResponse(
#                 TextMessage('assistant', delta),
#                 chunk,
#                 None
#             )


# class OpenAIMessage(TextMessage):
#     """A text message made to work with OpenAI. Especially makes
#     using prompt more straightforward
#     """
    
#     def prompt(self, model: typing.Union[AIModel, str], **kwarg_overrides) -> AIResponse:
#         """prompt the model

#         Args:
#             model (typing.Union[AIModel, str]): The model to execute

#         Returns:
#             AIResponse: The response from the model
#         """
#         if isinstance(model, str):
#             model = OpenAIChatModel(model, **kwarg_overrides)

#         return super().prompt(model, **kwarg_overrides)


# class OpenEmbeddingModel(AIModel):
#     """
#     """

#     def __init__(self, model: str, **kwargs) -> None:
#         """Create an OpenEmbeddingModel

#         Args:
#             model (str): The name of the model
#         """
#         super().__init__()
#         self.model = model
#         self.kwargs = kwargs

#     def convert(self, message: Message) -> typing.Dict:
#         """Convert the messages to a format to use for the OpenAI API

#         Args:
#             messages (typing.List[Message]): The messages

#         Returns:
#             typing.List[typing.Dict]: The input to the API
#         """
#         return list(
#             render(text_i) for text_i in message['texts']
#         )

#     def forward(self, prompt: AIPrompt, **kwarg_override) -> Dict:
#         """Execute the Embedding model

#         Args:
#             prompt (AIPrompt): 

#         Returns:
#             Dict: 
#         """

#         kwargs = {
#             **self.kwargs,
#             **kwarg_override
#         }
#         client = openai.OpenAI()

#         response = client.embeddings.create(
#             model=self.model,
#             input=self.convert(prompt),
#             **kwargs
#         )
#         embeddings = [e for e in response.data]
        
#         return AIResponse(
#             embeddings,
#             response,
#             embeddings
#         )

#     def stream(
#         self, prompt: AIPrompt, 
#         **kwarg_override
#     ) -> typing.Iterator[typing.Tuple[AIResponse, AIResponse]]:

#         response = self.forward(prompt, **kwarg_override)
#         yield response, response
        
#     async def aforward(
#         self, prompt: AIPrompt, 
#         **kwarg_override
#     ) -> typing.Tuple[str, typing.Dict]:
#         client = openai.AsyncOpenAI()

#         kwargs = {
#             **self.kwargs,
#             **kwarg_override
#         }
#         response = client.embeddings.create(
#             model=self.model,
#             input=self.convert(prompt),
#             **kwargs
#         )
#         embeddings = [e for e in response.data]
        
#         return AIResponse(
#             embeddings,
#             response,
#             embeddings
#         )

#     async def astream(self, prompt: AIPrompt, **kwarg_override) -> AsyncIterator[typing.Tuple[AIResponse, AIResponse]]:

#         response = await self.aforward(prompt, **kwarg_override)
#         yield response, response


# class OpenAIEmbeddingModel(AIModel):
#     """Adapter for calling OpenAI's Embedding API"""

#     def __init__(self, api_key: str, model_name: str = 'text-embedding-ada-002'):
#         openai.api_key = api_key
#         self.model_name = model_name

#     def convert(self, message: Message) -> str:
#         """Convert a Message to the format needed for OpenAI's Embedding API"""
#         if isinstance(message, TextMessage):
#             return message.text
#         else:
#             raise TypeError("Unsupported message type")

#     def forward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
#         """Run a query to the OpenAI Embedding API"""
#         # Convert messages and extract text
#         texts_to_embed = [self.convert(message) for message in prompt.aslist()]

#         # Send request to OpenAI Embedding API
#         response = openai.Embedding.create(
#             model=self.model_name,
#             input=texts_to_embed
#         )

#         # Generate AIResponse objects
#         return [
#             AIResponse(message=TextMessage(source="embedding_result", content=text), source={"embedding": data['embedding']})
#             for text, data in zip(texts_to_embed, response['data'])
#         ]

#     async def aforward(self, prompt: AIPrompt, **kwarg_override) -> List[AIResponse]:
#         """Run an asynchronous query to the OpenAI Embedding API"""
#         return self.forward(prompt, **kwarg_override)  # Use sync forward for simplicity


    # def convert(self, message: Message) -> typing.Dict:
    #     """Convert the messages to a format to use for the OpenAI API

    #     Args:
    #         messages (typing.List[Message]): The messages

    #     Returns:
    #         typing.List[typing.Dict]: The input to the API
    #     """
    #     text = message['text']
    #     if isinstance(text, Cue):
    #         text = text.render()
    #     return {
    #         'role': message.role,
    #         'content': text
    #     }

    # def system(self, text: typing.Union[Cue, str]) -> Message:
    #     if isinstance(text, Cue):
    #         return LLMSystemMessage(
    #             cue=cue 
    #         )
    #     return LLMSystemMessage(
    #         text=text
    #     )
    
    # def user(self, text: str) -> Message:
    #     return LLMUserMessage(
    #         text=text, 
    #     )

    # def tool(self) -> Message:
    #     return LLMToolMessage(
            
    #     )

    # def assistant(self) -> Message:
    #     return LLMAssistantMessage(
            
    #     )