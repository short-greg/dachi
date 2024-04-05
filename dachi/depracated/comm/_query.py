from openai import OpenAI
from openai import AsyncOpenAI
from abc import abstractmethod, ABC
from typing import Any
# from ._ui import UI
# from ..storage import MessageLister
# from functools import partial
import threading
import typing
import datetime
from functools import partial
# from ._requests import CALLBACK
from enum import Enum
from ._ui import UI
from dataclasses import field, dataclass


class RequestStatus(Enum):
    '''
    Use for defining the status of the request
    '''

    READY = 'ready'
    PROCESSING = 'processing'
    SUCCESS = 'success'
    FAILURE = 'failure'

    def is_done(self) -> bool:
        """
        Returns:
            bool: If the request has completed
        """
        return self == RequestStatus.SUCCESS or self == RequestStatus.FAILURE

    @property
    def success(self) -> bool:
        """
        Returns:
            bool: If the request was successful
        """
        return self == RequestStatus.SUCCESS

    @property
    def failure(self) -> bool:
        """
        Returns:
            bool: If the request failed
        """
        return self == RequestStatus.FAILURE

    @property
    def ready(self) -> bool:
        """
        Returns:
            bool: If the request is ready
        """
        return self == RequestStatus.READY

    @property
    def processing(self) -> bool:
        """
        Returns:
            bool: If the request is being processed
        """
        return self == RequestStatus.PROCESSING
    
    def complete(self, success: bool):

        if success is True:
            return RequestStatus.SUCCESS
        return RequestStatus.FAILURE


@dataclass
class Post:

    contents: typing.Any = None
    on_response: typing.Callable = None

    def __post_init__(self):

        self._status: RequestStatus = RequestStatus.READY
        self._response = None
        self._responded_at = None

    @property
    def response(self):
        return self._response
    
    @property
    def responded_at(self):
        return self._responded_at
    
    @response.setter
    def response(self, response):
        if self.status == RequestStatus.READY:
            self._status = RequestStatus.PROCESSING
        if self.status.is_done():
            return
    
        self._response = response
        self._responded_at = datetime.datetime.now()
        return response

    @property
    def status(self) -> 'RequestStatus':
        return self._status

    def request(self, request: 'Request'):
        
        self._status = self.status.complete(
            request.post(self)
        )

    async def async_request(self, request: 'Request'):
        self._status = self.status.complete(
            await request.async_post(self)
        )

    def threaded_request(self, request: 'Request'):
        self._status = self.status.complete(
            request.post_worker(self)
        )


class Request(ABC):
    """Base class for creating a message that will return a response
    after completion
    """
    def post_worker(self, post: Post) -> Post:
        thread = threading.Thread(target=self.post, args=[post])
        thread.start()          
        return post

    @abstractmethod
    def post(self, post: Post):
        pass

    async def async_post(self, post: Post):
        return self.post(post)


class OpenAIRequest(Request):

    def __init__(self, temperature: float=0.0, stream: bool=False):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.client = OpenAI()
        self.aclient = AsyncOpenAI()
        self.temperature = temperature
        self.stream = stream

    def _update(self, post: Post, response):

        if self.stream:
            for chunk in response:
                # improve this
                post.response += chunk.choices[0].delta.content
        else:
            post.response = response.choices[0].message.content

    def post(self, post: Post, stream_override: bool=None):

        stream = self.stream if stream_override is None else stream_override
        contents = post.content.as_messages().as_dict_list()

        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=contents, temperature=self.temperature,
            stream=stream
        )
        self._update(post, response)

    async def async_post(self, post: Post, stream_override: bool=False):
        
        stream = self.stream if stream_override is None else stream_override
        contents = post.content.as_messages().as_dict_list()

        response = await self.aclient.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=contents, temperature=self.temperature,
            stream=stream
        )
        self._update(post, response)


class UIQuery(Request):

    def __init__(self, ui_interface: UI):
        """

        Args:
            store (DataStore): 
        """
        super().__init__()
        self.ui_interface = ui_interface

    def respond(self, post: Post):
        pass

    def post(self, post: Post):
        self.ui_interface.request_message(partial(self.respond, post))



# class LLMQuery(Request):

#     @abstractmethod
#     def prepare_response(self, request: Request):
#         pass
    
#     def exec_post(self, request: Request) -> Any:
#         message = self.prepare_response(request)
#         self.respond(request, message)

#     def __call__(self, conv: MessageLister, asynchronous: bool=True):
#         request = Request(contents=conv.as_messages().as_dict_list())
#         self.post(request, asynchronous)
#         return request.response


# class LLMQUery2(Query2):

#     def __init__(self, temperature: float=0.0, stream: bool=False) -> None:
#         super().__init__()
#         self.temperature = temperature
#         self.stream = stream

#     def query(self, request: Request, callback: CALLBACK=None):

#         client = OpenAI()
#         response = client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             messages=request.contents, temperature=self.temperature,
#             stream=self.stream
#         )
#         if self.stream:
#             for chunk in response:
#                 # improve this
#                 request.response += chunk.choices[0].delta.content
#                 request.responded_at = datetime.now()
#         else:
#             request.response = response.choices[0].message.content
#             request.responded_at = datetime.now()
#         request.processed = True

#     async def query(self, request: Request):
        
#         client = AsyncOpenAI()
#         response = await client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             messages=request.contents, temperature=self.temperature,
#             stream=self.stream
#         )
        
#         if self.stream:
#             for chunk in response:
#                 request.response += chunk.choices[0].delta.content
#                 request.responded_at = datetime.now()
#         else:
#             request.response = response.choices[0].message.content
#             request.responded_at = datetime.now()
#         request.processed = True


# class Query
#   # async def __async_call__()
#   # def __call__(  callback=None) # if not called will call thread




# 1) async query
# 2) thread query => returns a response. can pass in a callback
# 3) normal => 



# TODO: Update SIGNAL

# class Signal(ABC):
#     """A message that does not respond to the user
#     """

#     def __init__(self):
#         """
#         """
#         self._on_post = []
    
#     @abstractmethod
#     def prepare_post(
#         self, request: Request,
#         on_post: CALLBACK=None, 
#     ):
#         """Sends a message. If the content cannot be processed it raises an ValueError

#         Args:
#             contents: The contents of the message
#         """
#         raise NotImplementedError

#     def post(
#         self, request: Request, asynchronous: bool=False
#     ):
#         """Sends a message. If the content cannot be processed it raises an ValueError

#         Args:
#             contents: The contents of the message
#         """
#         self.prepare_post(request)

#         request.post()
#         for on_post in self._on_post:
#             on_post(request)

#     def register(
#         self, 
#         on_post: typing.Union[str, typing.Callable]
#     ):
#         self._on_post.append(on_post)

#     def unregister(
#         self, 
#         on_post: typing.Union[str, typing.Callable]
#     ):
#         self._on_post.remove(on_post)


# class Query2(ABC):

#     @abstractmethod
#     def exec(self, request: Request, callback: CALLBACK=None):
#         pass

#     @abstractmethod
#     async def aexec(self, request: Request):
#         pass

#     def post(self, request: Request, callback: CALLBACK=None, use_thread: bool=False):
        
#         if use_thread:
#             thread = threading.Thread(target=self.exec_post, args=[request, callback])
#             thread.start()
#         else:
#             self.exec_post(request, callback)

#     async def apost(self, request: Request):
        
#         return await self.aexec(request)

