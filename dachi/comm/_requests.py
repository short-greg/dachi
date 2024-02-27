import typing
from abc import ABC, abstractmethod
import uuid
from dataclasses import dataclass
from enum import Enum
from dataclasses import field, dataclass

CALLBACK = typing.Union[str, typing.Callable]


@dataclass
class Request(object):
    """Make a request to a query
    """

    contents: typing.Any = None
    on_post: typing.Callable = None
    on_response: typing.Callable = None

    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self._posted: bool = False
        self._responded: bool = False
        self._response = None
        self._success = False

    def post(self):
        """Post the request
        """
        if self._posted:
            return

        self._posted = True
        if self.on_post is not None:
            self.on_post(self)

    @property
    def posted(self) -> bool:

        return self._posted
    
    def respond(self, response, success: bool=True):

        if self.responded:
            raise ValueError('Request has already been processed')

        self._response = response
        self._success = success
        self._responded = True
        if self.on_response is not None:
            self.on_response(self)

    @property
    def processing(self) -> bool:
        return self._processing

    @property
    def response(self) -> typing.Any:

        return self._response

    @property
    def responded(self) -> bool:

        return self._responded
    
    @property
    def success(self) -> bool:

        return self._success


@dataclass
class Processed:

    text: str
    succeeded: bool
    to_interrupt: bool = field(default=False)
    other: typing.Any = field(default_factory=dict)


class ProcessResponse(ABC):

    @abstractmethod
    def process(self, content) -> Processed:
        pass


class NullProcessResponse(ProcessResponse):

    def process(self, content) -> Processed:

        return Processed(
            content, True, False
        )


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



@dataclass
class Request2(object):
    """Make a request to a query
    """

    contents: typing.Any = None
    status: RequestStatus = field(default=RequestStatus.READY)
    on_response: typing.Callable = None

    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self._response = None
        self._success = False
    
    def process(self):

        if not self.status.is_done():
            self.status = RequestStatus.PROCESSING
    
    def respond(self, response, success: bool=True):

        if self.responded:
            raise ValueError('Request has already been processed')

        if success:
            self.status = RequestStatus.SUCCESS
        else:
            self.status = RequestStatus.FAILURE
        self._response = response
        if self.on_response is not None:
            self.on_response(self)

    @property
    def processing(self) -> bool:
        return self.status == RequestStatus.PROCESSING

    @property
    def response(self) -> typing.Any:

        return self._response

    @property
    def responded(self) -> bool:

        return self.status.is_done()
    
    @property
    def success(self) -> bool:

        return self.status.success

