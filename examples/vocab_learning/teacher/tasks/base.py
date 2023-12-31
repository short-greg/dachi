from dachi.behavior import Action, SangoStatus
from dachi.gengo import Prompt, PromptComponent
from dachi.comm import Terminal, Ref, Query, Request
from ..comm import IOHandler
import typing


class UIConvMessage(Action):

    def __init__(self, conv: Ref, request: Ref, query: Query, name: str=None):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__(name)
        self._query = query
        self._conv = conv
        self._request = request

    def process_response(self, request: Request, terminal: Terminal):

        self._request.set(request)
        terminal.storage.clear('request')
    
    def act(self, terminal: Terminal) -> SangoStatus:
        
        request = terminal.storage.get('request')
        if request is None:
            request = terminal.storage['request'] = Request()
            self._query.post(request)
        
        if request.processed is False:
            return SangoStatus.RUNNING
        if request.success is False:
            return SangoStatus.FAILURE
        
        conv = self._conv.get(terminal)
        conv.add_turn(role='user', message=request.contents)

        return self.process_response(request, terminal)


class AIConvMessage(Action):

    def __init__(
        self, conv: Ref, request: Ref, 
        query: Query, name: str=None
    ):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__(name)
        self._query = query
        self._conv = conv
        self._request = request
    
    def process_response(self, request: Request, terminal: Terminal):

        self._request.set(request)
        terminal.storage.clear('request')

    def act(self, terminal: Terminal) -> SangoStatus:
        
        request = terminal.storage.get('request')
        if request is None:
            request = terminal.storage['request'] = Request(
                contents=self._conv.get(terminal)
            )
            self._query.post(request)
        
        if request.processed is False:
            return SangoStatus.RUNNING
        if request.success is False:
            return SangoStatus.FAILURE
        
        return self.process_response(request, terminal)


class PreparePrompt(Action):

    def __init__(self, conv: Ref, prompt: Prompt, components: typing.Dict[str, PromptComponent]=None, name: str=None):

        super().__init__(name)
        self.prompt = prompt 
        self.conv = conv
        self.components = components or {}

    def act(self, terminal: Terminal) -> SangoStatus:

        conv = self.conv.get(terminal)

        if conv.empty():
            
            components = {}
            for k, component in self.components.items():
                if isinstance(component, Ref):
                    component = component.get(terminal)
                components[k] = component
            self.prompt.set(
                terminal, self.prompt.format(**components).as_text()
            )

        return SangoStatus.SUCCESS


class Display(Action):

    def __init__(self, message: Ref, io: IOHandler, name: str=None):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__(name)
        self._message = message
        self._io = io

    def act(self, terminal: Terminal) -> SangoStatus:
        
        message = self._message.get(terminal)
        if message is None:
            return self.FAILURE
        
        posted = self._io.post_bot_message(message)
        if not posted:
            return self.FAILURE
        
        return self.SUCCESS
