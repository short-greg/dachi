from dachi.behavior import Action, SangoStatus
from dachi.struct import Prompt, Conv
from dachi.comm import Query, Request
from ..comm import IOHandler


class AIConvMessage(Action):

    def __init__(
        self, conv: Conv, request: Request, query: Query, role: str='user'
    ):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__()
        self._query = query
        self._conv = conv
        self._request = request
        self._role = role

    def act(self) -> SangoStatus:
        
        self._request.contents = self._conv.as_dict()
        self._query.post(self._request)
        
        if self._request.processed is False:
            return SangoStatus.RUNNING
        if self._request.success is False:
            return SangoStatus.FAILURE
        
        self._conv.add_turn(self._role, self._request.response)
        return SangoStatus.SUCCESS


class UserConvMessage(Action):

    def __init__(
        self, request: Request, query: Query, role: str='user'
    ):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__()
        self._query = query
        self._request = request
        self._role = role

    def act(self) -> SangoStatus:
        
        self._request.contents = self._conv.as_dict()
        self._query.post(self._request)
        
        if self._request.processed is False:
            return SangoStatus.RUNNING
        if self._request.success is False:
            return SangoStatus.FAILURE
        
        self._conv.add_turn(self._role, self._request.response)
        return SangoStatus.SUCCESS


class PreparePrompt(Action):

    def __init__(self, conv: Conv, prompt: Prompt, **components):

        super().__init__()
        self.prompt = prompt 
        self.conv = conv
        self.components = components

    def act(self) -> SangoStatus:

        components = {}
        for k, component in self.components.items():
            components[k] = component
        self.prompt.format(**components).as_text()

        return SangoStatus.SUCCESS


class DisplayAI(Action):

    def __init__(self, conv: Conv, io: IOHandler):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__()
        self._conv = conv
        self._io = io
    
    def reset(self):
        super().reset()
        self._i = 0

    def act(self) -> SangoStatus:
        
        turns = self._conv.filter('assistant')

        if self._i >= len(turns):
            return self.FAILURE
        posted = self._io.post_bot_message(turns[self._i].text)
        if not posted:
            return self.FAILURE
        
        self._i += 1
        return self.SUCCESS
