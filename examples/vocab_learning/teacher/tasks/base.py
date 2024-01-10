from dachi.behavior import Action, SangoStatus
from dachi.struct import Prompt, Conv
from dachi.comm import Query, Request
from ..comm import UIInterface


class ChatConv(Conv):

    def __init__(self, max_turns: int=None):

        # add introductory message
        super().__init__(
            ['system', 'assistant', 'user'], 
            max_turns, True
        )
        self.add_turn('system', None)

    def set_system(self, prompt: Prompt):

        self[0].text = prompt.as_text()

    def reset(self):
        super().reset()
        self.add_turn('system', None)


class ConvMessage(Action):

    def __init__(
        self, conv: Conv, query: Query, role: str='user'
    ):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__()
        self._query = query
        self._conv = conv
        self._request = Request()
        self._role = role

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            self._request.contents = self._conv.as_dict()
            self._query.post(self._request)
        
        if self._request.responded is False:
            return SangoStatus.RUNNING
        if self._request.success is False:
            return SangoStatus.FAILURE
        
        self._conv.add_turn(self._role, self._request.response)
        return SangoStatus.SUCCESS


class PreparePrompt(Action):

    def __init__(self, conv: Conv, prompt: Prompt, **components):

        super().__init__()
        self.prompt = prompt 
        self._conv = conv
        self.components = components
        self._prepared = False

    def act(self) -> SangoStatus:

        if self._prepared:
            return SangoStatus.SUCCESS
        components = {}
        for k, component in self.components.items():
            components[k] = component
        prompt = self.prompt.format(**components, inplace=False)
        self._prepared = True
        self._conv.set_system(prompt)
        return SangoStatus.SUCCESS
    
    def reset(self):

        super().__init__()
        self._prepared = False


class DisplayAI(Action):

    def __init__(self, conv: Conv, user_interface: UIInterface):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__()
        self._conv = conv
        self._user_interface = user_interface
        self._i = 0
    
    def reset(self):
        super().reset()

    def act(self) -> SangoStatus:
        
        turns = self._conv.filter('assistant')
        if self._i >= len(turns):
            return self.FAILURE
        posted = self._user_interface.post_message('assistant', turns[self._i].text)
        if not posted:
            return self.FAILURE
        
        self._i += 1
        return self.SUCCESS
