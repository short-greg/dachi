from dachi.behavior import Action, SangoStatus
from dachi.struct import Prompt, Conv, Q
from dachi.comm import Query, Request
from ..comm import UI


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

    def __init__(self, conv: Conv, prompt: Prompt, replace: bool=False, **components: Q):

        super().__init__()
        self.prompt = prompt 
        self._conv = conv
        self.components = components
        self._prepared = False
        self.replace = replace

    def act(self) -> SangoStatus:

        if self._prepared and not self.replace:
            return SangoStatus.SUCCESS
        components = {}
        for k, component in self.components.items():
            components[k] = component()
        prompt = self.prompt.format(**components, inplace=False)
        self._prepared = True
        self._conv.set_system(prompt)
        return SangoStatus.SUCCESS
    
    def reset(self):

        super().__init__()
        self._prepared = False

# TODO: Improve R <= need to retrieve. Add f(). 
# TODO: Add a Buffer that is used for this
class DisplayAI(Action):

    def __init__(self, conv: Conv, user_interface: UI):
        """

        Args:
            name (str): 
            query (Query): 
        """
        super().__init__()
        self._conv = conv
        self._user_interface = user_interface
        self._cur = None
    
    def reset(self):
        super().reset()

    def act(self) -> SangoStatus:
        
        turns = self._conv.filter('assistant')
    
        posted = self._user_interface.post_message('assistant', turns[-1].text)
        if not posted:
            return self.FAILURE
        
        return self.SUCCESS
