from dachi.act import Action, TaskStatus
from dachi.storage import Prompt, Conv, Retrieve
from dachi.comm import Request, Request
from ....tools.comm import UI


class ConvMessage(Action):

    def __init__(
        self, conv: Conv, query: Request, role: str='user'
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

    def act(self) -> TaskStatus:
        
        if self._status == TaskStatus.READY:
            self._request.contents = self._conv
            self._query.post(self._request)
        
        if self._request.responded is False:
            return TaskStatus.RUNNING
        if self._request.success is False:
            return TaskStatus.FAILURE
        
        self._conv.add_turn(self._role, self._request.response)
        return TaskStatus.SUCCESS
    
    def reset(self):
        super().reset()
        self._request = Request()


class PreparePrompt(Action):

    def __init__(self, conv: Conv, prompt: Prompt, replace: bool=False, **components: Retrieve):

        super().__init__()
        self.prompt = prompt 
        self._conv = conv
        self.components = components
        self._prepared = False
        self.replace = replace

    def act(self) -> TaskStatus:

        if self._prepared and not self.replace:
            return TaskStatus.SUCCESS
        components = {}
        for k, component in self.components.items():
            components[k] = component()
        prompt = self.prompt.format(**components, inplace=False)
        self._prepared = True
        self._conv.set_system(prompt)
        return TaskStatus.SUCCESS
    
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

    def act(self) -> TaskStatus:
        
        turns = self._conv.filter('assistant')
    
        posted = self._user_interface.post_message('assistant', turns[-1].text)
        if not posted:

            return self.FAILURE
        
        return self.SUCCESS


# class ChatConv(Conv):

#     def __init__(self, max_turns: int=None):

#         # add introductory message
#         super().__init__(
#             ['system', 'assistant', 'user'], 
#             max_turns, True
#         )
#         self.add_turn('system', None)

#     def set_system(self, prompt: Prompt):

#         self[0].text = prompt.as_text()

#     def reset(self):
#         super().reset()
#         self.add_turn('system', None)
