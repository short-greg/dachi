from dachi.behavior import Action, SangoStatus
from dachi.storage import Prompt, Conv, Q, StoreList, Wrapper, R, Struct
from dachi.comm import Query, Request
from .queries import UIQuery, LLMQuery
from .comm import UI
import typing
import threading


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


# TODO: Add 
class PromptGen(Struct):

    def __init__(self, prompt: Prompt, **components):

        super().__init__()
        self.base_prompt = prompt
        self.components = components

    @property
    def prompt(self) -> Prompt:

        return self.base_prompt.format(
            **self.components
        )

    # TODO: Complete
    def as_text(self) -> str:
        return (
            f'{self.base_prompt.as_text()}\n'
        )

    def as_dict(self) -> typing.Dict:
        
        components = {}
        for k, v in self.components:
            if isinstance(v, Struct):
                components[k] = v.as_dict()
            else:
                components[k] = v
        return {
            'base_prompt': self.prompt.as_dict(),
            'components': components
        }


class Converse(Action):

    def __init__(
        self, prompt: PromptGen, conv: Conv, llm: LLMQuery, user_interface: UI
    ):
        # Use to send a message from the LLM
        
        super().__init__()
        self._conv = conv
        self._prompt = prompt
        self._llm_query = llm
        self._ui_query = UIQuery(user_interface)
        self._ui = user_interface
        self._request = Request()

    def converse_turn(self):

        self._conv[0].text = self._prompt.prompt.as_text()
        response = self._llm_query(self._conv, asynchronous=False)
        self._ui.post_message('assistant', response)
        self._conv.add_turn(
            'assistant', response
        )
        self._request.contents = self._conv
        self._ui_query.post(self._request)

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            # use threading here
            thread = threading.Thread(target=self.converse_turn, args=[])
            thread.start()

        if self._request.responded is True:
            if self._request.success is True:
                self._conv.add_turn(
                    'user', self._request.response
                )
                return SangoStatus.SUCCESS
            else:
                return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING

    def reset(self):
        super().reset()
        self._request = Request()


class Message(Action):
    # Use to send a message from the LLM

    def __init__(
        self, prompt: PromptGen, conv: Conv, llm: Query, user_interface: UI
    ):
        
        super().__init__()
        self._conv = conv
        self._prompt = prompt
        self._llm_query = llm
        self._ui = user_interface
        self._request = Request()

    def converse_turn(self):

        self._conv[0].text = self._prompt.prompt.as_text()
        response = self._llm_query(self._conv, asynchronous=False)
        self._ui.post_message('assistant', response)
        self._conv.add_turn(
            'assistant', response
        )

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            # use threading here
            thread = threading.Thread(target=self.converse_turn, args=[])
            thread.start()

        if self._request.responded is True:
            if self._request.success is True:
                self._conv.add_turn(
                    'user', self._request.response
                )
                return SangoStatus.SUCCESS
            else:
                return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING

    def reset(self):
        super().reset()
        self._request = Request()


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
            self._request.contents = self._conv
            self._query.post(self._request)
        
        if self._request.responded is False:
            return SangoStatus.RUNNING
        if self._request.success is False:
            return SangoStatus.FAILURE
        
        self._conv.add_turn(self._role, self._request.response)
        return SangoStatus.SUCCESS
    
    def reset(self):
        super().reset()
        self._request = Request()


class PreparePrompt(Action):

    def __init__(self, conv: Conv, prompt: Prompt, replace: bool=False, **structs: Q):

        super().__init__()
        self.prompt = prompt 
        self._conv = conv
        self.structs = structs
        self._prepared = False
        self.replace = replace

    def act(self) -> SangoStatus:

        if self._prepared and not self.replace:
            return SangoStatus.SUCCESS
        structs = {}
        for k, struct in self.structs.items():
            structs[k] = struct()
        if isinstance(self.prompt, R):
            prompt = self.prompt()
        else:
            prompt = self.prompt
        prompt = prompt.format(**structs, inplace=False)
        print('Formatted prompt ' + prompt.text)
        self._prepared = True
        self._conv.set_system(prompt)
        return SangoStatus.SUCCESS
    
    def reset(self):

        super().__init__()
        self._prepared = False


class AdvPrompt(Action):

    def __init__(self, prompts: StoreList, default: Prompt, wrapper: Wrapper[Prompt]):

        super().__init__()
        self._prompts = StoreList([*prompts, default])
        self._default = default
        self._wrapper = wrapper
        self._idx = 0
        self._wrapper.val = self._prompts[0]

    def act(self) -> SangoStatus:

        if self._idx >= len(self._prompts) - 1:
            return SangoStatus.FAILURE
        
        self._idx += 1
        self._wrapper.val = self._prompts[self._idx]

        return SangoStatus.SUCCESS
    
    def reset(self):

        super().__init__()
        self._idx = 0


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
