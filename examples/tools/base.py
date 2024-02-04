from dachi.behavior import Action, SangoStatus
from dachi.storage import Q, DDict, PromptConv, Completion
from dachi.comm import Request
from .queries import UIQuery, LLMQuery
from .comm import UI
import typing
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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


class Converse(Action):

    def __init__(
        self, prompt_conv: PromptConv, llm: LLMQuery, user_interface: UI,
        response_processor: ProcessResponse=None
    ):
        # Use to send a message from the LLM
        super().__init__()
        self._conv = prompt_conv
        self._llm_query = llm
        self._ui_query = UIQuery(user_interface)
        self._ui = user_interface
        self._request = Request()
        self._processed: Processed = None
        self._response_processor = response_processor or NullProcessResponse()

    def converse_turn(self):

        self._request = Request()
        response = self._llm_query(self._conv, asynchronous=False)
        self._processed  = self._response_processor.process(response)

        if self._processed.to_interrupt:
            return
        self._ui.post_message('assistant', self._processed.text)

        self._conv.add_turn(
            'assistant', self._processed.text
        )
        self._request.contents = self._conv
        self._ui_query.post(self._request)

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            # use threading here
            thread = threading.Thread(
                target=self.converse_turn, args=[]
            )
            thread.start()

        if self._processed is not None and (
            self._request.responded is True or self._processed.to_interrupt
        ):
            
            if not self._processed.to_interrupt and self._request.success is True:
                self._conv.add_turn(
                    'user', self._request.response
                )
            if self._processed.succeeded:
                print('SUCCEEDED!')
                return SangoStatus.SUCCESS
            else:
                print('FAILED!')
                return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING

    def reset(self):
        super().reset()
        self._interrupt = False
        self._request = Request()


class PromptCompleter(Action):
    # Use to send a message from the LLM

    def __init__(
        self, completion: Completion, llm: LLMQuery, user_interface: UI,
        post_processor: typing.Callable=None
    ):
        """

        Args:
            completion (Completion): 
            llm (Query): 
            user_interface (UI): 
        """
        super().__init__()
        self._completion = completion
        self._llm_query = llm
        self._ui = user_interface
        self._request = Request()
        self._post_processor = post_processor

    def respond(self):

        self._request = Request()

        response = self._llm_query(self._completion, asynchronous=False)
        self._ui.post_message('assistant', response)
        self._completion.response = response
        print(self._completion.response)
        if self._post_processor is not None:
            self._post_processor()
        self._request.respond(response)

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            # use threading here
            thread = threading.Thread(target=self.respond, args=[])
            thread.start()

        if self._request.responded is True:
            # self._request.status # wouldn't this be easier? 
            if self._request.success is True:
                return SangoStatus.SUCCESS
            else:
                return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING

    def reset(self):
        super().reset()
        self._request = Request()


# TODO: Reconsider the best way to do this
class Transfer(object):

    def __init__(self, q: Q, d: DDict, name: str):

        super().__init__()
        self.q = q
        self.d = d
        self.name = name

    def __call__(self):
        
        val = self.q()
        self.d.set(self.name, val)
        print(self.name, self.d.get(self.name), val)


# class ConvMessage(Action):

#     def __init__(
#         self, conv: Conv, query: Query, role: str='user'
#     ):
#         """

#         Args:
#             name (str): 
#             query (Query): 
#         """
#         super().__init__()
#         self._query = query
#         self._conv = conv
#         self._request = Request()
#         self._role = role

#     def act(self) -> SangoStatus:
        
#         if self._status == SangoStatus.READY:
#             self._request.contents = self._conv
#             self._query.post(self._request)
        
#         if self._request.responded is False:
#             return SangoStatus.RUNNING
#         if self._request.success is False:
#             return SangoStatus.FAILURE
        
#         self._conv.add_turn(self._role, self._request.response)
#         return SangoStatus.SUCCESS
    
#     def reset(self):
#         super().reset()
#         self._request = Request()


# class PreparePrompt(Action):

#     def __init__(
#         self, conv: Conv, prompt: Prompt, 
#         replace: bool=False, **structs: Q
#     ):

#         super().__init__()
#         self.prompt = prompt 
#         self._conv = conv
#         self.structs = structs
#         self._prepared = False
#         self.replace = replace

#     def act(self) -> SangoStatus:

#         if self._prepared and not self.replace:
#             return SangoStatus.SUCCESS
#         structs = {}
#         for k, struct in self.structs.items():
#             structs[k] = struct()
#         if isinstance(self.prompt, R):
#             prompt = self.prompt()
#         else:
#             prompt = self.prompt
#         prompt = prompt.format(**structs, inplace=False)
#         print('Formatted prompt ' + prompt.text)
#         self._prepared = True
#         self._conv.set_system(prompt)
#         return SangoStatus.SUCCESS
    
#     def reset(self):

#         super().__init__()
#         self._prepared = False


# class AdvPrompt(Action):

#     def __init__(self, prompts: DList, default: Prompt, wrapper: Wrapper[Prompt]):

#         super().__init__()
#         self._prompts = DList([*prompts, default])
#         self._default = default
#         self._wrapper = wrapper
#         self._idx = 0
#         self._wrapper.val = self._prompts[0]

#     def act(self) -> SangoStatus:

#         if self._idx >= len(self._prompts) - 1:
#             return SangoStatus.FAILURE
        
#         self._idx += 1
#         self._wrapper.val = self._prompts[self._idx]

#         return SangoStatus.SUCCESS
    
#     def reset(self):

#         super().__init__()
#         self._idx = 0


# # TODO: Improve R <= need to retrieve. Add f(). 
# # TODO: Add a Buffer that is used for this
# class DisplayAI(Action):

#     def __init__(self, conv: Conv, user_interface: UI):
#         """

#         Args:
#             name (str): 
#             query (Query): 
#         """
#         super().__init__()
#         self._conv = conv
#         self._user_interface = user_interface
#         self._cur = None
    
#     def reset(self):
#         super().reset()

#     def act(self) -> SangoStatus:
        
#         turns = self._conv.filter('assistant')
    
#         posted = self._user_interface.post_message('assistant', turns[-1].text)
#         if not posted:

#             return self.FAILURE
        
#         return self.SUCCESS
