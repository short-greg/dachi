import typing
from dataclasses import dataclass, field
from typing import Dict
from abc import abstractmethod, ABC
from ..base import Storable
from ._core import DList, Struct, Arg, Q


T = typing.TypeVar("T")


class Text(Struct):
    """A simple wrapper to use text as a prompt struct
    """

    def __init__(self, text: str):
        """Wrap the string 

        Args:
            text (str): The string to wrap
        """
        self.text = text

    def as_dict(self) -> Dict:
        return {
            "text": self.text
        }
    
    def as_text(self, heading: str=None) -> str:
        return self.structure(
            self.text, heading
        )
    
    def spawn(self) -> 'Text':
        return Text(
            self.text
        )

# TODO: Add Example, Add Document

@dataclass
class Role(object):

    name: str
    meta: typing.Dict = field(default_factory=dict)


@dataclass
class Message(Struct):

    role: Role
    text: Struct

    def __post_init__(self):

        if isinstance(self.role, str):
            self.role = Role(self.role)
        if isinstance(self.text, str):
            self.text = Text(self.text)

    def as_dict(self) -> Dict:
        return {
            'role': self.role.name,
            'content': self.text.as_text(),  
        }

    def as_text(self) -> str:
        return f"{self.role.name}: {self.text.as_text()}"


class Prompt(Struct):
    """Define a prompt to send to the LLM
    """

    def __init__(self, args: typing.List[typing.Union[Arg, str]], text: str):
        """Create a prompt with variable arguments

        Args:
            args (typing.List[typing.Union[Arg, str]]): The args for the prompt. If none use []
            text (str): The prompt text
        """
        super().__init__()
        self._args = {}
        for arg in args:
            if isinstance(arg, str):
                self._args[arg] = Arg(arg)
            else:
                self._args[arg.name] = arg
        self._text = text

    def format(self, inplace: bool=False, **kwargs) -> 'Prompt':
        """Format the prompt to remove its variables

        Raises:
            ValueError: If there are arguments passed in that are not 
            arguments to the prompt

        Returns:
            Prompt: The formatted prompt
        """
        input_names = set(kwargs.keys())
        difference = input_names - set(self._args)
        if len(difference) != 0:
            raise ValueError(
                f'Input has names, {input_names}, that are not arguments to the prompt {self._args}'
            )
        inputs = {}
        for k, _ in self._args.items():
            if k in kwargs:
                cur_v = kwargs[k]
                if isinstance(cur_v, Q):
                    cur_v = cur_v()
                if isinstance(cur_v, Struct):
                    cur_v = cur_v.as_text()
                inputs[k] = cur_v
            else:
                inputs[k] = "{" + f'{k}' + "}"
        prompt = Prompt(
            list(set(self._args) - input_names), 
            self._text.format(**inputs)
        )
        if inplace:
            self._args = prompt.args
            self._text = prompt.text
        return prompt
    
    def as_text(self, heading: str=None) -> str:

        return self.structure(self._text, heading)

    def as_dict(self) -> str:

        return {
            "args": self._args,
            "text": self._text
        }
    
    def spawn(self) -> 'Prompt':
        return Prompt(
            [*self._args], self._text
        )
    
    @property
    def text(self) -> str:
        return self._text

    @property
    def args(self) -> typing.List[typing.Union[Arg, str]]:
        return self._args


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

        prompt = self.base_prompt.format(**self.components)

        return (
            f'{prompt.as_text()}\n'
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


class MessageLister(ABC):

    @abstractmethod
    def as_messages(self) -> DList:
        pass


class Conv(Struct, MessageLister):

    def __init__(self, roles: typing.List[typing.Union[str, Role]]=None, max_turns: int=None, check_roles: bool=False):

        super().__init__()
        self._check_roles = check_roles
        roles = roles or []
        self._roles = {}
        for role in roles:
            self.add_role(role)
        self._turns = DList()
        self._max_turns = max_turns

    def add_turn(self, role: str, text: str) -> 'Conv':

        if role not in self._roles:
            self.add_role(role)
        if self._max_turns is not None and len(self._turns) == self._max_turns:
            self._turns = self._turns[1:]
        self._turns.append(Message(self._roles[role], text))

        return self
    
    def add_role(self, role: typing.Union[Role, str], overwrite: bool=False) -> 'Conv':

        if isinstance(role, str):
            role = Role(role)
        if overwrite and role.name in self._roles:
            raise KeyError(f'Role {role.name} already added.')
        if self._check_roles and role.name in self._roles:
            raise KeyError(f'Role {role.name} is an invalid role.')
        self._roles[role.name] = role
        return self
    
    def remap_role(self, **role_map) -> 'Conv':
        """Remap the names of the roles to a new set of roles

        Returns:
            Conversation: The conversation with remapped roles
        """
        roles = [role_map.get(role, role) for role in self._roles]
        turns = [
            (role_map.get(role, role), text)
            for role, text in self._turns
        ]
        conversation = self.__class__(
            roles=roles, max_turns=self._max_turns
        )
        for turn in turns:
            conversation.add_turn(turn[0], turn[1])
        return conversation

    def __getitem__(self, idx) -> Message:

        return self._turns[idx]
    
    def __setitem__(self, idx, turn: Message) -> Message:

        self._turns[idx] = turn
        return turn

    def filter(self, role: str):

        return [
            turn for turn in self._turns if turn.role.name == role]
    
    def to_dict(self) -> typing.Dict[str, str]:

        return {
            i: turn
            for i, turn in enumerate(self._turns)
        }
    
    @property
    def max_turns(self) -> int:
        """
        Returns:
            int: The max turns for the conversation
        """
        return self._max_turns
    
    @max_turns.setter
    def max_turns(self, max_turns) -> int:
        """

        Returns:
            int: _description_
        """
        if max_turns <= 0:
            raise ValueError('')
        self._max_turns = max_turns
        return self._max_turns
    
    def reset(self):
        """Clear the turns
        """
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
    
    def __iter__(self) -> typing.Tuple[Role, Message]:

        for turn in self._turns:
            yield turn.role, turn.text

    def as_dict(self) -> Dict:
        return {
            'roles': self._roles,
            'turns': self._turns.as_dict(),
            'max_turns': self._max_turns   
        }

    def as_text(self, heading: str=None) -> str:
        
        result = f'{heading}\n' if heading is not None else ''
        for i, turn in enumerate(self._turns):
            result += f'{turn.as_text()}'
            if i < len(self._turns) - 1:
                result += '\n'

        return result

    def spawn(self) -> 'Conv':

        return Conv(
            {**self._roles}, self._max_turns
        )

    @property
    def turns(self) -> typing.List[Message]:
        return self._turns
    
    def as_messages(self) -> DList[Message]:

        return DList(self._turns)
    
    # def range(self, from_: int=0, to_: int=-1) -> 'DList':

    #     return DList(
            
    #     )


class PromptConv(Conv):

    def __init__(self, prompt_gen: PromptGen, max_turns: int=None):

        # add introductory message
        super().__init__(
            ['assistant', 'user'], 
            max_turns, True
        )
        self.prompt_gen = prompt_gen

    def as_turns(self) -> typing.List[typing.Dict[str, str]]:

        return super().as_messages()

    def as_messages(self) -> typing.List[typing.Dict[str, str]]:

        return DList([
            Message('system', self.prompt_gen),
            *super().as_messages()
        ])

    def as_text(self, heading: str=None) -> str:
        
        result = f'{heading}\n' if heading is not None else ''
        result += self.prompt_gen.as_text()
        for i, turn in enumerate(self._turns):
            result += f'{turn.as_text()}'
            if i < len(self._turns) - 1:
                result += '\n'

        return result

    def spawn(self) -> 'PromptConv':

        return PromptConv(
            self.prompt_gen.spawn(),
            self._max_turns
        )

    def as_dict(self) -> Dict:
        return {
            'prompt_gen': self.prompt_gen.as_dict(),
            **super().as_dict()
        }


class Completion(Struct, MessageLister):
    """
    """
    
    def __init__(self, prompt: 'PromptGen', response: str=None, prompt_name: str='system'):
        """
        Args:
            prompt (PromptGen): The prompt for the completion
            response (str): The response for the completion
        """
        self.prompt_gen = prompt
        self.response = response
        self.prompt_name = prompt_name

    @property
    def prompt(self) -> 'prompt':
        return self.prompt_gen.prompt

    def as_text(
        self, 
        heading: str=None,
        prompt_heading: str="===Prompt===", 
        response_heading: str="===Response==="
    ) -> str:
        """
        Returns:
            typing.Dict: The completion object as a dict
        """
        body = f"""
        {self.structure(self.prompt_gen.as_text(), prompt_heading)}
        {self.structure(self._response, response_heading)}
        """
        return self.structure(
            body, heading
        )
    
    def as_messages(self) -> DList:

        return DList([Message('system', self.prompt_gen)])
    
    def as_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The completion object as a dict
        """
        return {
            "prompt": self.prompt_gen.prompt,
            "response": self.response
        }

    def spawn(self) -> 'Completion':
        return Completion(
            self.prompt_gen.spawn(), self.response
        )



# class MessageList(Struct, typing.List[Message]):

#     def __init__(self, messages: typing.List[Message]=None):

#         super().__init__()
#         if messages is not None:
#             self.extend(messages)

#     def filter(self, role: str):

#         return MessageList([
#             message for message in self._messages if message.role.name == role]
#         )
    
#     def reset(self):
#         """Clear the turns
#         """
#         self.clear()
    
#     def as_dict(self) -> Dict:
#         return {
#             i: turn
#             for i, turn in enumerate(self)
#         }

#     def as_text(self, heading: str=None) -> str:
        
#         result = f'{heading}\n' if heading is not None else ''
#         for i, message in enumerate(self._messages):
#             result += f'{message.as_text()}'
#             if i < len(self._turns) - 1:
#                 result += '\n'

#         return result

#     def spawn(self) -> 'MessageList':

#         return MessageList()



