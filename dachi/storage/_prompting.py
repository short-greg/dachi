import typing
from dataclasses import dataclass, field
from typing import Dict
from abc import abstractmethod, ABC
from ..base import Storable
from ._core import DList, Struct, Arg, Retrieve


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
                if isinstance(cur_v, Retrieve):
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
    
    def gen(self, **components) -> 'PromptGen':
        return PromptGen(
            self, **components
        )


# TODO: Add 
class PromptGen(Struct):

    def __init__(self, prompt: Prompt, **components):

        super().__init__()
        self.base_prompt = prompt
        self.components = components
    
    def with_components(self, **components):

        components = {
            **self.components,
            **components
        }
        return PromptGen(
            self.base_prompt, **components
        )

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
        """

        Args:
            roles (typing.List[typing.Union[str, Role]], optional): The roles in the conversation. Defaults to None.
            max_turns (int, optional): The max number of turns to record. Defaults to None.
            check_roles (bool, optional): Whether to check the roles. Defaults to False.
        """
        super().__init__()
        self._check_roles = check_roles
        roles = roles or []
        self._roles = {}
        for role in roles:
            self.add_role(role)
        self._turns = DList()
        self._max_turns = max_turns

    def add_turn(self, role: str, text: str) -> 'Conv':
        """Add a turn to the conversation

        Args:
            role (str): The role for a turn
            text (str): The text for the turn

        Returns:
            Conv: The conversation with updated turns
        """
        if role not in self._roles:
            self.add_role(role)
        if self._max_turns is not None and len(self._turns) == self._max_turns:
            self._turns = self._turns[1:]
        self._turns.append(Message(self._roles[role], text))

        return self
    
    def add_role(self, role: typing.Union[Role, str], overwrite: bool=False) -> 'Conv':
        """

        Args:
            role (typing.Union[Role, str]): The role to add
            overwrite (bool, optional): Whether to ovewrite the role. Defaults to False.

        Returns:
            Conv: The conversation with a new role
        """
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
        """
        Args:
            idx (int): The index of the message to get

        Returns:
            Message: The message
        """
        return self._turns[idx]
    
    def __setitem__(self, idx, turn: Message) -> Message:
        """
        Args:
            idx: The index for the message
            turn (Message): The turn to add

        Returns:
            Message: The added turn
        """
        self._turns[idx] = turn
        return turn

    def filter(self, role: str) -> typing.List[Message]:
        """Filter by a role

        Args:
            role (str): the string to filter by

        Returns:
            typing.List[Message]: The 
        """
        return [
            turn for turn in self._turns if turn.role.name == role
        ]
    
    def to_dict(self) -> typing.Dict[str, str]:
        """
        Returns:
            typing.Dict[str, str]: Convert the Conversation to a dict
        """
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
    
    def __iter__(self) -> typing.Iterator[typing.Tuple[Role, Message]]:
        """
        Yields:
            Iterator[typing.Iterator[typing.Tuple[Role, Message]]]: Iterate over each message
        """
        for turn in self._turns:
            yield turn.role, turn.text

    def as_dict(self) -> Dict:
        """
        Returns:
            Dict: The conversation as a dictionary
        """
        return {
            'roles': self._roles,
            'turns': self._turns.as_dict(),
            'max_turns': self._max_turns   
        }

    def as_text(self, heading: str=None) -> str:
        """
        Args:
            heading (str, optional): The heading of the conversation. Defaults to None.

        Returns:
            str: The conversation as text
        """
        result = f'{heading}\n' if heading is not None else ''
        for i, turn in enumerate(self._turns):
            result += f'{turn.as_text()}'
            if i < len(self._turns) - 1:
                result += '\n'

        return result

    def spawn(self) -> 'Conv':
        """
        Returns:
            Conv: The spawned conversation
        """
        return Conv(
            {**self._roles}, self._max_turns
        )

    @property
    def turns(self) -> typing.List[Message]:
        """
        Returns:
            typing.List[Message]: The turns for the conversation
        """
        return self._turns
    
    def as_messages(self) -> DList[Message]:
        """
        Returns:
            DList[Message]: The conversation as a list of messages
        """
        return DList(self._turns)


class PromptConv(Conv):

    def __init__(self, prompt_gen: PromptGen, max_turns: int=None):
        """Create a conversation with a system prompt 

        Args:
            prompt_gen (PromptGen): The generator for the prompt
            max_turns (int, optional): The number of turns in the conversation. Defaults to None.
        """
        # add introductory message
        super().__init__(
            ['assistant', 'user'], 
            max_turns, True
        )
        self.prompt_gen = prompt_gen

    def with_components(self, reset_turns: bool=False, **components: Struct) -> 'PromptConv':
        """
        Args:
            reset_turns (bool, optional): Reset the turns for the conversation. Defaults to False.

        Returns:
            PromptConv: The conversation updated with the components passed in
        """
        prompt_conv = PromptConv(self.prompt_gen.with_components(
            **components
        ), self.max_turns)
        if not reset_turns:
            prompt_conv._turns = DList([*self.turns])
        return prompt_conv

    def as_turns(self) -> typing.List[typing.Dict[str, str]]:
        """
        Returns:
            typing.List[typing.Dict[str, str]]: Convert to a list of messages
        """
        return super().as_messages()

    def as_messages(self) -> typing.List[typing.Dict[str, str]]:
        """
        Returns:
            typing.List[typing.Dict[str, str]]: The 
        """
        return DList([
            Message('system', self.prompt_gen),
            *super().as_messages()
        ])

    def as_text(self, heading: str=None) -> str:
        """
        Args:
            heading (str, optional): The heading for text. Defaults to None.

        Returns:
            str: The PromptConv as text
        """
        result = f'{heading}\n' if heading is not None else ''
        result += self.prompt_gen.as_text()
        for i, turn in enumerate(self._turns):
            result += f'{turn.as_text()}'
            if i < len(self._turns) - 1:
                result += '\n'

        return result

    def spawn(self) -> 'PromptConv':
        """
        Returns:
            PromptConv: The PromptConv spawned
        """
        return PromptConv(
            self.prompt_gen.spawn(),
            self._max_turns
        )

    def as_dict(self) -> Dict:
        """
        Returns:
            Dict: Convert the prompt conv to a dict
        """
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

    def with_components(self, **components: Struct) -> 'Completion':

        return Completion(self.prompt_gen.with_components(
            **components
        ), prompt_name=self.prompt_name)

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
        """
        Returns:
            DList: Return the Completion as a list of messages
        """
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
        """
        Returns:
            Completion: Spawn the completion
        """
        return Completion(
            self.prompt_gen.spawn(), self.response
        )
