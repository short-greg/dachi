import typing
from dataclasses import dataclass, field
from typing import Dict
from abc import abstractmethod, ABC
from ..base import Storable


T = typing.TypeVar("T")

@dataclass
class Arg:

    name: str
    description: str = field(default="")


class Q(ABC, typing.Generic[T]):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T:
        raise NotImplementedError


class D(Q, typing.Generic[T]):

    def __init__(self, data: 'Struct'):

        self._data = data

    def __call__(self) -> T:

        return self._data


class F(Q, typing.Generic[T]):

    def __init__(self, f, *args, **kwargs):

        self._f = f
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, *args, **kwargs) -> T:

        kwargs = {
            **self._kwargs,
            **kwargs
        }
        args = [*args, *self._args]
        return self._f(*args, **kwargs)


class R(Q, typing.Generic[T]):

    def __init__(self, data: 'Struct', name: str):
        
        self._data = data
        self._name = name

    def __call__(self) -> T:

        return self._data.get(self._name)


class Struct(Storable):

    @abstractmethod
    def as_text(self) -> str:
        pass

    @abstractmethod
    def as_dict(self) -> typing.Dict:
        pass

    @staticmethod
    def structure(text: str, heading: str=None, empty: float='Undefined'):

        text = text if text is not None else empty

        if heading is None:
            return f'{text}'
        
        return (
            f'{heading}\n'
            f'{text}'
        )

    def get(self, name: str) -> typing.Any:
        return getattr(self, name)

    def r(self, name: str) -> R:
        
        return R(self, name)

    @property
    def d(self) -> D:
        
        return D(self)
    
    def f(self, name: str, *args, **kwargs) -> F:
        return F(getattr(self, name), *args, **kwargs)

    def set(self, name: str, value):
        if name not in self.__dict__:
            raise KeyError(f'Key by name of {name} does not exist.')
        self.__dict__[name] = value
        return value

    def reset(self):
        pass


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
                f'Input has names that are not arguments to the prompt'
            )
        inputs = {}
        for k, _ in self._args.items():
            if k in kwargs:
                cur_v = kwargs[k]
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


class Completion(Struct):
    """
    """
    
    def __init__(self, prompt: Prompt, response: str=None):
        """
        Args:
            prompt (Prompt): The prompt for the completion
            response (str): The response for the completion
        """
        self.prompt = prompt
        self.response = response

    def format_prompt(self, **kwargs) -> 'Completion':
        """Format the prompt to remove its variables

        Raises:
            ValueError: If there are arguments passed in that are not 
            arguments to the prompt

        Returns:
            Prompt: The formatted prompt
        """
        return Completion(
            self.prompt.format(**kwargs)
        )

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
        {self.structure(self.prompt.as_text(), prompt_heading)}
        {self.structure(self.response, response_heading)}
        """
        return self.structure(
            body, heading
        )
    
    def as_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The completion object as a dict
        """
        return {
            "prompt": self.prompt.as_dict(),
            "response": self.response
        }

    def spawn(self) -> 'Completion':
        return Completion(
            self.prompt.spawn(), self.response
        )


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
class Turn(Struct):

    role: Role
    text: str

    def as_dict(self) -> Dict:
        return {
            'role': self.role.name,
            'text': self.text,  
        }

    def as_text(self) -> str:
        return f"{self.role.name}: {self.text}"


class Conv(Struct):

    def __init__(self, roles: typing.List[typing.Union[str, Role]]=None, max_turns: int=None, check_roles: bool=False):

        super().__init__()
        self._check_roles = check_roles
        roles = roles or []
        self._roles = {}
        for role in roles:
            self.add_role(role)
        self._turns: typing.List[Turn] = DDict()
        self._max_turns = max_turns

    def add_turn(self, role: str, text: str) -> 'Conv':

        if role not in self._roles:
            self.add_role(role)
        if self._max_turns is not None and len(self._turns) == self._max_turns:
            self._turns = self._turns[1:]
        self._turns.append(Turn(self._roles[role], text))

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

    def __getitem__(self, idx) -> Turn:

        return self._turns[idx]
    
    def __setitem__(self, idx, turn: Turn) -> Turn:

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
        """_summary_

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
    
    def __iter__(self) -> typing.Tuple[Role, str]:

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
    def turns(self) -> typing.List[Turn]:
        return self._turns
    
    def as_messages(self) -> typing.List[typing.Dict[str, str]]:

        return [
            {'role': turn.role.name, 'content': turn.text}
            for turn in self._turns
        ]
    
    def range(self, from_: int=0, to_: int=-1) -> 'DDict':

        return DDict(
            
        )


class DDict(typing.List, Struct):
    
    def as_dict(self) -> Dict:
        return dict(enumerate(self))

    def as_text(self, heading: str=None) -> str:
        
        text = '\n'.join(self)
        if heading is not None:
            return (
                f'{heading}'
                f'{text}'
            )
        return text
    
    def spawn(self) -> 'DDict':

        return DDict(
            [x.spawn() if isinstance(x, Storable) else x for x in self]
        )

    def load_state_dict(self, state_dict: typing.Dict):
        """

        Args:
            state_dict (typing.Dict): 
        """
        for i, v in enumerate(self):
            if isinstance(v, Storable):
                self[i] = v.load_state_dict(state_dict[i])
            else:
                self[i] = state_dict[i]
        
    def state_dict(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        cur = {}

        for i, v in enumerate(self):
            if isinstance(v, Storable):
                cur[i] = v.state_dict()
            else:
                cur[i] = v
        return cur


class DDict(typing.Dict, Struct):
    
    def as_dict(self) -> Dict:
        return self

    def as_text(self, heading: str=None) -> str:
        
        text = '\n'.join([f"{k}: {v}" for k, v in self.items()])
        if heading is not None:
            return (
                f'{heading}'
                f'{text}'
            )
        return text
    
    def spawn(self) -> 'DDict':

        return DDict(
            {k: x.spawn() if isinstance(x, Storable) else x for k, x in self.items()}
        )

    def load_state_dict(self, state_dict: typing.Dict):
        """

        Args:
            state_dict (typing.Dict): 
        """
        for k, v in enumerate(self.items()):
            if isinstance(v, Storable):
                self[k] = v.load_state_dict(state_dict[i])
            else:
                self[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """

        Returns:
            typing.Dict: 
        """
        cur = {}

        for k, v in self.items():
            if isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur


S = typing.TypeVar('S', bound=Struct)


class Wrapper(Struct, typing.Generic[S]):

    def __init__(self, wrapped: S=None):
        super().__init__()
        self.wrapped = wrapped

    def as_text(self) -> str:
        if self.wrapped is None:
            return ''
        return self.wrapped.as_text()

    def as_dict(self) -> typing.Dict:
        if self.wrapped is None:
            return {}
        return self.wrapped.as_text()

    def reset(self):
        super().__init__()
        self.wrapped = None
