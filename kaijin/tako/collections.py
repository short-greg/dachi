import typing
from dataclasses import dataclass, Field


@dataclass
class Role(object):

    name: str
    meta: Field(default_factory=dict)


@dataclass
class Turn(object):

    role: Role
    text: str


class Conversation(object):

    def __init__(self, roles: typing.Dict[str, Role]=None, max_turns: int=None):

        self._roles: typing.Dict[str, Role] = roles or {}
        self._turns: typing.List[Turn] = []
        self._max_turns = max_turns

    def add_turn(self, role: str, text: str) -> 'Conversation':

        if role not in self._roles:
            self._roles[role] = Role(role)
        if self._max_turns is not None and len(self._turns) == self._max_turns:
            self._turns = self._turns[1:]
        self._turns.append(Turn(self._roles[role], text))

        return self
    
    def add_role(self, role: Role, overwrite: bool=False) -> 'Conversation':

        if overwrite and role.name in self._roles:
            raise KeyError(f'Role {role.name} already added.')
        self._roles[role] = role
        return self
    
    def to_dict(self) -> typing.Dict[str, str]:

        return {
            turn.role.name: turn.text
            for turn in self._turns
        }

    def __len__(self) -> int:
        return len(self._turns)


@dataclass
class Arg:
    
    name: str
    description: str = Field("")
    

class Prompt(object):

    def __init__(self, args: typing.List[typing.Union[Arg, str]], text: str):
        
        super().__init__()
        self._args = {}
        for arg in args:
            if isinstance(arg, str):
                self._args[arg] = Arg(arg)
            else:
                self._args[arg.name] = arg
        self._text = text

    def format(self, **kwargs):

        input_names = set(kwargs.keys())
        difference = input_names - set(self._args)
        if len(difference) != 0:
            raise ValueError(f'Input has keys that are not arguments to the prompt')
        inputs = {}
        for k, v in self._args.items():
            if k in kwargs:
                inputs[k] = v
            else:
                inputs[k] = "{{}}"
        return Prompt(
            self._text.format(**inputs)
        )
