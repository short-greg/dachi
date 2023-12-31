import typing
from dataclasses import dataclass, Field

import typing
from dataclasses import dataclass, Field
from typing import Dict
from ..comm._serving import Component


@dataclass
class Role(object):

    name: str
    meta: Field(default_factory=dict)


@dataclass
class Turn(object):

    role: Role
    text: str


class Conversation(Component):

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
    
    def clear(self):
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
    
    def remap_role(self, **role_map) -> 'Conversation':
        """Remap the names of the roles to a new set of roles

        Returns:
            Conversation: The conversation with remapped roles
        """
        roles = [role_map.get(role, role) for role in self._roles]
        turns = [
            (role_map.get(role, role), text)
            for role, text in self._turns
        ]
        conversation = Conversation(
            roles=roles, max_turns=self._max_turns
        )
        conversation._turns = turns
        return conversation

    def as_dict(self) -> Dict:
        return {
            'roles': self._roles,
            'turns': self._turns,
            'max_turns': self._max_turns   
        }
    
    def as_text(self, heading: str=None) -> str:
        
        result = f'{heading}\n' if heading is not None else ''
        for turn in self._turns:
            result += f'{turn[0]}: {turn[1]}\n'
        return result


class Text(Component):
    """A simple wrapper to use text as a prompt component
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


# Example
# Plan    
# Completion
# 

# @dataclass
# class Arg:
    
#     name: str
#     description: str = Field("")
    

# class Prompt(object):

#     def __init__(self, args: typing.List[typing.Union[Arg, str]], text: str):
        
#         super().__init__()
#         self._args = {}
#         for arg in args:
#             if isinstance(arg, str):
#                 self._args[arg] = Arg(arg)
#             else:
#                 self._args[arg.name] = arg
#         self._text = text

#     def format(self, **kwargs):

#         input_names = set(kwargs.keys())
#         difference = input_names - set(self._args)
#         if len(difference) != 0:
#             raise ValueError(f'Input has keys that are not arguments to the prompt')
#         inputs = {}
#         for k, v in self._args.items():
#             if k in kwargs:
#                 inputs[k] = v
#             else:
#                 inputs[k] = "{{}}"
#         return Prompt(
#             self._text.format(**inputs)
#         )
    
#     @property
#     def text(self) -> str:

#         return self._text


# class Example
# class Plan
# class Document
# class Map
# ...
