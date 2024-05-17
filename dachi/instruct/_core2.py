# 1st party
import typing
from abc import abstractmethod, ABC

# 3rd party
import pandas as pd
import pydantic

# local
from ..store import Struct, Str
import inspect


# Do I want to inherit from
# pydantic.BaseModel

# pydantic.Struct

S = typing.TypeVar('S', bound=Struct)


class Description(Struct):
    """Provide context in the promptt
    """
    name: str

    @property
    @abstractmethod
    def text(self) -> str:
        pass

    @abstractmethod
    def update(self, **kwargs) -> 'Description':
        pass


class Ref(Description):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    description: Description

    @property
    def text(self) -> str:
        return ""

    def update(self, **kwargs) -> 'Description':
        # doesn't do anything since
        # it is a reference
        pass


class Instruction(Description):
    """Specific instruction for the model to use
    """
    name: str
    text: typing.Union[str, Str]
    incoming: typing.List['Description']

    @property
    def text(self) -> str:
        return self._text
    
    def incoming(self) -> typing.Iterator['Description']:

        for inc in self._incoming:
            yield inc

    def traverse(self, visited: typing.Set=None) -> typing.Iterator['Description']:

        visited = visited or set()

        if self in visited:
            return

        yield self
        visited.add(self)

        for inc in self._incoming:
            for inc_i in inc.traverse(visited):
                yield inc_i
    
    def update(self, **kwargs) -> 'Instruction':

        text = self._text(**kwargs)
        return Instruction(
            self._name, text, self._incoming
        )
    
    def list_up(self) -> typing.List['Description']:

        return [i for i in self.traverse()]


class InstructionSet(Instruction):

    # TODO: Update this
    
    def __init__(self, instructions: typing.List[Instruction], name: str):

        super().__init__(name, '\n\n'.join(
            i.text for i in instructions
        ), instructions)


def traverse(*instructions: Instruction) -> typing.Iterator[Description]:

    visited = set()
    for instruction in instructions:
        for description in instruction.traverse(visited):
            yield description


class Operation(object):

    def forward(
        self, *x: Description, **kwargs
    ) -> Instruction:
        pass



def op(x: typing.Union[typing.List[Description], Description], intruction: str, name: str) -> Instruction:

    if isinstance(x, Description):
        x = [x]
    
    resources = ', '.join(x_i.name for x_i in x)
    text = f'{intruction} . Use {resources}'
    return Instruction(
        name, text, x
    )


class Style(pydantic.BaseModel, typing.Generic[S], ABC):

    @abstractmethod
    def forward(self, struct: S) -> str:
        pass

    @abstractmethod
    def reverse(self, text: str) -> S:
        pass

    def load(self, text: str) -> S:
        return self.reverse(text)

    def __call__(self, struct: S) -> str:
        return self.forward(struct)

    @pydantic.field_validator('*', mode='before')
    def convert_to_string_template(cls, v, info: pydantic.ValidationInfo):
    
        outer_type = cls.model_fields[info.field_name].annotation
        if (inspect.isclass(outer_type) and issubclass(outer_type, Str)) and not isinstance(v, Str):
            return Str(text=v)
        return v


class Output(Struct, typing.Generic[S]):
    
    instruction: Instruction
    style: Style = None

    def read(self, text: str) -> S:

        if self.style is not None:
            return self.style.load(text)
        return S.load(text)
    
    @property
    def text(self) -> str:
        return self.instruction.text


class OutputList(Struct):
    
    header: Instruction
    outputs: typing.List[Output]
    footer: Instruction

    def read(self, text: str) -> S:

        if self.style is not None:
            return self.style.load(text)
        return S.load(text)

    # TODO: Not finished. Have to have to 
    #  indicate where one output starts
    #  and another begins etc
    @property
    def text(self) -> str:
        return (
            f"""
            {self.header.text}

            {'\n\n'.join(output.text for output in self.outputs)}

            {self.footer.text}            
            """
        )

