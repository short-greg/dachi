# 1st party
import typing
from typing_extensions import Self
from abc import abstractmethod, ABC

# 3rd party
import pydantic
from pydantic import Field

# local
from ..store import Struct, Str
import inspect


S = typing.TypeVar('S', bound=Struct)


class Description(Struct):
    """Provide context in the prompt
    """
    name: str

    @property
    @abstractmethod
    def text(self) -> str:
        pass

    @abstractmethod
    def update(self, **kwargs) -> Self:
        pass


# Do i want this to be a Pydantic model

class Ref(Struct):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    reference: Description

    @property
    def name(self) -> str:
        return self.reference.name

    @property
    def text(self) -> str:
        return ""

    def update(self, **kwargs) -> Self:
        # doesn't do anything since
        # it is a reference
        return self


class Instruction(Description):
    """Specific instruction for the model to use
    """
    instr: typing.Union[str, Str]
    incoming: typing.List['Description'] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return self.instr

    def traverse(self, visited: typing.Set=None) -> typing.Iterator['Description']:

        visited = visited or set()

        if id(self) in visited:
            return

        yield self
        visited.add(id(self))

        if self.incoming is None:
            return

        for inc in self.incoming:
            for inc_i in inc.traverse(visited):
                yield inc_i
    
    def update(self, **kwargs) -> 'Instruction':

        text = self.instr(**kwargs)
        return Instruction(
            self._name, text, self.incoming
        )


class InstructionSet(Description):

    instructions: typing.List[Instruction]

    def update(self, **kwargs) -> 'InstructionSet':
        
        return InstructionSet(
            [instruction.update(**kwargs) for instruction in self.instructions]
        )
    
    @property
    def text(self) -> str:
        return '\n\n'.join(
            i.text for i in self.instructions
        )


def traverse(*instructions: Instruction) -> typing.Iterator[Description]:

    visited = set()
    for instruction in instructions:
        for description in instruction.traverse(visited):
            yield description


class Composition(Description):
    """
    """
    descriptions: typing.List[Description]

    @property
    def text(self) -> str:

        return "\n".join(
            description.text for description in self.descriptions
        )

    def update(self, **kwargs) -> Self:
        
        descriptions = [
            for description in self.descriptions:
            description.update(**kwargs)
        ]
        return Composition(
            descriptions=descriptions, name=self.name
        )


class Operation(object):

    def forward(
        self, *x: Description, **kwargs
    ) -> Instruction:
        pass


def op(x: typing.Union[typing.List[Description], Description], intruction: str, name: str) -> Instruction:

    if isinstance(x, Description) or isinstance(x, Ref):
        x = [x]
    
    resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {intruction} --- With Inputs: {resources}'
    return Instruction(
        name=name, instr=text, incoming=x
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


def generic_class(t: typing.TypeVar, idx: int=0):

    return t.__orig_class__.__args__[idx]


class Output(typing.Generic[S], Struct):
    
    instruction: Instruction
    name: str
    style: Style = None

    def read(self, text: str) -> S:

        if self.style is not None:
            return self.style.load(text)
        
        Scls = generic_class(self)
        return Scls.from_text(text)
    
    @property
    def text(self) -> str:

        Scls = generic_class(self)
        return f"""
        {self.instruction.text}

        ===JSON OUTPUT TEMPLATE BELOW===

        {Scls.template()}
        """


class OutputList(Struct):
    
    outputs: typing.List[Output]
    header: Instruction = None
    footer: Instruction = None

    def read(self, text: str) -> typing.List:

        results = []
        # 1) split by header
        func_locs = []

        for output in self.outputs:
            cur = f"::OUT::{output.name}::"
            loc = text.find(cur)
            func_locs.append(
                (loc, loc + len(cur))
            )
        func_locs.append((-1, -1))

        for output, p1, p2 in zip(self.outputs, func_locs[:-1], func_locs[1:]):
            
            print(p1, p2)
            _ = text[p1[0]:p1[1]]
            func_response = text[p1[1]:p2[0]]
            print(func_response)
            results.append(
                output.read(func_response)
            )
        
        return results        

    @property
    def text(self) -> str:
        
        output_templates = []
        for output in self.outputs:
            output_templates.append(
                f"""
                ===OUTPUT TEMPLATE {output.name}===

                :::OUT:::{output.name}

                {output.text}
                """
            )

        out_text = ''
        if self.header is not None:
            out_text = f"""
            {self.header.text}

            """
        out_text = f"""
        {'\n\n'.join(output_templates)}
        """

        if self.footer is not None:
            out_text = f"""
            {out_text}

            {self.footer.text}
            """
        return out_text


class Instructor(ABC):

    def forward(self, *args, **kwargs) -> Output:
        pass

    def __call__(self, *args, **kwargs) -> Output:
        return self.forward(*args, **kwargs)
