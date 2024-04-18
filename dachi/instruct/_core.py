# 1st party
import typing
from abc import abstractmethod, ABC

# 3rd party
import pandas as pd
import pydantic

# local
from ..store import Struct, Str
import inspect


T = typing.TypeVar('T', bound=Struct)
S = typing.TypeVar('S', bound=Struct)


class IVar(Struct):

    name: str
    text: str


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


class Instruction(Struct):

    name: str
    style: typing.Optional[Style] = None

    def update(self, **kwargs) -> 'Instruction':
        
        new_args = {}
        for field in self.model_fields:
            attr = getattr(self, field)
            if isinstance(attr, Instruction):
                new_args[field] = attr.update(**kwargs)
            elif isinstance(attr, Str):
                new_args[field] = attr(**kwargs)
            else:
                new_args[field] = attr
        return self.__class__(**new_args)

    def forward(self, **kwargs) -> str:

        updated = self.update(**kwargs)
        if self.style is None:
            return updated.to_text()
        return self.style(self)


class Material(Instruction):
    
    data: Struct
    style: 'RStyle' = None


def mat(name: str, data: Struct, style: Style=None) -> Material:

    return Material(
        name=name, data=data, style=style
    )


class IOut(Struct, typing.Generic[S]):

    name: str
    text: str
    style: 'RStyle' = None

    def read(self, text: str) -> S:

        if self.style is not None:
            return self.style.load(text)
        return S.load(text)


class Op(Struct):
    
    descr: str
    out_name: str

    @abstractmethod
    def forward(self, inputs: typing.List[IVar]) -> str:
        pass


def assist(f: typing.Callable[[Material], IOut]):
    """Use assist to have a "sub" function in the module. The subfunction
    must output an IOut and take in one or more Materials. 

    Args:
        f (typing.Callable[[Material], IOut]): The sub function
    """
    # TODO: Make use of annotations or
    # code contained in the function
    def _(*args, **kwargs):
        
        out = f(*args, **kwargs)
        return out

    return _


def assistmethod(f):
    """Use assist to have a "sub" method for the instruction. The subfunction
    must output an IOut and take in one or more Materials. 

    Args:
        f (typing.Callable[[Material], IOut]): The sub function
    """
    def _(self, *args, **kwargs):

        out = f(self, *args, **kwargs)
        return out

    return _


# class Assist(Struct):

#     code: str
#     doc: str
#     inputs: typing.List[IVar]
#     outputs: IOut
#     signature: str

#     def forward(self, inputs: typing.List[IVar]) -> str:
#         pass


# class Func(pydantic.BaseModel):

#     name: str
#     doc: str
#     signature: str
#     code: str
#     inputs: typing.List[Struct]
#     outputs: typing.List['Output']


def op(inputs: typing.List[IVar], descr: str, name: str) -> IVar:

    name_list = ','.join(input_.name for input_ in inputs)
    text = f'Compute {name} from {name_list} - {descr} \n'
    return IVar(
        name=name,
        text=text
    )


def out(
    inputs: typing.List[IVar], descr: str, 
    name: str, style: 'RStyle'=None
) -> IOut[S]:
    """

    Args:
        inputs (typing.List[IVar]): 
        descr (str): 
        name (str): 
        style (RevStyle, optional): . Defaults to None.

    Returns:
        IOut[S]: 
    """

    name_list = ','.join(input_.name for input_ in inputs)
    text = f'Output {name} using {name_list} - {descr} \n'
    return IOut[S](
        name=name,
        text=text,
        style=style
    )


class RStyle(Style[S]):
    """Style with a verse method
    """

    @abstractmethod
    def reverse(self, text: str) -> S:
        pass

    def load(self, text: str) -> S:
        return self.reverse(text)


class Instructor(object):

    @abstractmethod
    def forward(self, *args, **kwargs) -> str:

        # 1) # run operations
        # 2) return out.to_text()

        # 1) # run operations
        # 2) return self.context(inputs, outputs)

        # 1) # call "functions"
        # 2) return self.context(**inputs, **outputs)

        # 1) Nested 

        # 1) output contains all of the input text
        # 2) f
        pass 

    def __call__(self, *args, **kwargs) -> str:

        return self.forward(*args, **kwargs)
    

class _op:

    def __getattr__(self, key: str):

        name = key
        def _op_creator(descr: str) -> Op:
            return Op(name=name, descr=descr)
        return _op_creator


op = _op()        




# class InstructChat(Chat):

#     def __init__(self, structs: typing.List[Message] = None, instructor: str='system'):
#         super().__init__(structs)
#         self.instructor = instructor

#     def chat(self) -> Chat:

#         return Chat(

#             s for s in self._structs if s.role != self.instructor
#         )


# TODO: Decide how to implement this

# class Param(typing.Generic[S]):

#     def __init__(
#         self, struct: S, style: Style
#     ):
#         """

#         Args:
#             struct (S): 
#             style (Style): 
#         """
#         self.struct = struct
#         self.style = style

#     def detach(self) -> 'Param':

#         return Param(self.struct)



# # How about this?


# # Have to specify the names of inputs here
# # they are contained in the annotation
