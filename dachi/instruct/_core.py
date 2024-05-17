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


class Instruct(Struct):

    name: str
    style: typing.Optional[Style] = None

    def update(self, **kwargs) -> 'Instruct':
        
        new_args = {}
        for field in self.model_fields:
            attr = getattr(self, field)
            if isinstance(attr, Instruct):
                new_args[field] = attr.update(**kwargs)
            elif isinstance(attr, Str):
                new_args[field] = attr(**kwargs)
            else:
                new_args[field] = attr
        return self.__class__(**new_args)

    @abstractmethod
    def forward(self, **kwargs) -> 'Instruction':
        pass


class Instruction(Struct):

    text: str

    def to_text(self) -> str:
        return self.text


class Process(Struct):

    def forward(self, response: str):
        pass


class Func(Struct, typing.Type[S]):

    name: str
    instruction: Instruction
    style: Style = None

    def to_text(self) -> str:
        return self.instruction.text
    
    def forward(self, response: str) -> typing.Union[S, None]:

        try:
            return S.read(response)
        except pydantic.ValidationError:
            return None


class FuncSet(Struct):

    header: Instruction
    funcs: typing.List[Func]
    footer: Instruction

    def to_text(self) -> str:

        func_texts = []
        for func in self.funcs:
            func_texts.append(
            f"""
            === {func.name} ===
            {func.to_text()}

            === {func.name} OUTPUT TEMPLATE ===
            ::OUT::{func.name}

            {func.template()}
            """
        )
            
        return f"""
        {self.header.text}
        {'\n'.join(func_texts)}
        {self.footer.text}
        """
    
    def forward(self, response: str) -> typing.List[Struct]:
        
        results = []
        # 1) split by header
        func_locs = []
        for func in self.funcs:
            cur = f"::OUT::{func.name}"
            loc = response.find(cur)
            func_locs.append(
                loc, len(cur)
            )
        func_locs.append((-1, -1))

        for func, p1, p2 in zip(self.funcs, func_locs[:-1], func_locs[1:]):
            
            header = response[p1[0]:p1[1]]
            func_response = header[p1[1]:p2[0]]
            results.append(
                func(func_response)
            )
        return results


class Material(Instruct):
    
    descr: str = ''
    data: typing.Union[Struct, Style[Struct]]
    # style: 'Template' = None

    def ref(self) -> 'Ref':
        return Ref(name=self.name, material=self)
    
    def forward(self) -> str:
        
        data = self.style(
            self.struct
        )
        return f"""
        === {self.name} ====

        {data}
        
        """


class Ref(Instruct):

    material: 'Material'

    def forward(self) -> Instruction:
        return self.name


def mat(name: str, data: Struct, style: Style=None) -> Material:

    return Material(
        name=name, data=data, style=style
    )


def instruct(f: typing.Callable[[Struct], Func]):
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


def instructmethod(f: typing.Callable[[Struct], Func]):
    """Use instructmethod to have a "sub" method for the instruction. The subfunction
    must output an IOut and take in one or more Materials. 

    Args:
        f (typing.Callable[[Material], IOut]): The sub function
    """
    def _(self, *args, **kwargs):

        out = f(self, *args, **kwargs)
        return out

    return _


class RStyle(Style[S]):
    """Style with a reverse method
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
    


# class Op(Struct):
    
#     descr: str
#     out_name: str

#     def forward(self, inputs: typing.List[IVar]) -> str:
#         pass


# TODO: Finalize this

# class _op:

#     def __getattr__(self, key: str):

#         name = key
#         def _op_creator(descr: str) -> Op:
#             return Op(name=name, descr=descr)
#         return _op_creator


# op = _op()        



# class IOut(Struct, typing.Generic[S]):

#     name: str
#     text: str
#     style: 'RStyle' = None

#     def read(self, text: str) -> S:

#         if self.style is not None:
#             return self.style.load(text)
#         return S.load(text)


# class Op(Struct):
    
#     descr: str
#     out_name: str

#     @abstractmethod
#     def forward(self, inputs: typing.List[IVar]) -> str:
#         pass


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


# def op(inputs: typing.List[IVar], descr: str, name: str) -> IVar:

#     name_list = ','.join(input_.name for input_ in inputs)
#     text = f'Compute {name} from {name_list} - {descr} \n'
#     return IVar(
#         name=name,
#         text=text
#     )


# def out(
#     inputs: typing.List[IVar], descr: str, 
#     name: str, style: 'RStyle'=None
# ) -> IOut[S]:
#     """

#     Args:
#         inputs (typing.List[IVar]): 
#         descr (str): 
#         name (str): 
#         style (RevStyle, optional): . Defaults to None.

#     Returns:
#         IOut[S]: 
#     """

#     name_list = ','.join(input_.name for input_ in inputs)
#     text = f'Output {name} using {name_list} - {descr} \n'
#     return IOut[S](
#         name=name,
#         text=text,
#         style=style
#     )





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
