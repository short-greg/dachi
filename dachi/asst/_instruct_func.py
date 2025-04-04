# 1st party
import typing
from abc import abstractmethod, ABC
from typing import Self
import inspect
from functools import wraps

# 3rd party
import pydantic

# local
from ..base import Trainable, Renderable
from ..proc._process import (
    Module, 
    AsyncModule, StreamModule, AsyncStreamModule,
    Param
)
from ._parse import FullParser, ParseConv, NullParser
from ..utils import is_primitive, primitives, str_formatter
from ._msg import (
    ToMsg, ToText
)
from ._asst import (
    AsyncAssist, AsyncStreamAssist, 
    Assist, StreamAssist
)
from ._out import (
    OutConv, NullOutConv,
    PrimConv, PydanticConv
)
from ..msg import render, render_multi


Engine: typing.TypeAlias = Assist | AsyncAssist | StreamAssist | AsyncStreamAssist

S = typing.TypeVar('S')

# TODO: MOVE OUT OF HERE

class Instruct(ABC):
    """
    """
    @abstractmethod
    def i(self) -> 'Cue':
        """Create an Instruct class used for instructions

        Returns:
            Cue: Get the cue
        """
        pass


class Cue(
    Trainable, 
    Instruct, typing.Generic[S], Renderable
):
    """Specific cue for the model to use
    """
    text: str
    _out: typing.Optional[OutConv] = pydantic.PrivateAttr(default=None)

    def __init__(
        self, text: str, name: str='', 
        out: typing.Optional[OutConv] = None
    ):
        """
        Initializes the instance with the provided text, name, and optional output converter.
        Args:
            text (str): The text to be processed.
            name (str, optional): The name associated with the text. Defaults to an empty string.
            out (Optional[OutConv], optional): The converter to use for processing the output. Defaults to None.
        """
        super().__init__(text=text, name=name)
        self._out = out

    def i(self) -> Self:
        return self

    @pydantic.field_validator('text', mode='before')
    def convert_renderable_to_string(cls, v):
        if isinstance(v, Renderable):
            return v.render()
        if is_primitive(v):
            return str(v)
        return v

    def render(self) -> str:
        """Render the cue

        Returns:
            str: The text for the cue 
        """
        return self.text

    def read(self, data: str) -> typing.Any:
        """Read the data

        Args:
            data (str): The data to read

        Raises:
            RuntimeError: If the cue does not have a out

        Returns:
            S: The result of the read process
        """
        if self._out is None:
            return data
            # raise RuntimeError(
            #     "Out has not been specified so can't read it"
            # )
        
        return self._out(data)

    def state_dict(self) -> typing.Dict:
        
        return {
            'text': self.text,
        }

    def load_state_dict(self, params: typing.Dict):
        
        self.text = params['text']

    @property
    def fixed_data(self):
        return {"out"}


X = typing.Union[str, Cue]

def validate_out(cues: typing.List[X]) -> typing.Optional[OutConv]:
    """Validate an Out based on several instructions

    Args:
        instructions (typing.List[X]): The instructions 

    Returns:
        Out: The resulting "Out" to use from the instructions
    """
    if isinstance(cues, dict):
        cues = cues.values()

    out = None
    for cue in cues:
        if not isinstance(cue, Cue):
            continue
        if out is None and cue._out is not None:
            out = cue._out
        elif cue._out is not None:
            raise RuntimeError(f'Out cannot be duplicated')
    return out



class IBase(ABC):
    """
    This is the base class for wrapping an Instruction functor. It is used to create an instruction when called.
    """

    def __init__(self, f, is_method: bool=False, out: OutConv=None, parser: ParseConv=None, content_name: str='content'):
        """ Initializes the IBase instance.
        Args:
            f (Callable): The function to be wrapped as an instruction.
            is_method (bool, optional): Indicates if the function is a method. Defaults to False.
            out (OutConv, optional): The output converter that converts the output of the LLM into a more useful format. Defaults to None.
        """
        
        self._f = f
        self._is_method = is_method
        self._docstring = f.__doc__
        self._name = f.__name__
        self._signature = str(inspect.signature(f))
        self._parameters = inspect.signature(f).parameters
        self._return_annotation = inspect.signature(f).return_annotation

        self.content_name = content_name
        out_cls = self._return_annotation
        if out is None:
            if self.out_cls in primitives:
                out = PrimConv(
                    name='out', from_='data',
                    out_cls=self.out_cls
                )
            elif issubclass(out_cls, pydantic.BaseModel):
                out = PydanticConv(name='out', from_='data', out_cls=out_cls)
            else:
                out = NullOutConv(name='out', from_='data')
        if parser is None:
            if self.out_cls is str:
                parser = NullParser('data', self.content_name)
            elif self.out_cls in primitives:
                parser = FullParser('data', self.content_name)
            elif issubclass(out_cls, pydantic.BaseModel):
                parser = FullParser('data', self.content_name)
            else:
                parser = NullParser('data', self.content_name)

        self._parser = parser
        self._out: OutConv = out

    def _align_params(
        self, *args, **kwargs
    ) -> typing.Dict:

        param_values = list(self._parameters.values())
        if self._is_method:
            param_values = param_values[1:]
        values = {}
        for value, param in zip(args, param_values):
            values[param.name] = value
        
        for k, value in kwargs.items():
            param = self._parameters[k]
            values[param.name] = value
            
        return values
    
    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the instruction
        """
        return self._name

    @property
    def f(self) -> typing.Callable:
        """
        Returns:
            typing.Callable: The function wrapped
        """
        return self._f

    @property
    def is_method(self) -> typing.Callable:
        """
        Returns:
            typing.Callable: Whether the function wrapped is a method or a regular function
        """
        return self._is_method
    
    @abstractmethod
    def __call__(self, *args, **kwds) -> Cue:
        """

        Returns:
            Cue: The cue produced by the 
        """
        pass

    @property
    @abstractmethod
    def out_cls(self) -> typing.Type:
        """

        Returns:
            typing.Type: 
        """
        pass

    @property
    def parser(self) -> ParseConv:
        return self._parser

    @property
    def out(self) -> OutConv:
        return self._out


class InstF(IBase):
    """
    InstF is a functor class that returns a Cue when called. The function it wraps must return the cue
    """

    def __call__(self, instance, *args, **kwargs) -> Cue:
        """

        Args:
            instance: The instance for the object wrapped by the instruction if it is a method

        Returns:
            Cue: The Cue returned by the function
        """
        params = self._align_params(
            *args, **kwargs
        )

        if instance is not None:
            return self._f(
                instance, **params
            )
        return self._f(
            **params
        )

    @property
    def out_cls(self) -> typing.Type:
        return self._return_annotation


class SigF(IBase):
    """
    SigF is a functor class that returns a Cue when called. The function it wraps must return a 
    dictionary of arguments that are inserted into
    the signature
    """

    def __init__(
        self, f, is_method = False, 
        doc: typing.Optional[str]=None,
        train: bool=False, out: OutConv=None,
        parser: ParseConv=None,
        content_name: str='content'
    ):
        """
        Args:
            f (_type_): The function to wrap
            is_method (bool, optional): Whether f is a method. Defaults to False.
            doc (typing.Optional[str], optional): The docstring for the function. Defaults to None.
            train (bool, optional): Whether to train the instruction. Defaults to False.
            out (OutConv, optional): The out to process the output. Defaults to None.
        """
        super().__init__(f, is_method, out, parser, content_name)

        self._doc = doc or self._docstring
        cue = Cue(
            text=self._doc
        )
        self._doc_param = Param(
            name=self._name,
            data=cue,
            training=train
        )
        
    def __call__(self, instance, *args, **kwargs) -> str:
        """Create the Cue from the docstring and the args, kwargs

        Args:
            instance: The instance for the function if a method

        Returns:
            str: The instruction
        """
        params = self._align_params(
            *args, **kwargs
        )

        if instance is None:
            cur_kwargs = self._f(**params) or {}
        else:
            cur_kwargs = self._f(
                instance, **params
            ) or {}
    
        kwargs = {**kwargs, **cur_kwargs}
        kwargs.update(params)
        doc = self._doc_param.render()
        if "{TEMPLATE}" in doc:
            kwargs['TEMPLATE'] = self._out.template()

        cue = Cue(
            text=str_formatter(
                doc, required=False, **kwargs
        ), name=self._name)
        return cue

    @property
    def out_cls(self) -> typing.Type:
        return self._return_annotation


class FuncDecBase:
    """This is used to decorate an "instruct" function
    that will be used 
    """

    def __init__(self, inst: IBase, instance=None):
        """

        Args:
            inst (IBase): 
            instance (_type_, optional): . Defaults to None.
        """
        self._inst = inst
        self._instance = instance

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        """
        pass

    def get_instance(self, args):
        """Get the instance and the args for the funciton.
        If the instance has not been stored and it is a
        method it will be the first argument.

        Args:
            args: The arguments sent to the function

        Returns:
            object: The instance if a method is decorated or None
        """
        if self._inst.is_method:
            if self._instance is not None:
                return self._instance, args
            
            return args[0], args[1:]
        return None, args

    def get_engine(self, instance):
        """Get the engine for the function

        Args:
            instance: The instance to use

        Returns:
            The engine
        """
        if isinstance(self._engine, str):
            return object.__getattribute__(instance, self._engine)
        return self._engine

    def __get__(self, instance, owner):
        """Set the instance on the SignatureMethod

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        if self._inst.name not in instance.__dict__:
            instance.__dict__[self._inst.name] = self.spawn(instance=instance)
    
        return instance.__dict__[self._inst.name]
    
    def spawn(self, instance=None) -> Self:
        """
        Spawns a new instance of the current class (Func Decorator) with the specified arguments.
        Args:
            instance (optional): An optional argument to specify a new instance.
        Returns:
            Self: A new instance of the current class with the provided arguments.
        """
        
        return self.__class__(
            inst=self._inst, 
            instance=instance
        )

    def i(self, *args, **kwargs) -> Cue:
        """
        Creates a Cue for the function decorator.
        This method retrieves an instance and processes the provided arguments
        and keyword arguments to generate a Cue object.
        Args:
            *args: Positional arguments to be passed to the instance.
            **kwargs: Keyword arguments to be passed to the instance.
        Returns:
            Cue: A Cue object created for the function decorator.
        """
        instance, args = self.get_instance(args)
        return self._inst(
            instance, *args, **kwargs
        )


class FuncDec(FuncDecBase, Module):
    """
    A class that allows one to decorate a function with an "instruction" so that the function will be an LLM (Language Model) call.
    """
    def __init__(
        self, inst, engine: Assist, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        """_summary_

        Args:
            inst: The 
            engine (Assist): The engine to use 
            to_msg (ToMsg, optional): Converts the input into a message. Defaults to None.
            kwargs (typing.Dict, optional): The kwargs to send to the engine. Defaults to None.
            instance (optional): The instance if it is a method. Defaults to None.
        """
        super().__init__(inst, instance)
        self._engine = engine
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    def forward(self, *args, **kwargs):
        """
        Executes the assistant with the provided arguments and processes the result.
        This method retrieves an instance and its arguments, constructs a message
        from the instance, and passes it to the engine for execution. The result
        from the engine is then parsed and processed to produce the final output.
        Args:
            *args: Positional arguments to be passed to the instance and engine.
            **kwargs: Keyword arguments to be passed to the instance and engine.
        Returns:
            The processed output obtained after parsing and processing the engine's result.
        """
        
        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue.text)
        engine = self.get_engine(instance)

        res_msg = engine(
            [msg], **self._kwargs
        )
        res_msg = self._inst.parser(res_msg)
        return self._inst.out(res_msg).m['out']

    def spawn(self, instance=None) -> Self:
        """
        Spawns a new instance of the current class, optionally updating the instance.
        Args:
            instance (optional): The new instance to be used. If not specified, 
                the current instance is retained.
        Returns:
            Self: A new instance of the current class with updated attributes.
        """
        
        return self.__class__(
            inst=self._inst, 
            engine=self._engine, instance=instance,
            to_msg=self._to_msg, kwargs=self._kwargs
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class AFuncDec(FuncDecBase, AsyncModule):
    """
    A class that allows one to decorate an async function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    def __init__(
        self, inst, engine: AsyncAssist, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        
        super().__init__(inst, instance)
        self._engine = engine
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    async def aforward(self, *args, **kwargs):
        """
        Asynchronously forwards the given arguments to the instance's engine.
        This method retrieves the instance from the provided arguments, constructs
        a cue using the instance and additional arguments, converts the cue text
        to a message, and then forwards the message to the instance's engine
        asynchronously. The response from the engine is then read and returned.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            The result read from the engine's response.
        """

        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue.text)
        engine = self.get_engine(instance)
        res_msg = await engine.aforward(
            [msg], **self._kwargs
        )
        res_msg = self._inst.parser(res_msg)
        return self._inst.out(res_msg)
    
    async def __call__(self, *args, **kwargs):
        """
        Asynchronously calls the aforward method with the given arguments.
        This method acts as an alias for the aforward method, allowing the instance
        to be called directly as a function.
        Args:
            *args: Variable length argument list to be passed to aforward.
            **kwargs: Arbitrary keyword arguments to be passed to aforward.
        Returns:
            The result of the aforward method.
        """

        return await self.aforward(*args, **kwargs)

    def spawn(self, instance=None) -> Self:
        """
        Spawns a new AsyncFuncDec instance, allowing the user to set the instance.
        Args:
            instance (optional): The instance to set for the new AsyncFuncDec.
        Returns:
            Self: A new instance of the AsyncFuncDec class with the specified instance.
        """
        
        return self.__class__(
            inst=self._inst,
            engine=self._engine, instance=instance,
            to_msg=self._to_msg, kwargs=self._kwargs
        )
    

class StreamDec(FuncDecBase, StreamModule):
    """
    A class that allows one to decorate a streaming function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    def __init__(
        self, inst: IBase, engine: StreamAssist, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        super().__init__(inst, instance)
        self._engine = engine
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    def stream(self, *args, **kwargs):
        """
        Streams the execution of an instruction by calling the LLM.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Yields:
            The delta of the response from the LLM engine.
        """
        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue.text)
        engine = self.get_engine(
            instance
        )

        for resp_msg in engine.stream(
            [msg], **self._kwargs
        ):
            resp = self._inst.parser(resp_msg)
            resp = self._inst.out(resp).m['out']
            yield resp


    def spawn(self, instance=None) -> Self:
        """
        Spawns a new instance of the class.
        If an instance is specified, it will create that instance.
        Args:
            instance (optional): The instance to be created. Defaults to None.
        Returns:
            Self: A new instance of the class.
        """
        return self.__class__(
            inst=self._inst, 
            engine=self._engine, instance=instance,
            kwargs=self._kwargs
        )
    
    def __call__(self, *args, **kwargs):
        """
        Alias for stream(). Streams the execution of an instruction by calling the LLM.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Yields:
            The delta of the response from the LLM engine.
        """
        
        yield from self.stream(*args, **kwargs)


class AStreamDec(FuncDecBase, AsyncStreamModule):
    """
    A class that allows one to decorate a asynchronous streaming function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    def __init__(
        self, inst, engine: AsyncStreamAssist, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        """

        Args:
            inst: The instruction to execute
            engine (AsyncStreamAssist): The LLM engine to use in execution
            to_msg (ToMsg, optional): The method to convert to a message. Defaults to None.
            kwargs (typing.Dict, optional): The kwargs to send to the LLM. Defaults to None.
            instance (optional): The instance for the function if a method. Defaults to None.
        """
        super().__init__(inst, instance)
        self._engine = engine
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    async def astream(self, *args, **kwargs):
        """
        Asynchronously streams the execution of an instruction by calling the LLM.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Yields:
            The delta of the response from the LLM engine.
        """
        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue.text)

        engine = self.get_engine(
            instance
        )

        async for resp_msg in engine.astream(
            [msg], **self._kwargs
        ):
            resp = self._inst.parser(resp_msg)
            resp = self._inst.out(resp).m['out']
            yield resp

    def spawn(self, instance=None) -> Self:
        
        return self.__class__(
            inst=self._inst, 
            engine=self._engine, instance=instance,
            to_msg=self._to_msg,
            kwargs=self._kwargs
        )

    async def __call__(self, *args, **kwargs):
        
        async for res in self.astream(
            *args, **kwargs
        ):
            yield res


def instructfunc(
    engine: Engine=None,
    out: OutConv=None,
    parser: ParseConv=None,
    is_method: bool=False,
    to_async: bool=False,
    to_stream: bool=False,
    to_msg: ToMsg=None,
    content_name: str='content',
    **kwargs
):
    """Decorate a method with instructfunc

    Args:
        engine (AIModel, optional): The engine for the AI . Defaults to None.
        is_method (bool): Whether it is a method

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        inst = InstF(f, is_method, out, parser=parser, content_name=content_name)

        if not to_async and not to_stream:
            return FuncDec(
                inst, engine, to_msg,
                kwargs=kwargs
            )
        if not to_stream:
            return AFuncDec(
                inst, engine, to_msg,
                kwargs=kwargs
            )
        if not to_async:
            return StreamDec(
                inst, engine, to_msg,
                kwargs=kwargs
            )
        return AStreamDec(
            inst, engine, to_msg,
                kwargs=kwargs
        )

    return _


def instructmethod(
    engine: Engine=None,
    out: OutConv=None,
    parser: ParseConv=None,
    to_async: bool=False,
    to_stream: bool=False,
    to_msg: ToMsg=None,
    content_name: str='content',
    **kwargs
):
    """Decorate a method with instructfunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return instructfunc(
        engine, out=out, is_method=True, to_msg=to_msg, to_async=to_async, 
        to_stream=to_stream, 
        parser=parser,
        content_name=content_name,
        **kwargs
    )


def signaturefunc(
    engine: Engine=None,
    out: OutConv=None,
    parser: ParseConv=None,
    to_msg: ToMsg=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    is_method: bool=False,
    to_async: bool=False,
    to_stream: bool=False,
    train: bool=False,
    content_name: str='content',
    **kwargs
):
    """Decorate a method with instructfunc

    Args:
        engine (AIModel, optional): The engine for the AI . Defaults to None.
        is_method (bool): Whether it is a method

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        inst = SigF(
            f, is_method, train=train, 
            doc=doc, out=out, parser=parser,
            content_name=content_name
        )

        if not to_async and not to_stream:
            return FuncDec(
                inst, engine, to_msg,
                kwargs=kwargs
            )
        if not to_stream:
            return AFuncDec(
                inst, engine, to_msg,
                kwargs=kwargs
            )
        if not to_async:
            return StreamDec(
                inst, engine, to_msg,
                kwargs=kwargs
            )
        return AStreamDec(
            inst, engine, to_msg,
            kwargs=kwargs
        )

    return _


def signaturemethod(
    engine: Engine=None, 
    out: OutConv=None,
    parser: ParseConv=None,
    to_msg: ToMsg=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    to_async: bool=False,
    to_stream: bool=False,
    train: bool=False,
    content_name: str='content',
    **kwargs
):
    """Decorate a method with SignatureFunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.
        out (out, optional): The out to use for the method. Defaults to None.
        doc (typing.Union[str, typing.Callable[[], str]], optional): A docstring to override with. Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return signaturefunc(
        engine, out=out, doc=doc, is_method=True,
        to_msg=to_msg, to_async=to_async, to_stream=to_stream,
        train=train,
        parser=parser,
        content_name=content_name,
        **kwargs
    )


def inst(x: typing.Union[typing.Iterable[X], X], cue: X) -> Cue:
    """Execute an operation on a cue

    Args:
        x (typing.Union[typing.Iterable[X], X]): The input
        cue (X): The cue for the operation

    Returns:
        Cue: The resulting cue
    """
    if not isinstance(x, typing.Iterable):
        x = [x]

    out = validate_out([*x, cue])
    resources = ', '.join(render_multi(x))
    # resources = ', '.join(x_i.name for x_i in x)
    text = f'Do: {render(cue)} --- With Inputs: {resources}'
    return Cue(
        text=text, out=out
    )


# class Inst(Module):
#     """An operation acts on an cue to produce a new cue
#     """

#     def __init__(
#         self, name: str, cue: X, 
#         tunable: bool=False):
#         """Create an operation specifying the name

#         Args:
#             name (str): The name of the operation
#             cue (X): The cue for the operation
#             tunable: whether the cue is tunable
#         """
#         self.name = name
#         if not isinstance(cue, Cue):
#             cue = Param(
#                 name=self.name, training=tunable, 
#                 data=Cue(
#                     text=render(cue)
#             ))
        
#         self.cue = cue
        
#     def forward(
#         self, *args: X, **kwargs: X
#     ) -> Cue:
#         """Fill in the cue with the inputs

#         Returns:
#             Cue: 
#         """
#         cue = render(self.cue)
#         out = validate_out(
#             [*args, *kwargs.values(), self.cue]
#         )

#         return Cue(
#             text=fill(cue, *args, **kwargs), out=out
#         )
