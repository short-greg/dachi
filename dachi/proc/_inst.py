"""
This module provides classes and decorators for creating and specifying functions as instructions for a language model.

There are two main types
1) signaturefunc - where the function returns a dictionary of arguments to be inserted into a docstring template
2) instructfunc - where the function returns a string that is the instruction to be sent to the LLM

Example usage:
```python
from dachi.proc import instructfunc, signaturefunc

@instructfunc(engine=llm_engine)
def my_function(param1: str, param2: int) -> str:
    return f"Process {param1} with number {param2}"
result = my_function("example", 42)
print(result)  # Output from the LLM based on the instruction

@signaturefunc(engine=llm_engine)
def summarize(text: str) -> dict:
    \"""
    Process the following text and return a summary.
    
    {text}
    \"""
    pass


"""

# 1st party
import typing
from abc import abstractmethod
from typing import Self
import inspect
from functools import wraps

# 3rd party
import pydantic

# local
from ..core import Module
from ._process import (
    Process, AsyncProcess, StreamProcess, 
    AsyncStreamProcess
)
from ..core import Param, END_TOK
from ..utils import primitives, str_formatter
from ._msg import ToPrompt, NullToPrompt
from ._resp import (
    ToOut, PrimOut, TextOut, StructOut
)
Engine: typing.TypeAlias = Process | AsyncProcess | StreamProcess | AsyncStreamProcess
from dachi.core import render
from dachi.inst import style_formatter

S = typing.TypeVar('S')
# TODO: MOVE OUT OF HERE

from pydantic import Field, PrivateAttr

class IBase(Module):
    """
    This is the base class for wrapping an Instruction functor. It is used to create an instruction when called.
    """
    f: typing.Callable
    is_method: bool = False
    out_conv: ToOut = Field(default=None, exclude=True)

    _docstring: str = PrivateAttr(default='')
    _name: str = PrivateAttr(default='')
    _signature: str = PrivateAttr(default='')
    _sig_params: typing.Dict[str, inspect.Parameter] = PrivateAttr(default_factory=dict)
    _return_annotation: typing.Type = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """ Initializes the IBase instance.
        Args:
            f (Callable): The function to be wrapped as an instruction.
            is_method (bool, optional): Indicates if the function is a method. Defaults to False.
            out (OutConv, optional): The output converter that converts the output of the LLM into a more useful format. Defaults to None.
        """
        super().model_post_init(__context)
        self._docstring = self.f.__doc__
        self._name = self.f.__name__
        self._signature = str(inspect.signature(self.f))
        self._sig_params = inspect.signature(self.f).parameters
        self._return_annotation = inspect.signature(self.f).return_annotation

        out_cls = self._return_annotation
        if self.out_conv is None:
            if out_cls != inspect.Signature.empty and hasattr(out_cls, '__bases__') and issubclass(out_cls, pydantic.BaseModel):
                # For pydantic models, use StructOut
                self.out_conv = StructOut(struct=out_cls)
            elif out_cls is str:
                # For string outputs, use TextOut (good for streaming)
                self.out_conv = TextOut()
            elif out_cls in primitives:
                # For other primitives, use PrimOut
                self.out_conv = PrimOut(out_cls=out_cls)
            else:
                # Default case - use TextOut
                self.out_conv = TextOut()

    def _align_params(
        self, *args, **kwargs
    ) -> typing.Dict:
        """
        Returns:
            typing.Dict: Get the parameters
        """
        param_values = list(self._sig_params.values())
        if self.is_method:
            param_values = param_values[1:]
        values = {}
        for value, param in zip(args, param_values):
            values[param.name] = value
        
        for k, value in kwargs.items():
            param = self._sig_params[k]
            values[param.name] = value
            
        return values
    
    def template(self) -> str:
        """Produce a template for the instruction

        Returns:
            str: the template
        """
        if self._out_conv is None:
            return ''
        return self.out_conv.template()

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the instruction
        """
        return self._name
    
    @abstractmethod
    def __call__(self, *args, **kwds) -> typing.Any:
        """

        Returns:
            typing.Any: The instructions for the LLM
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


class InstF(IBase):
    """
    InstF is a functor class that returns a Cue when called. The function it wraps must return the cue
    """

    def __call__(self, instance, *args, **kwargs) -> str:
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
            return self.f(
                instance, **params
            )
        return self.f(
            **params
        )

    @property
    def out_cls(self) -> typing.Type:
        """
        Returns:
            typing.Type: Get the type for the output
        """
        return self._return_annotation


class SigF(IBase):
    """
    SigF is a functor class that returns a Cue when called. The function it wraps must return a 
    dictionary of arguments that are inserted into
    the signature
    """
    train: bool = False
    doc: str = ''

    def model_post_init(self, __context):
        """
        Args:
            f: The function to wrap
            is_method (bool, optional): Whether f is a method. Defaults to False.
            doc (typing.Optional[str], optional): The docstring for the function. Defaults to None.
            train (bool, optional): Whether to train the instruction. Defaults to False.
            out (OutConv, optional): The out to process the output. Defaults to None.
        """
        super().model_post_init(__context)
        self.doc = self.doc or self._docstring
        self._doc_param = Param[str](
            data=self.doc,
            _fixed=not self.train
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
            cur_kwargs = self.f(**params) or {}
        else:
            cur_kwargs = self.f(
                instance, **params
            ) or {}
    
        kwargs = {**kwargs, **cur_kwargs}
        kwargs.update(params)
        # TODO: Update this to use rendering
        doc = self._doc_param.data
        if "{TEMPLATE}" in doc:
            kwargs['TEMPLATE'] = self.template()

        return str_formatter(
                doc, required=False, **kwargs
        )

    @property
    def out_cls(self) -> typing.Type:
        """Get the type of the return value

        Returns:
            typing.Type: The type of the return value
        """
        return self._return_annotation


class FuncDecBase(pydantic.BaseModel):
    """This is used to decorate an "instruct" function
    that will be used 
    """

    engine: StreamProcess | Process | AsyncProcess | AsyncStreamProcess | str | None = None
    inst: IBase
    to_msg: ToPrompt
    _instance: typing.Any = pydantic.PrivateAttr(default=None)
    _kwargs: typing.Dict | None = pydantic.PrivateAttr(default=None)

    @property
    def kwargs(self):
        """Access _kwargs via property for backward compatibility"""
        return self._kwargs or {}

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
        if self.inst.is_method:
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
        if isinstance(self.engine, str):
            return object.__getattribute__(instance, self.engine)
        return self.engine

    def __get__(self, instance, owner):
        """Set the instance on the SignatureMethod

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        if self.inst.name not in instance.__dict__:
            instance.__dict__[self.inst.name] = self.spawn(instance=instance)
    
        return instance.__dict__[self.inst.name]
    
    def spawn(self, instance=None) -> Self:
        """
        Spawns a new instance of the current class (Func Decorator) with the specified arguments.
        Args:
            instance (optional): An optional argument to specify a new instance.
        Returns:
            Self: A new instance of the current class with the provided arguments.
        """

        new_inst = self.__class__(
            engine=self.engine,
            inst=self.inst,
            to_msg=self.to_msg
        )
        new_inst._instance = instance
        new_inst._kwargs = self._kwargs
        return new_inst

    def i(self, *args, **kwargs) -> str:
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
        return self.inst(
            instance, *args, **kwargs
        )


class FuncDec(FuncDecBase):
    """
    A class that allows one to decorate a function with an "instruction" so that the function will be an LLM (Language Model) call.

    Example:
        @instructfunc(engine=llm_engine)
        def my_function(param1: str, param2: int) -> str:
            return f"Process {param1} with number {param2}"
        result = my_function("example", 42)
        print(result)  # Output from the LLM based on the instruction
    """
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
        cue = self.inst(
            instance, *args, **kwargs
        )
        msg = self.to_msg(cue)
        engine = self.get_engine(instance)

        text, _ = engine(msg, **self.kwargs)
        result = self.inst.out_conv.forward(text)
        return result

    def spawn(self, instance=None) -> Self:
        """
        Spawns a new instance of the current class, optionally updating the instance.
        Args:
            instance (optional): The new instance to be used. If not specified,
                the current instance is retained.
        Returns:
            Self: A new instance of the current class with updated attributes.
        """
        new_inst = self.__class__(
            engine=self.engine,
            inst=self.inst,
            to_msg=self.to_msg
        )
        new_inst._instance = instance
        new_inst._kwargs = self._kwargs
        return new_inst
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class AFuncDec(FuncDecBase):
    """
    A class that allows one to decorate an async function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

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
        cue = self.inst(
            instance, *args, **kwargs
        )
        engine = self.get_engine(instance)
        res_msg = await engine.aforward(
            cue.text, **self.kwargs
        )
        result = self.inst.out_conv.forward(res_msg.text)
        return result
    
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

        new_inst = self.__class__(
            engine=self.engine,
            inst=self.inst,
            to_msg=self.to_msg
        )
        new_inst._instance = instance
        new_inst._kwargs = self._kwargs
        return new_inst
    

class StreamDec(FuncDecBase, StreamProcess):
    """
    A class that allows one to decorate a streaming function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    # engine: StreamProcess
    # inst: IBase
    # to_msg: ToPrompt

    # def __init__(
    #     self, 
    #     engine: StreamProcess,
    #     inst: IBase, 
    #     to_msg: ToPrompt=None,
    #     instance=None,
    #     kwargs: typing.Dict=None
    # ):
    #     super().__init__(
    #         engine, inst, to_msg, instance, kwargs
    #     )

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
        cue = self.inst(
            instance, *args, **kwargs
        )
        msg = self.to_msg(cue)
        engine = self.get_engine(
            instance
        )

        delta_store = {}
        for text, _ in engine.stream(msg, **self.kwargs):
            result = self.inst.out_conv.delta(text, delta_store, is_last=False)
            yield result

    def spawn(self, instance=None) -> Self:
        """
        Spawns a new instance of the class.
        If an instance is specified, it will create that instance.
        Args:
            instance (optional): The instance to be created. Defaults to None.
        Returns:
            Self: A new instance of the class.
        """
        new_inst = self.__class__(
            engine=self.engine,
            inst=self.inst,
            to_msg=self.to_msg
        )
        new_inst._instance = instance
        new_inst._kwargs = self._kwargs
        return new_inst
    
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


class AStreamDec(FuncDecBase, AsyncStreamProcess):
    """
    A class that allows one to decorate a asynchronous streaming function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

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
        cue = self.inst(
            instance, *args, **kwargs
        )
        msg = self.to_msg(cue)

        engine = self.get_engine(
            instance
        )

        delta_store = {}
        async for resp in engine.astream(
            msg, **self.kwargs
        ):
            is_last = resp.data['response'] == END_TOK
            result = self.inst.out_conv.delta(resp.delta.text, delta_store, is_last)
            resp.out = result
            yield resp.out

    def spawn(self, instance=None) -> Self:

        new_inst = self.__class__(
            engine=self.engine,
            inst=self.inst,
            to_msg=self.to_msg
        )
        new_inst._instance = instance
        new_inst._kwargs = self._kwargs
        return new_inst

    async def __call__(self, *args, **kwargs):
        """
        Alias for astream(). Asynchronously streams the execution of an instruction by calling the LLM.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Yields:
            The delta of the response from the LLM engine.
        """
        
        async for res in self.astream(
            *args, **kwargs
        ):
            yield res


def instructfunc(
    engine: Engine=None, 
    is_method: bool=False,
    to_async: bool=False,
    to_stream: bool=False,
    kwargs: typing.Dict=None
):
    """Decorate a method with instructfunc

    Args:
        engine (AIModel, optional): The engine for the AI . Defaults to None.
        is_method (bool): Whether it is a method

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    kwargs = kwargs or {}
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        inst = InstF(
            f=f, is_method=is_method
        )

        if not to_async and not to_stream:
            dec = FuncDec(engine=engine, inst=inst, to_msg=NullToPrompt())
            dec._kwargs = kwargs
            return dec
        if not to_stream:
            dec = AFuncDec(engine=engine, inst=inst, to_msg=NullToPrompt())
            dec._kwargs = kwargs
            return dec
        if not to_async:
            dec = StreamDec(engine=engine, inst=inst, to_msg=NullToPrompt())
            dec._kwargs = kwargs
            return dec
        dec = AStreamDec(engine=engine, inst=inst, to_msg=NullToPrompt())
        dec._kwargs = kwargs
        return dec
    return _


def instructmethod(
    engine: Engine=None, 
    to_async: bool=False,
    to_stream: bool=False,
    kwargs: typing.Dict=None
):
    """Decorate a method with instructfunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return instructfunc(
        engine, is_method=True, to_async=to_async, 
        to_stream=to_stream,
        kwargs=kwargs
    )


def signaturefunc(
    engine: Engine=None, 
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    is_method: bool=False,
    to_async: bool=False,
    to_stream: bool=False,
    train: bool=False,
    kwargs: typing.Dict=None
):
    """Decorate a method with instructfunc

    Args:
        engine (AIModel, optional): The engine for the AI . Defaults to None.
        is_method (bool): Whether it is a method

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    kwargs = kwargs or {}
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        inst = SigF(
            f=f, is_method=is_method, train=train,
            doc=doc or ''
        )

        if not to_async and not to_stream:
            dec = FuncDec(engine=engine, inst=inst, to_msg=NullToPrompt())
            dec._kwargs = kwargs
            return dec
        if not to_stream:
            dec = AFuncDec(engine=engine, inst=inst, to_msg=NullToPrompt())
            dec._kwargs = kwargs
            return dec
        if not to_async:
            dec = StreamDec(engine=engine, inst=inst, to_msg=NullToPrompt())
            dec._kwargs = kwargs
            return dec
        dec = AStreamDec(engine=engine, inst=inst, to_msg=NullToPrompt())
        dec._kwargs = kwargs
        return dec

    return _


def signaturemethod(
    engine: Engine=None, 
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    to_async: bool=False,
    to_stream: bool=False,
    train: bool=False,
    kwargs: typing.Dict=None
):
    """Decorate a method with SignatureFunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.
        doc (typing.Union[str, typing.Callable[[], str]], optional): A docstring to override with. Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return signaturefunc(
        engine, doc=doc, is_method=True,
        to_async=to_async, to_stream=to_stream,
        train=train,
        kwargs=kwargs
    )


class TemplateFormatter(Process):
    """Formats a prompt template with objective, constraints, and evaluations.
    """

    prompt_template: str
    to_render: bool = True
    style: typing.Dict[str, typing.Any] | None = None

    def forward(self, **kwargs) -> str:
        if self.to_render:
            kwargs = {
                k: render(v, False) for k, v in kwargs.items()
            }
        if self.style is not None:
            return style_formatter(
                self.prompt_template, _style=self.style, **kwargs
            )
        return self.prompt_template.format(
            **kwargs
        )
