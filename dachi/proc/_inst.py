# 1st party
import typing
from abc import abstractmethod, ABC
from typing import Self
import inspect
from functools import wraps

from dataclasses import InitVar

# 3rd party
import pydantic

# local
from ..core import BaseModule
from ._process import (
    Process, AsyncProcess, StreamProcess, 
    AsyncStreamProcess
)
from ..core import Param
from ..utils import primitives, str_formatter
from ._msg import FromMsg
from ._out import (
    ToOut, ParseOut,
    PrimOut, PydanticOut,
)
Engine: typing.TypeAlias = Process | AsyncProcess | StreamProcess | AsyncStreamProcess

S = typing.TypeVar('S')
# TODO: MOVE OUT OF HERE


class IBase(BaseModule):
    """
    This is the base class for wrapping an Instruction functor. It is used to create an instruction when called.
    """

    f: typing.Callable
    is_method: bool = False
    out_conv: ToOut = None
    llm_out: str = 'content'

    def __post_init__(self):
        """ Initializes the IBase instance.
        Args:
            f (Callable): The function to be wrapped as an instruction.
            is_method (bool, optional): Indicates if the function is a method. Defaults to False.
            out (OutConv, optional): The output converter that converts the output of the LLM into a more useful format. Defaults to None.
        """
        self._docstring = self.f.__doc__
        self._name = self.f.__name__
        self._signature = str(inspect.signature(self.f))
        self._parameters = inspect.signature(self.f).parameters
        self._return_annotation = inspect.signature(self.f).return_annotation

        out_cls = self._return_annotation
        if out_conv is None:
            if out_cls in primitives:
                out_conv = PrimOut(
                    name='out', from_=self.llm_out,
                    out_cls=self.out_cls
                )
            elif issubclass(out_cls, pydantic.BaseModel):
                out_conv = PydanticOut(name='out', from_=self.llm_out, out_cls=out_cls)
            else:
                out_conv = ParseOut(name='out', from_=self.llm_out)

        self._out = FromMsg('out')

    def _align_params(
        self, *args, **kwargs
    ) -> typing.Dict:
        """
        Returns:
            typing.Dict: Get the parameters
        """
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

    @property
    def f(self) -> typing.Callable:
        """
        Returns:
            typing.Callable: The function wrapped
        """
        return self.f

    @property
    def is_method(self) -> typing.Callable:
        """
        Returns:
            typing.Callable: Whether the function wrapped is a method or a regular function
        """
        return self._is_method
    
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

    @property
    def out_conv(self) -> ToOut:
        return self._out_conv
    
    @property
    def from_msg(self) -> FromMsg:
        return self._out


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

    def model_post_init(
        self
    ):
        """
        Args:
            f (_type_): The function to wrap
            is_method (bool, optional): Whether f is a method. Defaults to False.
            doc (typing.Optional[str], optional): The docstring for the function. Defaults to None.
            train (bool, optional): Whether to train the instruction. Defaults to False.
            out (OutConv, optional): The out to process the output. Defaults to None.
        """
        super().model_post_init()
        self._doc = self._doc or self._docstring
        self._doc_param = Param[str](
            name=self._name,
            data=self._doc,
            training=self.train
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
        doc = self._doc_param.render()
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


class FuncDecBase(object):
    """This is used to decorate an "instruct" function
    that will be used 
    """

    def __init__(
        self, 
        inst: IBase, 
        instance=None,
        kwargs: typing.Dict=None
    ):
        super().__init__()
        self.instance = instance
        self.inst = inst
        self.kwargs = kwargs

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
        
        return self.__class__(
            inst=self.inst, 
            instance=instance
        )

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


class FuncDec(FuncDecBase, Process):
    """
    A class that allows one to decorate a function with an "instruction" so that the function will be an LLM (Language Model) call.
    """
    
    def __init__(
        self, 
        engine: Process,
        inst: IBase, 
        instance=None,
        kwargs: typing.Dict=None
    ):
        self.engine = engine
        super().__init__(
            inst, instance, kwargs
        )

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
        # msg = self._to_msg(cue.text)
        engine = self.get_engine(instance)

        res_msg = engine(
            cue, **self._kwargs
        )
        print(self.inst.out_conv)
        res_msg = self.inst.out_conv(res_msg)
        res, filtered = self.inst.from_msg.filter(res_msg)

        if filtered:
            return None
        return res

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
            inst=self.inst, 
            engine=self._engine, instance=instance,
            to_msg=self._to_msg, kwargs=self._kwargs
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class AFuncDec(FuncDecBase, AsyncProcess):
    """
    A class that allows one to decorate an async function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    def __init__(
        self, 
        engine: AsyncProcess,
        inst: IBase, 
        instance=None,
        kwargs: typing.Dict=None
    ):
        self.engine = engine
        super().__init__(
            inst, instance, kwargs
        )

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
            cue.text, **self._kwargs
        )
        res_msg = self.inst.out_conv(res_msg)
        res, filtered = self.inst.from_msg.filter(res_msg)
        if filtered:
            return None
        return res
    
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
            inst=self.inst,
            engine=self._engine, instance=instance,
            to_msg=self._to_msg, kwargs=self._kwargs
        )
    

class StreamDec(FuncDecBase, StreamProcess):
    """
    A class that allows one to decorate a streaming function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    def __init__(
        self, 
        engine: StreamProcess,
        inst: IBase, 
        instance=None,
        kwargs: typing.Dict=None
    ):
        self.engine = engine
        super().__init__(
            inst, instance, kwargs
        )

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
        msg = self._to_msg(cue)
        engine = self.get_engine(
            instance
        )

        for resp in engine.stream(
            [msg], **self._kwargs
        ):
            resp, filtered = self.inst.from_msg.filter(resp)
            if filtered:
                yield None
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
            inst=self.inst, 
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


class AStreamDec(FuncDecBase, AsyncStreamProcess):
    """
    A class that allows one to decorate a asynchronous streaming function with an "instruction" so that the function will be an LLM (Language Model) call.
    """

    def __init__(
        self, 
        engine: AsyncStreamProcess,
        inst: IBase, 
        instance=None,
        kwargs: typing.Dict=None
    ):
        self.engine = engine
        super().__init__(
            inst, instance, kwargs
        )

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
        msg = self._to_msg(cue)

        engine = self.get_engine(
            instance
        )

        async for resp_msg in engine.astream(
            [msg], **self._kwargs
        ):
            resp = self.inst.out_conv(resp_msg)
            resp, filtered = self.inst.from_msg.filter(resp)

            if filtered:
                yield None
            yield resp

    def spawn(self, instance=None) -> Self:
        
        return self.__class__(
            inst=self.inst, 
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
    is_method: bool=False,
    to_async: bool=False,
    to_stream: bool=False,
    out: typing.Tuple[str] | str = 'content',
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
            f, is_method, out=out
        )

        if not to_async and not to_stream:
            return FuncDec(
                inst, engine,
                kwargs=kwargs
            )
        if not to_stream:
            return AFuncDec(
                inst, engine,
                kwargs=kwargs
            )
        if not to_async:
            return StreamDec(
                inst, engine, 
                kwargs=kwargs
            )
        return AStreamDec(
            inst, engine, 
                kwargs=kwargs
        )
    return _


def instructmethod(
    engine: Engine=None, 
    to_async: bool=False,
    to_stream: bool=False,
    out: typing.Tuple[str] | str = 'content',
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
        llm_out=out,
        kwargs=kwargs
    )


def signaturefunc(
    engine: Engine=None, 
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    is_method: bool=False,
    to_async: bool=False,
    to_stream: bool=False,
    train: bool=False,
    out: typing.Tuple[str] | str = 'content',
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
            f, is_method, train=train, 
            doc=doc, out=out,
        )

        if not to_async and not to_stream:
            return FuncDec(
                inst, engine,
                kwargs=kwargs
            )
        if not to_stream:
            return AFuncDec(
                inst, engine,
                kwargs=kwargs
            )
        if not to_async:
            return StreamDec(
                inst, engine,
                kwargs=kwargs
            )
        return AStreamDec(
            inst, engine,
            kwargs=kwargs
        )

    return _


def signaturemethod(
    engine: Engine=None, 
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    to_async: bool=False,
    to_stream: bool=False,
    train: bool=False,
    out: typing.Tuple[str] | str = 'content',
    kwargs: typing.Dict=None
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
        engine, doc=doc, is_method=True,
        to_async=to_async, to_stream=to_stream,
        train=train,
        out=out,
        kwargs=kwargs
    )

