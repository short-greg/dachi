# 1st party
import typing
from abc import abstractmethod, ABC
from typing import Self
import inspect
from functools import wraps

# 3rd party
import pydantic

# local
from .._core._process import (
    Module, 
    AsyncModule, StreamModule, AsyncStreamModule
)
from .._core._param import Param
from ..adapt._ai import (
    AsyncLLM, LLM, LLMBase,
    StreamLLM, AsyncStreamLLM, ToMsg,
    ToText
)
from .._core._param import Trainable
from .._core._core import Renderable
from ..utils import is_primitive

from ..adapt._read import TextProc, NullTextProc
from ..utils._utils import str_formatter


S = typing.TypeVar('S', bound=pydantic.BaseModel)

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
    out: typing.Optional[TextProc] = None

    def __init__(self, text: str, name: str='', out: typing.Optional[TextProc] = None):

        super().__init__(text=text, name=name, out=out)

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

    def read(self, data: str) -> S:
        """Read the data

        Args:
            data (str): The data to read

        Raises:
            RuntimeError: If the cue does not have a reader

        Returns:
            S: The result of the read process
        """
        if self.out is None:
            return data
            # raise RuntimeError(
            #     "Out has not been specified so can't read it"
            # )
        
        return self.out(data)

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

def validate_out(cues: typing.List[X]) -> typing.Optional[TextProc]:
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
        if out is None and cue.out is not None:
            out = cue.out
        elif cue.out is not None:
            raise RuntimeError(f'Out cannot be duplicated')
    return out


class IBase(ABC):

    def __init__(self, f, is_method: bool=False):
        
        self._f = f
        self._is_method = is_method
        self._docstring = f.__doc__
        self._name = f.__name__
        self._signature = str(inspect.signature(f))
        self._parameters = inspect.signature(f).parameters
        self._return_annotation = inspect.signature(f).return_annotation

    def _align_params(self, *args, **kwargs) -> typing.Dict:

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
        return self._name

    @property
    def f(self) -> typing.Callable:
        return self._f

    @property
    def is_method(self) -> typing.Callable:
        return self._is_method
    
    @abstractmethod
    def __call__(self, *args, **kwds) -> Cue:
        pass

    
class Inst(IBase):

    def __call__(self, instance, *args, **kwargs) -> Cue:
        
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


class Sig(IBase):

    def __init__(
        self, f, is_method = False, 
        train: bool=False
    ):
        super().__init__(f, is_method)
        cue = Cue(
            text=self._docstring
        )
        self._doc_param = Param(
            name=self._name,
            data=cue,
            training=train
        )

    def __call__(self, instance, *args, **kwargs) -> str:
        
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
            kwargs['TEMPLATE'] = self._reader.template()

        cue = Cue(text=str_formatter(
            doc, required=False, **kwargs
        ), name=self._name)
        return cue


class FuncDecBase:

    def __init__(self, inst: IBase, instance=None):
        """

        Args:
            inst (IBase): _description_
            instance (_type_, optional): _description_. Defaults to None.
        """
        self._inst = inst
        self._instance = instance

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def get_instance(self, args):

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
        
        return self.__class__(
            inst=self._inst, 
            instance=instance
        )

    def i(self, *args, **kwargs) -> Cue:
        instance, args = self.get_instance(args)
        return self._inst(
            instance, *args, **kwargs
        )


class FuncDec(FuncDecBase, Module):

    def __init__(
        self, inst, engine: LLM, 
        reader: TextProc=None, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        super().__init__(inst, instance)
        self._engine = engine
        self._reader = reader or NullTextProc()
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    def forward(self, *args, **kwargs):
        
        instance, args = self.get_instance(args)
        print(self._inst, type(self._inst))
        cue = self._inst(
            instance, *args, **kwargs
        )
        print(cue)
        msg = self._to_msg(cue)
        engine = self.get_engine(instance)
        _, res = engine(
            [msg], **self._kwargs
        )
        return self._reader(res)

    def spawn(self, instance=None) -> Self:
        
        return self.__class__(
            inst=self._inst, reader=self._reader,
            engine=self._engine, instance=instance,
            to_msg=self._to_msg
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



class AFuncDec(FuncDecBase, AsyncModule):

    def __init__(
        self, inst, engine: AsyncLLM, 
        reader: TextProc=None, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        super().__init__(inst, instance)
        self._engine = engine
        self._reader = reader or NullTextProc()
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    async def aforward(self, *args, **kwargs):
        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue)
        engine = self.get_engine(instance)
        _, res = await engine.aforward(
            [msg], **self._kwargs
        )
        return self._reader(res)
    
    async def __call__(self, *args, **kwargs):
        return await self.aforward(*args, **kwargs)

    def spawn(self, instance=None) -> Self:
        
        return self.__class__(
            inst=self._inst, reader=self._reader,
            engine=self._engine, instance=instance,
            to_msg=self._to_msg
        )

AFuncDec.__call__ = AFuncDec.aforward


class StreamDec(FuncDecBase, StreamModule):
    
    def __init__(
        self, inst, engine: StreamLLM, 
        reader: TextProc=None, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        super().__init__(inst, instance)
        self._engine = engine
        self._reader = reader or NullTextProc()
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    def stream(self, *args, **kwargs):
        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue)
        engine = self.get_engine(
            instance
        )
        delta_store = {}
        for _, res in engine.stream(
            [msg], **self._kwargs
        ):
            yield self._reader.delta(res, delta_store)

    def spawn(self, instance=None) -> Self:
        
        return self.__class__(
            inst=self._inst, reader=self._reader,
            engine=self._engine, instance=instance
        )


StreamDec.__call__ = StreamDec.stream


class AStreamDec(FuncDecBase, AsyncStreamModule):

    def __init__(
        self, inst, engine: AsyncStreamLLM, 
        reader: TextProc=None, 
        to_msg: ToMsg=None,
        kwargs: typing.Dict=None,
        instance=None
    ):
        super().__init__(inst, instance)
        self._engine = engine
        self._reader = reader or NullTextProc()
        self._to_msg = to_msg or ToText()
        self._kwargs = kwargs or {}

    async def astream(self, *args, **kwargs):
        instance, args = self.get_instance(args)
        cue = self._inst(
            instance, *args, **kwargs
        )
        msg = self._to_msg(cue)

        engine = self.get_engine(
            instance
        )
        async for _, res in await engine.astream(
            [msg], **self._kwargs
        ):
            yield self._reader.delta(res)

    def spawn(self, instance=None) -> Self:
        
        return self.__class__(
            inst=self._inst, reader=self._reader,
            engine=self._engine, instance=instance,
            to_msg=self._to_msg
        )

AStreamDec.__call__ = AStreamDec.astream


def instructfunc(
    engine: LLMBase=None,
    reader: TextProc=None,
    is_method: bool=False,
    is_async: bool=False,
    is_stream: bool=False,
    to_msg: ToMsg=None,
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
        
        inst = Inst(f, is_method)

        if not is_async and not is_stream:
            return FuncDec(
                inst, engine, reader, to_msg,
                kwargs=kwargs
            )
        if not is_stream:
            return AFuncDec(
                inst, engine, reader, to_msg,
                kwargs=kwargs
            )
        if not is_async:
            return StreamDec(
                inst, engine, reader, to_msg,
                kwargs=kwargs
            )
        return AStreamDec(
            inst, engine, reader, to_msg,
                kwargs=kwargs
        )

    return _


def instructmethod(
    engine: LLMBase=None,
    reader: TextProc=None,
    is_async: bool=False,
    is_stream: bool=False,
    to_msg: ToMsg=None,
    **kwargs
):
    """Decorate a method with instructfunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return instructfunc(
        engine, reader=reader, is_method=True, to_msg=to_msg, is_async=is_async, 
        is_stream=is_stream, 
        **kwargs
    )


def signaturefunc(
    engine: LLMBase=None,
    reader: TextProc=None,
    to_msg: ToMsg=None,
    is_method: bool=False,
    is_async: bool=False,
    is_stream: bool=False,
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
        
        inst = Sig(f, is_method)

        if not is_async and not is_stream:
            return FuncDec(
                inst, engine, reader, to_msg,
                kwargs=kwargs
            )
        if not is_stream:
            return AFuncDec(
                inst, engine, reader, to_msg,
                kwargs=kwargs
            )
        if not is_async:
            return StreamDec(
                inst, engine, reader, to_msg,
                kwargs=kwargs
            )
        return AStreamDec(
            inst, engine, reader, to_msg,
                kwargs=kwargs
        )

    return _


def signaturemethod(
    engine: LLMBase=None, 
    reader: TextProc=None,
    to_msg: ToMsg=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    is_async: bool=False,
    is_stream: bool=False,
    **kwargs
):
    """Decorate a method with SignatureFunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.
        reader (Reader, optional): The reader to use for the method. Defaults to None.
        doc (typing.Union[str, typing.Callable[[], str]], optional): A docstring to override with. Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return signaturefunc(
        engine, reader=reader, doc=doc, is_method=True,
        to_msg=to_msg, is_async=is_async, is_stream=is_stream,
        **kwargs
    )

