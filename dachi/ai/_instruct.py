# 1st party
import typing
from functools import wraps
import inspect
from itertools import chain
import inspect
from ._ai import ToMsg, ToText

# local
from .._core._core import (
    Cue, Param,
    Instruct, Reader
)
from ._ai import LLM
from .._core._messages import Msg
from ..utils._utils import (
    str_formatter
)
from .._core._process import Module
from ..utils._f_utils import (
    is_async_function, get_return_type,
    is_generator_function, get_iterator_type
)

X = typing.Union[str, Cue]


def validate_out(cues: typing.List[X]) -> typing.Optional[Reader]:
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


class InstructCall(Module, Instruct):
    """InstructCall is used within an instructf or
    signaturef to allow one to loop over "incoming"
    instructions
    """
    def __init__(self, i: Instruct, *args, **kwargs):
        """Create the "Call" passing in the cue
        and its inputs

        Args:
            i (Instruct): The cue to wrap
        """
        self._instruct = i
        self.args = args
        self.kwargs = kwargs

    def __iter__(self) -> typing.Iterator['InstructCall']:
        """Loop over the "sub" InstructCalls

        Yields:
            typing.Iterator['InstructCall']: The InstructCalls used
        """
        for arg in chain(self.args, self.kwargs.values()):
            if isinstance(arg, InstructCall):
                for arg_i in arg:
                    yield arg_i
                yield arg
        yield self

    def i(self) -> Cue:
        """Get the cue 

        Returns:
            Cue: The cue
        """
        return self._instruct.i()
        
    def forward(self) -> typing.Any:
        """Execute the cue

        Returns:
            typing.Any: Execute the Instruct
        """
        args = [
            arg() if isinstance(arg, InstructCall) 
            else arg for arg in self.args
        ]
        kwargs = {
            k: v() if isinstance(v, Instruct) else v
            for k, v in self.kwargs.items()
        }
        return self._instruct(*args, **kwargs)


class IFunc(object):
    """
    """
    def __init__(self, f, is_method: bool, instance=None):
        """Create a function wrapper for an instruct

        Args:
            f: The function to wrap
            is_method (bool): Whether the function is a method
            instance (optional): The instance if this is a method. Defaults to None.
        """
        self._f = f
        self._is_async = is_async_function(f)
        self._is_generator = is_generator_function(f)
        self._is_method = is_method
        self._is_generator = is_generator_function(f)
        self._is_async = is_async_function(f)
        self._docstring = f.__doc__
        self._name = f.__name__
        self._signature = str(inspect.signature(f))
        self._instance = instance
        self._parameters = inspect.signature(f).parameters
        self._return_annotation = inspect.signature(f).return_annotation
    
    def is_generator(self) -> bool:
        return self._is_generator

    def is_async(self) -> bool:
        return self._is_async
    
    @property
    def name(self) -> str:

        return self._name

    def fparams(self, instance, *args, **kwargs):
        """Get the parameters for the function

        Args:
            instance: The instance if this is a method
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            dict: The parameters
        """
        param_values = list(self._parameters.values())
        values = {}
        instance, args = self.get_instance(instance, args)

        if instance is not None:
            param_values = param_values[1:]

        for value, param in zip(args, param_values):
            values[param.name] = value
        
        for k, value in kwargs.items():
            param = self._parameters[k]
            values[param.name] = value

        return values
            
    def get_instance(self, instance, args):
        """Get the instance for the function

        Args:
            args: The arguments to use

        Returns:
            tuple: The instance and the arguments
        """
        if instance is not None:
            return instance, args
        if self._instance is not None:
            return self._instance, args
        
        if self._is_method:
            return args[0], args[1:]
        
        return None, args
    
    def get_member(self, instance, member: str, args):
        """Get the member for the function

        Args:
            member: The member to get
            args: The arguments to use

        Returns:
            The member
        """
        instance, _ = self.get_instance(instance, args)

        if instance is None:
            raise RuntimeError(
                'Cannot get member of instance'
            )

        return object.__getattribute__(instance, member)

    @property
    def docstring(self):
        """Get the docstring for the function

        Returns:
            str: The docstring
        """
        return self._docstring
    
    def __call__(self, *args, instance=None, **kwargs):
        """Call the function

        Args:
            args: The arguments to use
            instance: The instance if this is a method
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        instance, args = self.get_instance(instance, args)
        if instance is None or hasattr(self._f, "__self__"):

            return self._f(*args, **kwargs)
        # instance = self.get_instance(instance, args)
        return self._f(instance, *args, **kwargs)


class ModuleIFunc(IFunc):
    """ModuleIFunc is a function wrapper for an instruct that is a module
    """
    async def aforward(self, *args, **kwargs):
        """Execute the function asynchronously

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        return await self._f.aforward(*args, **kwargs)
    
    def stream(self, *args, **kwargs):
        """Stream the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        for _, d in self._f.stream(*args, **kwargs):
            yield d
    
    async def astream(self, *args, **kwargs):
        """Stream the function asynchronously

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        async for d in await self._f.astream(*args, **kwargs):
            yield d

    def forward(self, *args, **kwargs):
        """Execute the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        return self._forward(*args, **kwargs)
    

class FIFunc(IFunc):
    """FIFunc is a function wrapper for an instruct that is not a module
    """
    async def aforward(self, *args, **kwargs):
        """Execute the function asynchronously

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        return await self._f(*args, **kwargs)
    
    def stream(self, *args, **kwargs):
        """Stream the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        for d in self._f(*args, **kwargs):
            yield d
    
    async def astream(self, *args, **kwargs):
        
        async for d in await self._f(*args, **kwargs):
            yield d

    def forward(self, *args, **kwargs):
        return self._f(*args, **kwargs)


class SignatureFunc(Module, Instruct):
    """SignatureFunc is a method where you define the cue in
    the function signature
    """
    def __init__(
        self, ifunc: IFunc, engine: LLM=None, 
        reader: typing.Optional[Reader]=None,
        doc: typing.Union[str, typing.Callable[[], str]]=None,
        is_method: bool=False,
        train: bool=False, 
        instance=None,
        to_msg: ToMsg=None
    ):
        """Wrap the signature method with a particular engine and
        dialog factory

        Args:
            f (typing.Callable): The function to wrap
            engine (AIModel): The engine to use for getting the response 
            train (bool, optional): Whether to train the cues or not. Defaults to False.
            instance (optional): The instance. Defaults to None.
        """
        super().__init__()
        self._engine = engine
        self._reader = reader
        self._instance = instance
        self._ifunc = ifunc
        self._train = train
        if doc is not None and not isinstance(doc, str):
            doc = doc()
        self._doc = doc if doc is not None else ifunc.docstring
        self._is_method = is_method
        docstring = Cue(text=self._doc)
        self._docstring = Param(
            name=self._ifunc.name,
            cue=docstring,
            training=train
        )
        self._conv_msg = to_msg or ToText()
        if self._ifunc.is_generator() and self._ifunc.is_async():
            self.__call__ = self.astream
        elif self._ifunc.is_async():
            self.__call__ = self.aforward
        elif self._ifunc.is_generator():
            self.__call__ = self.stream
        else:
            self.__call__ = self.forward

    def get_instance(self, args):
        """Get the instance for the function

        Args:
            args: The arguments to use

        Returns:
            tuple: The instance and the arguments
        """
        if not self._is_method:
            return None, args
        if self._instance is None:
            return args[0], args[1:]
        return self._instance, args

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
        if self._ifunc.name not in instance.__dict__:
            instance.__dict__[self._ifunc.name] = SignatureFunc(
                self._ifunc,
                engine=self._engine,
                reader=self._reader,
                is_method=self._is_method, 
                train=self._train,
                instance=instance
            )
    
        return instance.__dict__[self._ifunc.name]

    def _prepare(self, instance, *args, **kwargs):
        """Prepare the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The cue
        """
        params = self._ifunc.fparams(
            instance, *args, **kwargs
        )
        cur_kwargs = self._ifunc(
            *args, instance=instance, **kwargs
        )
    
        cur_kwargs = cur_kwargs if cur_kwargs is not None else {}
        # kwargs = {**kwargs, **cur_kwargs}
        cur_kwargs.update(params)
        
        if "TEMPLATE" in self._docstring:
            cur_kwargs['TEMPLATE'] = self._reader.template()
        doc = self._docstring.render()
        cue = str_formatter(
            doc, required=False, **cur_kwargs
        )
        return cue
    
    def _prepare_msg(self, instance, *args, **kwargs) -> typing.Any:
        """
        """
        i = self._prepare(instance, *args, **kwargs)
        return self._conv_msg(i)

    def forward(self, *args, **kwargs) -> typing.Any:
        """Execute the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare_msg(instance, *args, **kwargs)
        _, res = engine(cue)
        if self._reader is not None:
            return self._reader.read(res)
        return res

    async def aforward(self, *args, **kwargs) -> typing.Any:
        """Execute the function asynchronously

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare_msg(instance, *args, **kwargs)
        if isinstance(engine, Module):
            _, res = await engine.aforward(cue)
        else:
            _, res = await engine(cue)
        if self._reader is not None:
            return self._reader.read(res)
        return res

    def stream(self, *args, **kwargs) -> typing.Any:
        """Stream the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare_msg(instance, *args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        for _, v in f(cue):
            if self._reader is not None:
                v = self._reader.read(v)
            yield v

    async def astream(self, *args, **kwargs) -> typing.Any:
        """Stream the function asynchronously

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare_msg(instance, *args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        async for _, v in f.astream(cue):
            if self._reader is not None:
                v = self._reader.read(v)
            yield v

    def i(self, *args, **kwargs) -> Cue:
        """Get the cue for the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The cue
        """ 
        instance, args = self.get_instance(args)
        cue = self._prepare(instance, *args, **kwargs)
        return Cue(
            cue, self._ifunc.name, 
            out=self._reader
        )
    
    def spawn(
        self, 
        engine: LLM=None, 
        train: bool=False
    ) -> 'SignatureFunc':
        """Spawn a new SignatureMethod. Especially use to create a trainable one

        Args:
            engine (AIModel, optional): Spawn a new . Defaults to None.
            train (bool, optional): _description_. Defaults to False.

        Returns:
            SignatureMethod: 
        """
        return SignatureFunc(
            self._ifunc,
            engine=engine or self._engine,
            reader=self._reader,
            is_method=self._is_method, 
            train=train if train is not None else self._train,
            instance=self._instance
        )


class InstructFunc(Instruct, Module):
    """SignatureFunc is a method where you define the cue in
    the function signature
    """
    def __init__(
        self, ifunc: IFunc, 
        engine: LLM=None, 
        reader: typing.Optional[Reader]=None,
        is_method: bool=False,
        train: bool=False, 
        instance=None,
        to_msg: ToMsg=None
    ):
        """Wrap the signature method with a particular engine and
        dialog factory

        Args:
            f (typing.Callable): The function to wrap
            engine (AIModel): The engine to use for getting the response 
            train (bool, optional): Whether to train the cues or not. Defaults to False.
            instance (optional): The instance. Defaults to None.
        """
        super().__init__()
        self._ifunc = ifunc
        self._train = train
        self._engine = engine
        self._instance = instance
        self._reader = reader
        self._is_method = is_method
        self._conv_msg = to_msg or ToText()
        if self._ifunc.is_generator() and self._ifunc.is_async():
            self.__call__ = self.astream
        elif self._ifunc.is_async():
            self.__call__ = self.aforward
        elif self._ifunc.is_generator():
            self.__call__ = self.stream
        else:
            self.__call__ = self.forward

    def get_instance(self, args):
        """Get the instance for the function

        Args:
            args: The arguments to use

        Returns:
            tuple: The instance and the arguments
        """
        if self._instance is not None:
            return self._instance, args
        
        if self._is_method is True:
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
        if self._ifunc.name not in instance.__dict__:
            instance.__dict__[self._ifunc.name] = InstructFunc(
                self._ifunc, self._engine, 
                self._reader, self._train, 
                self._is_method, instance,
                self._conv_msg
            )
        return instance.__dict__[self._ifunc.name]
    
    def _prepare(self, instance, *args, **kwargs) -> Msg:
        # instance, args = self.get_instance(args)

        # cur_kwargs = self._ifunc(instance, *args, **kwargs)
        # kwargs = {**kwargs, **cur_kwargs}
        res = self._ifunc(*args, instance=instance, **kwargs)
        print('Instruction: ', res)
        return res

    def _prepare_msg(self, instance, *args, **kwargs) -> typing.Tuple[Cue, Msg]:
        """
        """
        cue = self._prepare(instance, *args, **kwargs)
        return cue, self._conv_msg(
            cue.text
        )

    def forward(self, *args, **kwargs) -> typing.Any:

        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue, msg = self._prepare_msg(instance, *args, **kwargs)
        _, res = engine(msg)
        if self._reader is not None:
            return self._reader.read(res)
        return cue.read(res)

    async def aforward(self, *args, **kwargs) -> typing.Any:
        """Execute the function asynchronously

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The result of the function
        """
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue, msg = self._prepare_msg(instance, *args, **kwargs)
        if isinstance(engine, Module):
            _, res = await engine.aforward(cue)
        else:
            _, res = await engine(cue)
        if self._reader is not None:
            return self._reader.read(res)
        return cue.read(res)

    def stream(self, *args, **kwargs) -> typing.Any:
        """Stream the instruction function"""
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue, msg = self._prepare_msg(instance, *args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        # print('Cue: ', cue)
        for _, v in f(msg):
            if self._reader is not None:
                v = self._reader.read(v)
            else:
                v = cue.read(v)
            yield v

    async def astream(self, *args, **kwargs) -> typing.Any:
        """Stream the instruction function asynchronously"""

        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue, msg = self._prepare_msg(instance, *args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        async for _, v in f.astream(msg):
            # if  is not None:
            if self._reader is not None:
                v = self._reader.read(v)
            else:
                v = cue.read(v)
            yield v

    def i(self, *args, **kwargs) -> Cue:
        """Get the cue for the function"""
        instance, args = self.get_instance(args)
        return self._prepare(instance, *args, **kwargs)

    def spawn(
        self, 
        engine: LLM=None, 
        train: bool=None
    ) -> 'InstructFunc':
        """Spawn a new SignatureMethod. Especially use to create a trainable one

        Args:
            engine (AIModel, optional): Spawn a new . Defaults to None.
            train (bool, optional): _description_. Defaults to False.

        Returns:
            SignatureMethod: 
        """
        return InstructFunc(
            self._ifunc, 
            engine=engine or self._engine,
            reader=self._reader,
            is_method=self._is_method, 
            train=train if train is not None else self._train,
            instance=self._instance
        )


def instructfunc(
    engine: LLM=None,
    reader: Reader=None,
    is_method: bool=False,
    to_msg: ToMsg=None
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
        
        # TODO: Use wrapper
        if isinstance(f, Module):
            ifunc = ModuleIFunc(f, is_method)
        else:
            ifunc = FIFunc(f, is_method)

        return InstructFunc(
            ifunc, engine, reader, is_method=is_method, to_msg=to_msg
        )
    return _


def instructmethod(
    engine: LLM=None,
    reader: Reader=None,
    to_msg: ToMsg=None
):
    """Decorate a method with instructfunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return instructfunc(
        engine, reader=reader, is_method=True, to_msg=to_msg
    )


def signaturefunc(
    engine: LLM=None,
    reader: Reader=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    is_method=False,
    to_msg: ToMsg=None
):
    """Decorate a method with SignatureFunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.
        reader (Reader, optional): The reader to use for the method. Defaults to None.
        doc (typing.Union[str, typing.Callable[[], str]], optional): A docstring to override with. Defaults to None.
        is_method (bool): Whether the function is a method. 

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        if isinstance(f, Module):
            ifunc = ModuleIFunc(f, is_method, doc)
        else:
            ifunc = FIFunc(f, is_method, doc)
        
        return SignatureFunc(
            ifunc, engine, reader, is_method=is_method,
            doc=doc, to_msg=to_msg
        )

    return _


def signaturemethod(
    engine: LLM=None, 
    reader: Reader=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    to_msg: ToMsg=None
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
        to_msg=to_msg
    )
