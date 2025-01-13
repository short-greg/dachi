# 1st party
import typing
from functools import wraps
import inspect
from itertools import chain
from typing import Any, Iterator, AsyncIterator
import inspect

# local
from .._core._core import (
    Cue, Param,
    Instruct, Reader
)
from ._ai import LLM

from ..utils._utils import (
    str_formatter
)
from .._core._process import Module


# TODO: SIMPLIFY THESE FUNCTIONS

def is_async_function(func) -> bool:
    """Check if a function is asynchronous."""
    return inspect.iscoroutinefunction(func)
from typing import Any, get_type_hints


def get_return_type(func) -> Any:
    """Get the return type of a function."""
    type_hints = get_type_hints(func)
    return type_hints.get('return', None)


def is_generator_function(func) -> bool:
    """Check if a function is a generator."""
    return inspect.isgeneratorfunction(func)


def get_iterator_type(func) -> Any:
    """
    Get the type of items yielded by an iterator function.
    Works for Iterator or AsyncIterator.
    """
    return_type = get_return_type(func)
    if return_type and hasattr(return_type, '__origin__'):
        if issubclass(return_type.__origin__, Iterator):
            return return_type.__args__[0]  # Type of the iterator
        elif issubclass(return_type.__origin__, AsyncIterator):
            return return_type.__args__[0]  # Type of the async iterator
    return None


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
        self._name = f.__name__
        self._signature = str(inspect.signature(f))
        self._instance = instance
        self._parameters = inspect.signature(f).parameters
        self._return_annotation = inspect.signature(f).return_annotation
    
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

        instance, args = self._get_instance(args)

        if instance is not None:
            param_values = param_values[1:]

        for value, param in zip(args, param_values):
            values[param.name] = value
        
        for k, value in kwargs.items():
            param = self._parameters[k]
            values[param.name] = value

        return values
            
    def get_instance(self, args):
        """Get the instance for the function

        Args:
            args: The arguments to use

        Returns:
            tuple: The instance and the arguments
        """
        if not self._is_method:
            return None, args
        
        if self._instance is not None:
            return self._instance, args
        return args[0], args[1:]
    
    def get_member(self, member: str, args):
        """Get the member for the function

        Args:
            member: The member to get
            args: The arguments to use

        Returns:
            The member
        """
        instance, _ = self.get_instance(args)

        if instance is None:
            raise RuntimeError('Cannot get member of ')

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
        if instance is None:
            return self.f(*args, **kwargs)
        return self.f(instance, *args, **kwargs)


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
        for d in self._f.stream(*args, **kwargs):
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
        return self._f.forward(*args, **kwargs)
    

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
        self, ifunc: IFunc, engine: typing.Callable[[typing.Any], typing.Any]=None, 
        reader: typing.Optional[Reader]=None,
        doc: typing.Union[str, typing.Callable[[], str]]=None,
        is_method: bool=False,
        train: bool=False, 
        ai_kwargs: typing.Dict=None,
        instance=None,
    ):
        """Wrap the signature method with a particular engine and
        dialog factory

        Args:
            f (typing.Callable): The function to wrap
            engine (AIModel): The engine to use for getting the response 
            train (bool, optional): Whether to train the cues or not. Defaults to False.
            instance (optional): The instance. Defaults to None.
        """
        super().__init__(
            ifunc, engine, reader, ai_kwargs, instance
        )
        self._ifunc = ifunc
        self._train = train
        if doc is not None and not isinstance(doc, str):
            doc = doc()
        self._doc = doc if doc is not None else ifunc.docstring
        self._is_method = is_method
        docstring = Cue(text=docstring)
        self._docstring = Param(
            name=self.name,
            cue=docstring,
            training=train
        )
        self._ai_kwargs = ai_kwargs
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
        if isinstance(self.engine, str):
            return object.__getattribute__(instance, self.engine)
        return self._instance

    def __get__(self, instance, owner):
        """Set the instance on the SignatureMethod

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        if self.f.__name__ not in instance.__dict__:
            instance.__dict__[self.f.__name__] = SignatureFunc(
                self._ifunc,
                engine=self._engine,
                reader=self._reader,
                is_method=self._is_method, 
                train=self._train,
                ai_kwargs=self._ai_kwargs,
                instance=self._instance
            )
    
        return instance.__dict__[self.f.__name__]

    def _prepare(self, *args, **kwargs):
        """Prepare the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The cue
        """
        instance, args = self.get_instance(args)

        cur_kwargs = self._ifunc(instance, *args, **kwargs)
        kwargs = {**kwargs, **cur_kwargs}
        params = self._ifunc.fparams(instance, *args, **kwargs)
        if "TEMPLATE" in doc:
            params['TEMPLATE'] = self._reader.template()
        doc = self._docstring.render()
        cue = str_formatter(
            doc, required=False, **params
        )
        return cue

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
        cue = self._prepare(*args, **kwargs)
        res = engine(cue, **self._ai_kwargs)
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
        cue = self._prepare(*args, **kwargs)
        if isinstance(engine, Module):
            res = await engine.aforward(cue, **self._ai_kwargs)
        else:
            res = await engine(cue, **self._ai_kwargs)
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
        cue = self._prepare(*args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        for v in f(cue, **self._ai_kwargs):
            if self._reader is not None:
                v = self._reader.read(v)
            yield v
        # return v

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
        cue = self._prepare(*args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        async for v in f.astream(cue, **self._ai_kwargs):
            if self._reader is not None:
                v = self._reader.read(v)
            yield v
        # return v

    def i(self, *args, **kwargs) -> Cue:
        """Get the cue for the function

        Args:
            args: The arguments to use
            kwargs: The keyword arguments to use

        Returns:
            The cue
        """ 
        cue = self._prepare(*args, **kwargs)

        return Cue(
            cue, self._name, out=self._reader
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
            ai_kwargs=self._ai_kwargs,
            instance=self._instance
        )


class InstructFunc(Module, Instruct):
    """SignatureFunc is a method where you define the cue in
    the function signature
    """
    def __init__(
        self, ifunc: IFunc, engine: typing.Callable[[typing.Any], typing.Any]=None, 
        reader: typing.Optional[Reader]=None,
        is_method: bool=False,
        train: bool=False, 
        ai_kwargs: typing.Dict=None,
        instance=None,
    ):
        """Wrap the signature method with a particular engine and
        dialog factory

        Args:
            f (typing.Callable): The function to wrap
            engine (AIModel): The engine to use for getting the response 
            train (bool, optional): Whether to train the cues or not. Defaults to False.
            instance (optional): The instance. Defaults to None.
        """
        super().__init__(
            ifunc, engine, reader, ai_kwargs, instance
        )
        self._ifunc = ifunc
        self._train = train
        self._engine = engine
        self._instance = instance
        self._ai_kwargs = ai_kwargs
        self._is_method = is_method
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
        if isinstance(self.engine, str):
            return object.__getattribute__(instance, self.engine)
        return self._instance

    def __get__(self, instance, owner):
        """Set the instance on the SignatureMethod

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        if self.f.__name__ not in instance.__dict__:
            instance.__dict__[self.f.__name__] = InstructFunc(
                self._ifunc, self.engine, 
                self._reader, self._train, self._ai_kwargs,
                self._is_method, self._instance
            )
        return instance.__dict__[self.f.__name__]
    
    def _prepare(self, *args, **kwargs):
        instance, args = self.get_instance(args)

        cur_kwargs = self._ifunc(instance, *args, **kwargs)
        kwargs = {**kwargs, **cur_kwargs}

        return self._ifunc(*args, **kwargs)

    def forward(self, *args, **kwargs) -> typing.Any:

        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare(*args, **kwargs)
        res = engine(cue, **self._ai_kwargs)
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
        cue = self._prepare(*args, **kwargs)
        if isinstance(engine, Module):
            res = await engine.aforward(cue, **self._ai_kwargs)
        else:
            res = await engine(cue, **self._ai_kwargs)
        if self._reader is not None:
            return self._reader.read(res)
        return res

    def stream(self, *args, **kwargs) -> typing.Any:
        """Stream the instruction function"""
        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare(*args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        for v in f(cue, **self._ai_kwargs):
            if self._reader is not None:
                v = self._reader.read(v)
            yield v
        # return v

    async def astream(self, *args, **kwargs) -> typing.Any:
        """Stream the instruction function asynchronously"""

        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare(*args, **kwargs)

        if isinstance(engine, Module):
            f = engine.stream
        else:
            f = engine
        async for v in f.astream(cue, **self._ai_kwargs):
            if self._reader is not None:
                v = self._reader.read(v)
            yield v
        # return v

    def i(self, *args, **kwargs) -> Cue:
        """Get the cue for the function"""
        return self._prepare(*args, **kwargs)

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
            ai_kwargs=self._ai_kwargs,
            instance=self._instance
        )


def instructfunc(
    engine: LLM=None,
    reader: Reader=None,
    is_method: bool=False,
    **ai_kwargs
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
            ifunc, engine, reader, is_method=is_method,
            ai_kwargs=ai_kwargs
        )
    return _


def instructmethod(
    engine: LLM=None,
    **ai_kwargs
):
    """Decorate a method with instructfunc

    Args:
        engine (PromptModel, optional): The engine for the AI . Defaults to None.

    Returns:
        typing.Callable[[function], SignatureFunc]
    """
    return instructfunc(
        engine, True, **ai_kwargs
    )


def signaturefunc(
    engine: LLM=None, 
    reader: Reader=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    is_method=False,
    **ai_kwargs
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
            ai_kwargs=ai_kwargs, 
            doc=doc, 
        )

    return _


def signaturemethod(
    engine: LLM=None, 
    reader: Reader=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    **ai_kwargs
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
        engine, reader, doc, True, **ai_kwargs
    )
