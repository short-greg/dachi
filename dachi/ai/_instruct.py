# 1st party
import typing
from functools import wraps, update_wrapper
import inspect
from itertools import chain

from typing import Any, Iterator, AsyncIterator
import inspect

import pydantic

# local
from .._core._core import (
    render, 
    Cue, render, Param, 
    Instruct, Reader, NullRead,
)
from ._ai import LLM

from .._core import (
    Dialog, TextMessage
)
from .._core._read import (
    PydanticRead, PrimRead
)
from ..utils._utils import (
    str_formatter, primitives, get_member
)
from .._core._process import Module

import inspect

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


# Make this a cue
# and a parameter


X = typing.Union[str, Cue]

# UPDATE THE FOLLOWING

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

        if not self._is_method:
            return None, args
        
        if self._instance is not None:
            return self._instance, args
        return args[0], args[1:]
    
    def get_member(self, member: str, args):
        
        instance, _ = self.get_instance(args)

        if instance is None:
            raise RuntimeError('Cannot get member of ')

        return object.__getattribute__(instance, member)

    @property
    def docstring(self):
        return self._docstring
    
    def __call__(self, *args, instance=None, **kwargs):
        
        if instance is None:
            return self.f(*args, **kwargs)
        return self.f(instance, *args, **kwargs)
    
    # def fill_template(self, template: str, *args, reader: Reader=None, **kwargs):
    #     param_values = self.fparams(*args, **kwargs)
    #     filled = set()

    #     for param in param_values:
    #         if param.name in filled:
    #             continue
    #         if param.default == inspect.Parameter.empty:
    #             raise RuntimeError('Param has not been defined and no value')

class ModuleIFunc(IFunc):

    async def aforward(self, *args, **kwargs):

        return await self._f.aforward(*args, **kwargs)
    
    def stream(self, *args, **kwargs):

        for d in self._f.stream(*args, **kwargs):
            yield d
    
    async def astream(self, *args, **kwargs):
        async for d in await self._f.astream(*args, **kwargs):
            yield d

    def forward(self, *args, **kwargs):
        return self._f.forward(*args, **kwargs)
    

class FIFunc(IFunc):

    async def aforward(self, *args, **kwargs):
        return await self._f(*args, **kwargs)
    
    def stream(self, *args, **kwargs):

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

        # docstring = inspect.getdoc(f) if doc is None else doc
        # self._signature = str(inspect.signature(f))
        # self._parameters = inspect.signature(f).parameters
        # self._return_annotation = inspect.signature(f).return_annotation
        # if not isinstance(, typing.Callable):
        #     docstring = Cue(text=docstring)
        #     self._docstring = Param(
        #         name=self.name,
        #         cue=docstring,
        #         training=train
        #     )
        # elif train:
        #     raise ValueError('Cannot set to train if the docstring is a callable')
        # else:
        #     self._docstring = docstring
        # update_wrapper(self, f) 
    
    def get_instance(self, args):
        if not self._is_method:
            return None, args
        
        if self._instance is None:
            return args[0], args[1:]
        return self._instance, args

    def get_engine(self, instance):
        
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
            return SignatureFunc(
                self._ifunc,
                engine=self._engine,
                reader=self._reader,
                is_method=self._is_method, 
                train=self._train,
                ai_kwargs=self._ai_kwargs,
                instance=self._instance
            )
            # instance.__dict__[self.f.__name__] = SignatureFunc(
            #     self._ifunc, self.engine, self._doc,
            #     self._reader, self._train, self._ai_kwargs,
            #     self._is_method, self._instance
            # )
        return instance.__dict__[self.f.__name__]

    def _prepare(self, *args, **kwargs):
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

        instance, args = self.get_instance(args)
        engine = self.get_engine(instance)
        cue = self._prepare(*args, **kwargs)
        res = engine(cue, **self._ai_kwargs)
        if self._reader is not None:
            return self._reader.read(res)
        return res

    async def aforward(self, *args, **kwargs) -> typing.Any:

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
        return v

    async def astream(self, *args, **kwargs) -> typing.Any:

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
        return v

    def i(self, *args, **kwargs) -> Cue:
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
        if not self._is_method:
            return None, args
        
        if self._instance is None:
            return args[0], args[1:]
        return self._instance, args

    def get_engine(self, instance):
        
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
        return v

    async def astream(self, *args, **kwargs) -> typing.Any:

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
        return v

    def i(self, *args, **kwargs) -> Cue:
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


# 1) choose whether to use async, stream etc. __call__ = self._astream etc
# 2) simplify the code
# 3) figure out how to use the reader -> easiest if the function returns text
# 4) allow the user to pass in a module or just a function..


# class FBase(object):

#     def __init__(self, f, is_method: bool):
#         self.f = f
#         self._is_method = is_method
#         self._is_generator = is_generator_function(f)
#         self._is_async = is_async_function(f)
#         self.name = f.__name__
#         self.signature = str(inspect.signature(f))
#         self.parameters = inspect.signature(f).parameters
#         self.return_annotation = inspect.signature(f).return_annotation


# class IFuncBase(Instruct):
#     """SignatureFunc is a method where you define the cue in
#     the function signature
#     """
#     def __init__(
#         self, f: typing.Callable, engine: typing.Union[LLM, str, typing.Callable[[], LLM]], 
#         reader: typing.Optional[Reader]=None,
#         ai_kwargs: typing.Dict=None,
#         is_method: bool=False,
#         instance=None,
#     ):
#         """Wrap the signature method with a particular engine and
#         dialog factory

#         Args:
#             f (typing.Callable): The function to wrap
#             engine (AIModel): The engine to use for getting the response
#             dialog_factory (typing.Optional[typing.Callable[[], Dialog]], optional): The dialog to use. Defaults to None.
#             train (bool, optional): Whether to train the cues or not. Defaults to False.
#             instance (optional): The instance. Defaults to None.
#         """
#         self.engine = engine
        
#         self._out_cls = (
#             self.return_annotation if self.return_annotation is not None 
#             else typing.Any 
#         )

#         if reader is None:
#             if self._out_cls in primitives:
#                 reader = PrimRead(name=self.name, out_cls=self._out_cls)
#             elif issubclass(self._out_cls, pydantic.BaseModel):
#                 reader = PydanticRead(name=self.name, out_cls=self._out_cls)
#             else:
#                 reader = NullRead(name=self.name)
        
#         self._reader = reader

#         update_wrapper(self, f) 
#         self._ai_kwargs = ai_kwargs

#         if is_async_function(f) and is_generator_function(f):
#             self.__call__ = self.astream
#         elif is_generator_function(f):
#             self.__call__ = self.stream
#         elif is_async_function(f):
#             self.__call__ = self.aforward
#         else:
#             self.__call__ = self.forward

#     def get_instance(self, args):

#         if self._is_method and self._instance is None:
#             return args[0]
#         if self._instance is not None:
#             return self._instance
#         raise RuntimeError(f'Function {self.f} is labeled as a method but there is no instance.')
    
#     def get_reader(self, instance):

#         if isinstance(self._reader, str) and instance is not None:
#             if instance is None:
#                 raise RuntimeError('Reader must be defined if not an object')
#             return get_member(instance, self._reader)
#         return self._reader

#     def get_engine(self, instance, engine):

#         if engine is not None:
#             pass
#         elif hasattr(instance, 'engine'):
#             engine = object.__getattribute__(self, 'engine')

#         # TODO: How to make the engine a factory
#         # if not isinstance(engine, LLM) and isinstance(engine, typing.Callable[[], LLM]):
#         #     engine = engine()
    
#         raise RuntimeError(f'Function {self.f} has no engine available.')




# 1) create f
# 2) ... The reader must work for streams.. If 
#        not it will not stream

# instance, args = self.instance()
# vars = self.f(*args, **kwargs)
# cue = self.f.doc_cue(vars, self._reader)
# engine = self.engine(instance, engine)
# return self._reader.read(self.engine(engine)(cue, **kwargs))
# 

# instance, args = self.get_instance(args)
# cue = self.f(*args, **kwargs)
# engine = self.get_engine(instance, engine)
# # engine must return it in a format that can be read
# return self._reader.stream(engine(cue, **kwarg_overrides))


# 1) read fails
# 2) read is successful

# for stream

# stream=True, async=True
# .. For the first version.. 


# self.f


# class InstructFunc(Module, Instruct):
#     """InstructMethod is a method where you define the cue by
#     doing operations on that instructions
#     """
#     def __init__(
#         self, f: typing.Callable, engine: typing.Union[LLM, str, typing.Callable[[], LLM]], 
#         dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
#         ai_kwargs=None,
#         is_method: bool=False,
#         instance=None,
#         is_generator: bool=False,
#         is_async: bool=False
#     ):
#         """Create an InstructMethod that decorates a function that returns 
#         a cue

#         Args:
#             f (typing.Callable): The function to decorate
#             train (bool, optional): Whether to train . Defaults to True.
#             instance (, optional): The instance to use. Defaults to None.

#         """
#         # need to set __call__ for this
#         # set every method to be "private"
#         # self.__call__ = self._aforward # for instance
#         # then I need to 

#         self.f = f
#         self.engine = engine
#         update_wrapper(self, f) 
#         self._instance = instance
#         self._stored = None
#         self._is_method = is_method
#         self.dialog_factory = dialog_factory or Dialog
#         self.return_annotation = inspect.signature(f).return_annotation

#         # make it so it can automatically set this up
#         # rather than using the "Null version"
#         # if reader is None:
#         #     if self.out_cls in primitives:
#         #         reader = PrimRead(name=self.name, out_cls=self.out_cls)
#         #     elif issubclass(self.out_cls, Struct):
#         #         reader = StructRead(name=self.name, out_cls=self.out_cls)
#         #     else:
#         #         reader = NullRead(name=self.name)
        
#         # self.reader = reader or NullRead()
#         self.ai_kwargs = ai_kwargs or {}

#     def i(self, *args, **kwargs) -> Cue:
#         """Get the cue based on the arguments

#         Returns:
#             Cue: Get the cue
#         """

#         if self._instance is not None:
#             instance = self._instance
#         elif self._is_method:
#             instance = args[0]
#             args = args[1:]
#         else:
#             instance = None

#         if instance is not None:
#             result = self.f(instance, *args, **kwargs)
#         else:
#             result = self.f(*args, **kwargs)
    
#         if isinstance(result, InstructCall):
#             result = result()
#         return result

#     def forward(self, *args, _engine: LLM=None, **kwargs) -> typing.Any:        
#         """Execute the instruct method and then process the output

#         Args:
#             _engine (PromptModel, optional): Engine to override with. Defaults to None.

#         Returns:
#             typing.Any: The resulting
#         """
#         if self._instance is not None:
#             instance = self._instance
#         elif self._is_method:
#             instance = args[0]

#         engine = _engine or self.engine
#         if isinstance(engine, str) and instance is not None:
#             engine = get_member(instance, engine)
#         elif not isinstance(engine, LLM) and isinstance(engine, typing.Callable):
#             engine = engine()

#         cue = self.i(*args, **kwargs)
#         result = engine(TextMessage('system', cue), **self.ai_kwargs)
#         if self.return_annotation is typing.Any:
#             return result
#         return result.val

#     def stream(
#         self, *args,
#         _engine: LLM=None, **kwargs
#     ) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:
#         """Execute the cue and get the output 

#         Args:
#             _engine (AIModel, optional): The engine to override with. Defaults to None.

#         Returns:
#             typing.Any: The result of processing the cue
#         """
#         engine = _engine or self.engine

#         if self._instance is not None:
#             instance = self._instance
#         elif self._is_method:
#             instance = args[0]

#         if isinstance(engine, str) and self._instance is not None:
#             engine = get_member(instance, engine)
        
#         elif not isinstance(engine, LLM) and isinstance(engine, typing.Callable[[], LLM]):
#             engine = engine()

#         cue = self.i(*args,  **kwargs)
#         for cur, dx in engine.stream(TextMessage('system', cue), **self.ai_kwargs):

#             if self.return_annotation is typing.Any:
#                 yield cur, dx
#             else:
#                 yield cur.val, dx.val

#     async def aforward(self, *args, **kwargs) -> typing.Any:
#         """Execute forward asynchronously

#         Returns:
#             typing.Any: The result of the forward method
#         """
#         # TODO: Update this to use the Async for the Engine
#         return self.forward(*args, **kwargs)

#     def __get__(self, instance, owner):
#         """Get the SignatureMethod with the instance specified

#         Args:
#             instance (): The instance to use
#             owner (): 

#         Returns:
#             SignatureMethod
#         """
#         if self.f.__name__ not in instance.__dict__:
#             instance.__dict__[self.f.__name__] = InstructFunc(
#                 self.f, self.engine, self.dialog_factory,
#                 self.ai_kwargs, True, instance
#             )
#         return instance.__dict__[self.f.__name__]

#     def __iter__(self, *args, **kwargs):
#         """Loop over all child InstructCalls of this "Instruct"

#         Yields:
#             InstructCall
#         """
#         res = self(*args, **kwargs)
#         if isinstance(res, InstructCall):
#             for res_i in res:
#                 yield res_i


# class SignatureFunc(IFuncBase, Instruct):
#     """SignatureFunc is a method where you define the cue in
#     the function signature
#     """
#     def __init__(
            
#         self, f: typing.Callable, engine: typing.Union[LLM, str, typing.Callable[[], LLM]], 
#         doc: typing.Optional[str]=None,
#         reader: typing.Optional[Reader]=None,
#         train: bool=False, 
#         ai_kwargs: typing.Dict=None,
#         is_method: bool=False,
#         instance=None,
#     ):
#         """Wrap the signature method with a particular engine and
#         dialog factory

#         Args:
#             f (typing.Callable): The function to wrap
#             engine (AIModel): The engine to use for getting the response 
#             train (bool, optional): Whether to train the cues or not. Defaults to False.
#             instance (optional): The instance. Defaults to None.
#         """
#         super().__init__(
#             f, engine, reader, ai_kwargs, is_method, instance
#         )
#         self._train = train
#         self._doc = doc
#         docstring = inspect.getdoc(f) if doc is None else doc
#         self._signature = str(inspect.signature(f))
#         self._parameters = inspect.signature(f).parameters
#         self._return_annotation = inspect.signature(f).return_annotation

#         if not isinstance(docstring, typing.Callable):
#             docstring = Cue(text=docstring)
#             self._docstring = Param(
#                 name=self.name,
#                 cue=docstring,
#                 training=train
#             )
#         elif train:
#             raise ValueError('Cannot set to train if the docstring is a callable')
#         else:
#             self._docstring = docstring

#         update_wrapper(self, f) 

#     def spawn(
#         self, 
#         engine: LLM=None, 
#         train: bool=False
#     ) -> 'SignatureFunc':
#         """Spawn a new SignatureMethod. Especially use to create a trainable one

#         Args:
#             engine (AIModel, optional): Spawn a new . Defaults to None.
#             train (bool, optional): _description_. Defaults to False.

#         Returns:
#             SignatureMethod: 
#         """
#         return SignatureFunc(
#             f=self.f,
#             engine=engine or self._engine,
#             reader=self._reader,
#             doc=self._doc,
#             train=train,
#             instance=self._instance,
#             ai_kwargs=self._ai_kwargs
#         )
    
#     def get_values(self, base_values, filled_docstring, reader):

#         values = values if base_values is not None else {}
#         values = {k: v() if isinstance(k, InstructCall) else v for k, v in values.items()}
    
#         if '{TEMPLATE}' in filled_docstring:
#             values['TEMPLATE'] = reader.template()
#         return values
    
#     def fill_docstring(self):

#         if isinstance(self._docstring, Param):
#             return self._docstring.render()
#         return self._docstring()
    
#     def i(self, *args, **kwargs) -> Cue:
#         """Get the cue

#         Returns:
#             Cue: Get the cue
#         """
#         instance = self.instance(args)



#         instance, args, param_values = self.get_instance(args)
#         values = self.get_values(self.f(*args, **kwargs))
#         docstring = self.docstring.fill(args, kwargs, template=template)
#         self.fill_params(rea)

#         filled = set()

#         # param_values = list(self._parameters.values())

#         # if instance is not None:
#         #     values = self.f(instance, *args, **kwargs)
#         #     param_values = param_values[1:]
#         # else:
#         #     values = self.f(*args, **kwargs)
#         # values = values if values is not None else {}
#         # values = {k: v() if isinstance(k, InstructCall) else v for k, v in values.items()}

#         # what if one of the parameters is an cue?
#         # values.update(dict(zip(args, [v for v in param_values])))
        
#         if '{TEMPLATE}' in filled_docstring:
#             values['TEMPLATE'] = reader.template()

#         # for value, param in zip(args, param_values):
#         #     values[param.name] = value
#         #     filled.add(param.name)
        
#         # for k, value in kwargs.items():
#         #     param = self._parameters[k]
#         #     values[param.name] = value
#         #     filled.add(param.name)

#         # for param in param_values:
#         #     if param.name in filled:
#         #         continue
#         #     if param.default == inspect.Parameter.empty:
#         #         raise RuntimeError('Param has not been defined and no value')
            
#         #     values[param.name] = param.default

#         # TODO: Determine how to handle this
#         out = validate_out(values)
#         values = {key: render(v) for key, v in values.items()}
        
#         filled_docstring = str_formatter(
#             filled_docstring, required=False, **values
#         )

#         return Cue(
#             text=filled_docstring,
#             out=reader, 
#             # out=StructFormatter(name=self.name, out_cls=self.out_cls)
#         )

#     def forward(
#         self, *args,
#         _engine: LLM=None, **kwargs
#     ) -> typing.Any:
#         """Execute the cue and get the output 

#         Args:
#             _engine (AIModel, optional): The engine to override with. Defaults to None.

#         Returns:
#             typing.Any: The result of processing the cue
#         """
#         instance = self.get_instance(args)
#         engine = self.get_engine(instance, _engine)

#         if isinstance(engine, str) and instance is not None:
#             engine = get_member(instance, engine)
#         elif not isinstance(engine, LLM) and isinstance(engine, typing.Callable):
#             engine = engine()

#         cue = self.i(*args,  **kwargs)

#         result = engine(TextMessage('system', cue), **self._ai_kwargs)
#         if self.out_cls is typing.Any:
#             return result
#         return result.val

#     def stream(
#         self, *args,
#         _engine: LLM=None, **kwargs
#     ) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:
#         """Execute the cue and get the output 

#         Args:
#             _engine (AIModel, optional): The engine to override with. Defaults to None.

#         Returns:
#             typing.Any: The result of processing the cue
#         """

#         instance = self.get_instance(args)
#         engine = self.get_engine(instance, _engine)

#         cue = self.i(*args,  **kwargs)
#         for cur, dx in engine.stream(TextMessage('system', cue), **self._ai_kwargs):

#             if self.out_cls is typing.Any:
#                 yield cur, dx
#             else:
#                 yield cur.val, dx.val

#     async def aforward(
#         self, *args, 
#         _engine: LLM=None, 
#         **kwargs
#     ) -> typing.Any:
#         """Execute the cue and get the output

#         Args:
#             _engine (AIModel, optional): The engine to override with. Defaults to None.

#         Returns:
#             typing.Any: The result of processing the cue
#         """
#         return self.forward(
#             *args, 
#             _engine=_engine,
#             **kwargs
#         )

#     def __iter__(self, *args, **kwargs) -> typing.Iterator[InstructCall]:
#         """Loop over all child InstructCalls of this "Instruct"

#         Yields:
#             InstructCall
#         """
#         res = self(*args, **kwargs)
#         for k, v in res.items():
#             if isinstance(v, InstructCall):
#                 for v_i in v:
#                     yield v_i
#         yield InstructCall(self, *args, **kwargs)

#     def __get__(self, instance, owner):
#         """Set the instance on the SignatureMethod

#         Args:
#             instance (): The instance to use
#             owner (): 

#         Returns:
#             SignatureMethod
#         """
#         if self.f.__name__ not in instance.__dict__:
#             instance.__dict__[self.f.__name__] = SignatureFunc(
#                 self.f, self.engine, self.dialog_factory,
#                 self._doc, self._reader, self._train,
#                 self._ai_kwargs, True, instance
#             )
#         return instance.__dict__[self.f.__name__]

