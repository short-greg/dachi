# 1st party
import typing
from functools import wraps, update_wrapper
import inspect
from itertools import chain

import pydantic

# local
from ._core import (
    render, 
    Instruction, render, Param, 
    Instruct, Reader, NullRead,
)
from .._core._ai import (
    Dialog, TextMessage, AIModel, AIResponse
)
from ._read import (
    PydanticRead, PrimRead
)
from ..utils._utils import (
    str_formatter, primitives
)
from .._core._process import Module


X = typing.Union[str, Instruction]


def validate_out(instructions: typing.List[X]) -> typing.Optional[Reader]:
    """Validate an Out based on several instructions

    Args:
        instructions (typing.List[X]): The instructions 

    Returns:
        Out: The resulting "Out" to use from the instructions
    """
    if isinstance(instructions, dict):
        instructions = instructions.values()

    out = None
    for instruction in instructions:
        if not isinstance(instruction, Instruction):
            continue
        if out is None and instruction.out is not None:
            out = instruction.out
        elif instruction.out is not None:
            raise RuntimeError(f'Out cannot be duplicated')
    return out


class InstructCall(Module, Instruct):
    """InstructCall is used within an instructf or
    signaturef to allow one to loop over "incoming"
    instructions
    """

    def __init__(self, i: Instruct, *args, **kwargs):
        """Create the "Call" passing in the instruction
        and its inputs

        Args:
            i (Instruct): The instruction to wrap
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

    def i(self) -> Instruction:
        """Get the instruction 

        Returns:
            Instruction: The instruction
        """
        return self._instruct.i()
        
    def forward(self) -> typing.Any:
        """Execute the instruction

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


class SignatureFunc(Module, Instruct):
    """SignatureFunc is a method where you define the instruction in
    the function signature
    """
    def __init__(
        self, f: typing.Callable, engine: typing.Union[AIModel, str, typing.Callable[[], AIModel]], 
        dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
        is_method: bool=False,
        doc: typing.Optional[str]=None,
        reader: typing.Optional[Reader]=None,
        train: bool=False, 
        ai_kwargs: typing.Dict=None
    ):
        """Wrap the signature method with a particular engine and
        dialog factory

        Args:
            f (typing.Callable): The function to wrap
            engine (AIModel): The engine to use for getting the response
            dialog_factory (typing.Optional[typing.Callable[[], Dialog]], optional): The dialog to use. Defaults to None.
            train (bool, optional): Whether to train the instructions or not. Defaults to False.
            is_method (bool, optional): Whether it is a method or not. Defaults to False.
            instance (optional): The instance. Defaults to None.
        """
        self.f = f
        self.name = f.__name__
        self._is_method = is_method
        self.engine = engine
        self._train = train
        self._doc = doc
        docstring = inspect.getdoc(f) if doc is None else doc
        self.signature = str(inspect.signature(f))
        self.parameters = inspect.signature(f).parameters
        self.return_annotation = inspect.signature(f).return_annotation

        if not isinstance(docstring, typing.Callable):
            docstring = Instruction(text=docstring)
            self._docstring = Param(
                name=self.name,
                instruction=docstring,
                training=train
            )
        elif train:
            raise ValueError('Cannot set to train if the docstring is a callable')
        else:
            self._docstring = docstring

        self.out_cls = (
            self.return_annotation if self.return_annotation is not None 
            else AIResponse 
        )

        if reader is None:
            if self.out_cls in primitives:
                reader = PrimRead(name=self.name, out_cls=self.out_cls)
            elif issubclass(self.out_cls, pydantic.BaseModel):
                reader = PydanticRead(name=self.name, out_cls=self.out_cls)
            else:
                reader = NullRead(name=self.name)
        
        self.reader = reader

        update_wrapper(self, f) 
        self.instance = None
        # self._stored = None
        self.dialog_factory = dialog_factory or Dialog
        self.ai_kwargs = ai_kwargs

    def spawn(
        self, 
        engine: AIModel=None, 
        dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
        train: bool=False
    ) -> 'SignatureFunc':
        """Spawn a new SignatureMethod. Especially use to create a trainable one

        Args:
            engine (AIModel, optional): Spawn a new . Defaults to None.
            dialog_factory (typing.Optional[typing.Callable[[], Dialog]], optional): _description_. Defaults to None.
            train (bool, optional): _description_. Defaults to False.

        Returns:
            SignatureMethod: 
        """
        return SignatureFunc(
            f=self.f,
            engine=engine or self.engine,
            is_method=self._is_method,
            dialog_factory=dialog_factory or self.dialog_factory,
            reader=self.reader,
            doc=self._doc,
            train=train,
            ai_kwargs=self.ai_kwargs
        )

    def i(self, *args, **kwargs) -> Instruction:
        """Get the instruction

        Returns:
            Instruction: Get the instruction
        """
        if isinstance(self._docstring, Param):
            filled_docstring = self._docstring.render()
        else:
            filled_docstring = self._docstring()

        filled = set()

        param_values = list(self.parameters.values())

        if isinstance(self.reader, str) and self._is_method:
            reader = getattr(self.instance, self.reader)
        else:
            reader = self.reader

        if self._is_method:
            values = self.f(self.instance, *args, **kwargs)
        else:
            values = self.f(*args, **kwargs)
        values = values if values is not None else {}
        values = {k: v() if isinstance(k, InstructCall) else v for k, v in values.items()}

        if self._is_method:            
            param_values = param_values[1:]

        # what if one of the parameters is an instruction?
        # values.update(dict(zip(args, [v for v in param_values])))
        
        if '{TEMPLATE}' in filled_docstring:
            values['TEMPLATE'] = reader.template()

        for value, param in zip(args, param_values):
            values[param.name] = value
            filled.add(param.name)
        
        for k, value in kwargs.items():
            param = self.parameters[k]
            values[param.name] = value
            filled.add(param.name)

        for param in param_values:
            if param.name in filled:
                continue
            if param.default == inspect.Parameter.empty:
                raise RuntimeError('Param has not been defined and no value')
            
            values[param.name] = param.default

        # TODO: Determine how to handle this
        out = validate_out(values)
        values = {key: render(v) for key, v in values.items()}
            
        filled_docstring = str_formatter(
            filled_docstring, required=False, **values
        )

        return Instruction(
            text=filled_docstring,
            out=reader, 
            # out=StructFormatter(name=self.name, out_cls=self.out_cls)
        )

    def forward(
        self, *args,
        _engine: AIModel=None, **kwargs
    ) -> typing.Any:
        """Execute the instruction and get the output 

        Args:
            _engine (AIModel, optional): The engine to override with. Defaults to None.

        Returns:
            typing.Any: The result of processing the instruction
        """
        engine = _engine or self.engine

        if isinstance(engine, str) and self._is_method:
            engine = getattr(self.instance, engine)
        elif not isinstance(engine, AIModel) and isinstance(engine, typing.Callable[[], AIModel]):
            engine = engine()

        instruction = self.i(*args,  **kwargs)
        result = engine(TextMessage('system', instruction), **self.ai_kwargs)
        if self.out_cls is AIResponse:
            return result
        return result.val

    def stream_forward(
        self, *args,
        _engine: AIModel=None, **kwargs
    ) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:
        """Execute the instruction and get the output 

        Args:
            _engine (AIModel, optional): The engine to override with. Defaults to None.

        Returns:
            typing.Any: The result of processing the instruction
        """
        engine = _engine or self.engine

        if isinstance(engine, str) and self._is_method:
            engine = getattr(self.instance, engine)
        elif not isinstance(engine, AIModel) and isinstance(engine, typing.Callable[[], AIModel]):
            engine = engine()

        instruction = self.i(*args,  **kwargs)
        for cur, dx in engine.stream_forward(TextMessage('system', instruction), **self.ai_kwargs):

            if self.out_cls is AIResponse:
                yield cur, dx
            else:
                yield cur.val, dx.val

    async def async_forward(
        self, *args, 
        _engine: AIModel=None, 
        **kwargs
    ) -> typing.Any:
        """Execute the instruction and get the output

        Args:
            _engine (AIModel, optional): The engine to override with. Defaults to None.

        Returns:
            typing.Any: The result of processing the instruction
        """
        return self.forward(
            *args, 
            _engine=_engine,
            **kwargs
        )

    def __iter__(self, *args, **kwargs) -> typing.Iterator[InstructCall]:
        """Loop over all child InstructCalls of this "Instruct"

        Yields:
            InstructCall
        """
        res = self(*args, **kwargs)
        for k, v in res.items():
            if isinstance(v, InstructCall):
                for v_i in v:
                    yield v_i
        yield InstructCall(self, *args, **kwargs)

    def __get__(self, instance, owner):
        """Set the instance on the SignatureMethod

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        self.instance = instance
        return self


class InstructFunc(Module, Instruct):
    """InstructMethod is a method where you define the instruction by
    doing operations on that instructions
    """

    def __init__(
        self, f: typing.Callable, engine: typing.Union[AIModel, str, typing.Callable[[], AIModel]], 
        dialog_factory: typing.Optional[typing.Callable[[], Dialog]]=None,
        is_method: bool=False,
        ai_kwargs=None
        #reader: typing.Optional[Reader]=None,
    ):
        """Create an InstructMethod that decorates a function that returns 
        an instruction

        Args:
            f (typing.Callable): The function to decorate
            train (bool, optional): Whether to train . Defaults to True.
            instance (, optional): The instance to use. Defaults to None.

        """
        self.f = f
        self._is_method = is_method
        self.engine = engine
        update_wrapper(self, f) 
        self.instance = None
        self._stored = None
        self.dialog_factory = dialog_factory or Dialog
        self.return_annotation = inspect.signature(f).return_annotation

        # make it so it can automatically set this up
        # rather than using the "Null version"
        # if reader is None:
        #     if self.out_cls in primitives:
        #         reader = PrimRead(name=self.name, out_cls=self.out_cls)
        #     elif issubclass(self.out_cls, Struct):
        #         reader = StructRead(name=self.name, out_cls=self.out_cls)
        #     else:
        #         reader = NullRead(name=self.name)
        
        # self.reader = reader or NullRead()
        self.ai_kwargs = ai_kwargs or {}

    def i(self, *args, **kwargs) -> Instruction:
        """Get the instruction based on the arguments

        Returns:
            Instruction: Get the instruction
        """

        if self._is_method:
            result = self.f(self.instance, *args, **kwargs)
        else:
            result = self.f(*args, **kwargs)
    
        if isinstance(result, InstructCall):
            result = result()
        return result

    def forward(self, *args, _engine: AIModel=None, **kwargs) -> typing.Any:        
        """Execute the instruct method and then process the output

        Args:
            _engine (AIModel, optional): Engine to override with. Defaults to None.

        Returns:
            typing.Any: The resulting
        """
        engine = _engine or self.engine
        if isinstance(engine, str) and self._is_method:
            engine = getattr(self.instance, engine)
        elif not isinstance(engine, AIModel) and isinstance(engine, typing.Callable[[], AIModel]):
            engine = engine()
        instruction = self.i(*args, **kwargs)
        result = engine(TextMessage('system', instruction), **self.ai_kwargs)
        if self.return_annotation is AIResponse:
            return result
        return result.val

    def stream_forward(
        self, *args,
        _engine: AIModel=None, **kwargs
    ) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:
        """Execute the instruction and get the output 

        Args:
            _engine (AIModel, optional): The engine to override with. Defaults to None.

        Returns:
            typing.Any: The result of processing the instruction
        """
        engine = _engine or self.engine

        if isinstance(engine, str) and self._is_method:
            engine = getattr(self.instance, engine)
        elif not isinstance(engine, AIModel) and isinstance(engine, typing.Callable[[], AIModel]):
            engine = engine()

        instruction = self.i(*args,  **kwargs)
        for cur, dx in engine.stream_forward(TextMessage('system', instruction), **self.ai_kwargs):

            if self.return_annotation is AIResponse:
                yield cur, dx
            else:
                yield cur.val, dx.val

    async def async_forward(self, *args, **kwargs) -> typing.Any:
        """Execute forward asynchronously

        Returns:
            typing.Any: The result of the forward method
        """
        # TODO: Update this to use the Async for the Engine
        return self.forward(*args, **kwargs)

    def __get__(self, instance, owner):
        """Get the SignatureMethod with the instance specified

        Args:
            instance (): The instance to use
            owner (): 

        Returns:
            SignatureMethod
        """
        self.instance = instance
        return self

    def __iter__(self, *args, **kwargs):
        """Loop over all child InstructCalls of this "Instruct"

        Yields:
            InstructCall
        """
        res = self(*args, **kwargs)
        if isinstance(res, InstructCall):
            for res_i in res:
                yield res_i


def instructfunc(
    engine: AIModel=None, 
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # TODO: Use wrapper
        return InstructFunc(f, engine, None, False, ai_kwargs=ai_kwargs)
    return _


def instructmethod(
    engine: AIModel=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        # TODO: Use wrapper
        return InstructFunc(
            f, engine, None, True, 
            ai_kwargs=ai_kwargs
        )
    return _


def signaturemethod(
    engine: AIModel=None, 
    reader: Reader=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        return SignatureFunc(
            f, engine, None, True, doc=doc,
            reader=reader, ai_kwargs=ai_kwargs
        )

    return _


def signaturefunc(
    engine: AIModel=None, 
    reader: Reader=None,
    doc: typing.Union[str, typing.Callable[[], str]]=None,
    **ai_kwargs
):
    """Decorator for using a function signature

    Args:
        train (bool, optional): Whether to train the function or not. Defaults to True.
    """
    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return SignatureFunc(
            f, engine, None, False, doc=doc, reader=reader, 
            ai_kwargs=ai_kwargs
        )

    return _
