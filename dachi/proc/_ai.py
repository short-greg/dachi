import typing as t
from abc import abstractmethod, ABC
import pydantic
from ._process import (
    Process,
    AsyncProcess,
    StreamProcess,
    AsyncStreamProcess,
)
from dachi.core import Inp, Registry, Runtime


class LangModel(
    Process, 
    AsyncProcess, 
    StreamProcess, 
    AsyncStreamProcess
):
    """A simple LLM process that echoes input with a prefix.
    """

    @abstractmethod
    def forward(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Synchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the LLM's response string (str, List[Inp], Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), and the third element is the raw response from the LLM
        
        """
        pass

    @abstractmethod
    async def aforward(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        """Asynchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response.
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            A tuple containing the LLM's response string (str, List[Inp], Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), and the third element is the raw response from the LLM
        
        """
        pass

    @abstractmethod
    def stream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp], t.Any]]:
        """Streaming synchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response. 
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            An iterator yielding tuples containing the LLM's response string (str, List[Inp], Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), the message for the current call will not be added until the stream is complete, and the third element is the raw response from the LLM, could be a chunk object
        """
        pass

    @abstractmethod
    async def astream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp], t.Any]]:
        """Streaming asynchronous LLM response.
        
        Args:
            prompt: The input prompt(s) to the LLM. These will be converted to the
            appropriate format from the API being adapted if they are not already. They will also be returned like this.
            structure: Optional JSON structure to guide the LLM's response. 
            tools: Optional tools to assist the LLM. The schema for the tool must be provided here.
            **kwargs: Additional keyword arguments.
        Returns:
            An async iterator yielding tuples containing the LLM's response string (str, List[Inp
            , Any). The first element is the response text,
            the second element is a list of messages that can be passed as input to subsequent calls (must work for the API being adapted), the message for the current call will not be added until the stream is complete, and the third element is the raw response from the LLM, could be a chunk object
        """
        pass


LANG_MODEL = t.TypeVar("LANG_MODEL", bound=LangModel)

Engines = Registry[LangModel]()


class Op(
    Process, 
    AsyncProcess, 
    StreamProcess, 
    AsyncStreamProcess
):
    """An operation that uses a language model to process input and generate output.
    """
    def model_post_init(self, __context):
        """Post init to set up model property.
        """
        super().model_post_init(__context)
        self._model = Runtime[LangModel](
            data=None
        )

    @property
    def model(self) -> LangModel | str | None:
        """The language model used by this operation.
        """
        return self._model.data
    
    @model.setter
    def model(
        self, 
        value: LangModel | str | None
    ) -> LangModel | str | None:
        self._model.data = value
        return self._model.data

    def get_model(self, override: t.Optional[LangModel | str]=None) -> LangModel:
        """Get the language model, raising an error if it is not set.
        """
        if override is None:
            override = self._model.data
        if override is None:
            raise ValueError("Model is not set.")
        model = Engines.get(override)
        if model is None:
            raise ValueError(f"Model '{override}' not found in registry.")
        return override
    
    def forward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return model.forward(prompt, structure=structure, tools=tools, **kwargs)
    
    async def aforward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return await model.aforward(prompt, structure=structure, tools=tools, **kwargs)
    
    def stream(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Iterator[t.Tuple[str, t.List[Inp], t.Any]]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return model.stream(prompt, structure=structure, tools=tools, **kwargs)
    
    async def astream(
        self, 
        prompt: list[Inp] | Inp,
        structure: t.Dict | None | pydantic.BaseModel = None,
        tools: t.Dict | None | pydantic.BaseModel = None,
        _model: LangModel | None = None,
        **kwargs
    ) -> t.AsyncIterator[t.Tuple[str, t.List[Inp], t.Any]]:
        model = self.get_model(_model)
        if model is None:
            raise ValueError("Model is not set so must pass in to use.")
        return await model.astream(prompt, structure=structure, tools=tools, **kwargs)


class ToolUser(Op):
    """An operation that uses tools with a language model to process input and generate output.
    """
    
    def forward(
        self, 
        prompt: list[Inp] | Inp, 
        structure: t.Dict | None | pydantic.BaseModel = None, 
        tools: t.Dict | None | pydantic.BaseModel = None, 
        _model: LangModel | None = None, **kwargs
    ) -> t.Tuple[str, t.List[Inp], t.Any]:
        if tools is None:
            raise ValueError("Tools must be provided for ToolUser operations.")
        res = super().forward(prompt, structure=structure, tools=tools, _model=_model, **kwargs)
        # check if tools were used in the response, raise error if not
        


OP = t.TypeVar("OP", bound=Op)


