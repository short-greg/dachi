import typing as t

import pydantic

from dachi.config import config
from dachi.utils.text import str_formatter
from ._lang import Engines, LangEngine, LangModel


def difference(
    a: str, 
    b: str, 
    _prompt: str | None = None, 
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None, 
) -> str:
    """Get the difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        The difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.Difference.prompt
    model = _model if _model is not None else config.Ops.Difference.model
    response, _, _ = Engines.get(model).forward(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    )
    return response


def difference_stream(a: str, b: str, _prompt: str | None = None, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> t.Iterator[str]:
    """Get the difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        An iterator over the difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.Difference.prompt
    model = _model if _model is not None else config.Ops.Difference.model
    for chunk, _, _ in Engines.get(model).stream(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    ):
        yield chunk


async def async_difference(a: str, b: str, _prompt: str | None = None, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> str:
    """Get the difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        The difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.Difference.prompt
    model = _model if _model is not None else config.Ops.Difference.model
    response, _, _ = await Engines.get(model).aforward(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    )
    return response


async def difference_astream(
    a: str, 
    b: str, 
    _prompt: str | None = None, 
    _model: LangModel | None | str = None, 
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.AsyncIterator[str]:
    """Get the difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        An async iterator over the difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.Difference.prompt
    model = _model if _model is not None else config.Ops.Difference.model
    async for chunk, _, _ in Engines.get(model).astream(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    ):
        yield chunk


def _add_header(
    texts: list[str], 
    header: str, 
    sep: str
) -> str:
    with_header = [header.format(i) + t for i, t in enumerate(texts, 1)]
    combined_text = sep.join(with_header)
    prompt = f"Combine the following texts into one unified text, separating sections with '{sep}':\n\n{combined_text}"
    return prompt


def union(
    *text: str, 
    _sep: str | None = None, 
    _header: str | None = None, 
    _prompt: str | None = None, 
    _model: LangModel | None | str = None, 
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Combine multiple texts into one unified text using a language model.

    Args:
        *text: The texts to combine.
        _separator: The separator to use between texts.
        _prompt: The prompt to use for the language model.

    Returns:
        The combined text.
    """
    sep = _sep if _sep is not None else config.Ops.Union.sep
    header = _header if _header is not None else config.Ops.Union.header
    prompt = _prompt if _prompt is not None else config.Ops.Union.prompt
    model = _model if _model is not None else config.Ops.Union.model
    data = _add_header(list(text), header, sep)
    response, _, _ = Engines.get(model).forward(
        prompt=str_formatter(prompt, texts=data, sep=sep), structure=_structure
    )
    return response


def union_stream(*text: str, _sep: str | None = None, _header: str | None = None, _prompt: str | None = None, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> t.Iterator[str]:
    """Combine multiple texts into one unified text using a language model.

    Args:
        *text: The texts to combine.
        _separator: The separator to use between texts.
        _prompt: The prompt to use for the language model.

    Returns:
        An iterator over the combined text.
    """
    sep = _sep if _sep is not None else config.Ops.Union.sep
    header = _header if _header is not None else config.Ops.Union.header
    prompt = _prompt if _prompt is not None else config.Ops.Union.prompt
    model = _model if _model is not None else config.Ops.Union.model
    data = _add_header(list(text), header, sep)
    for chunk, _, _ in Engines.get(model).stream(
        prompt=str_formatter(prompt, texts=data, sep=sep),
        structure=_structure
    ):
        yield chunk


async def async_union(*text: str, _sep: str | None = None, _header: str | None = None, _prompt: str | None = None, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> str:
    """Combine multiple texts into one unified text using a language model.

    Args:
        *text: The texts to combine.
        _separator: The separator to use between texts.
        _prompt: The prompt to use for the language model.

    Returns:
        The combined text.
    """
    sep = _sep if _sep is not None else config.Ops.Union.sep
    header = _header if _header is not None else config.Ops.Union.header
    prompt = _prompt if _prompt is not None else config.Ops.Union.prompt
    model = _model if _model is not None else config.Ops.Union.model
    data = _add_header(list(text), header, sep)
    response, _, _ = await Engines.get(model).aforward(
        prompt=str_formatter(prompt, texts=data, sep=sep), structure=_structure
    )
    return response


async def union_astream(
    *text: str, 
    _sep: str | None = None, 
    _header: str | None = None, 
    _prompt: str | None = None, 
    _model: LangModel | None | str = None, 
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.AsyncIterator[str]:
    """Combine multiple texts into one unified text using a language model.

    Args:
        *text: The texts to combine.
        _separator: The separator to use between texts.
        _prompt: The prompt to use for the language model.

    Returns:
        An async iterator over the combined text.
    """
    sep = _sep if _sep is not None else config.Ops.Union.sep
    header = _header if _header is not None else config.Ops.Union.header
    prompt = _prompt if _prompt is not None else config.Ops.Union.prompt
    model = _model if _model is not None else config.Ops.Union.model
    data = _add_header(list(text), header, sep)
    async for chunk, _, _ in Engines.get(model).astream(
        prompt=str_formatter(prompt, texts=data, sep=sep),
        structure=_structure
    ):
        yield chunk




def intersect(
    *texts: str,
    _sep: str | None = None,
    _header: str | None = None,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        The intersection of the texts.
    """
    sep = _sep if _sep is not None else config.Ops.Intersection.sep
    header = _header if _header is not None else config.Ops.Intersection.header
    prompt = _prompt if _prompt is not None else config.Ops.Intersection.prompt
    model = _model if _model is not None else config.Ops.Intersection.model
    data = _add_header(list(texts), header, sep)
    response, _, _ = Engines.get(model).forward(
        prompt=str_formatter(prompt, texts=data), structure=_structure
    )
    return response


def intersect_stream(
    *texts: str,
    _sep: str | None = None,
    _header: str | None = None,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.Iterator[str]:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        An iterator over the intersection of the texts.
    """
    sep = _sep if _sep is not None else config.Ops.Intersection.sep
    header = _header if _header is not None else config.Ops.Intersection.header
    prompt = _prompt if _prompt is not None else config.Ops.Intersection.prompt
    model = _model if _model is not None else config.Ops.Intersection.model
    data = _add_header(list(texts), header, sep)
    for chunk, _, _ in Engines.get(model).stream(
        prompt=str_formatter(prompt, texts=data), structure=_structure
    ):
        yield chunk

async def async_intersect(
    *texts: str,
    _sep: str | None = None,
    _header: str | None = None,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        The intersection of the texts.
    """
    sep = _sep if _sep is not None else config.Ops.Intersection.sep
    header = _header if _header is not None else config.Ops.Intersection.header
    prompt = _prompt if _prompt is not None else config.Ops.Intersection.prompt
    model = _model if _model is not None else config.Ops.Intersection.model
    data = _add_header(list(texts), header, sep)
    response, _, _ = await Engines.get(model).aforward(
        prompt=str_formatter(prompt, texts=data), structure=_structure
    )
    return response


async def intersect_astream(
    *texts: str,
    _sep: str | None = None,
    _header: str | None = None,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.AsyncIterator[str]:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        An async iterator over the intersection of the texts.
    """
    sep = _sep if _sep is not None else config.Ops.Intersection.sep
    header = _header if _header is not None else config.Ops.Intersection.header
    prompt = _prompt if _prompt is not None else config.Ops.Intersection.prompt
    model = _model if _model is not None else config.Ops.Intersection.model
    data = _add_header(list(texts), header, sep)
    async for chunk, _, _ in Engines.get(model).astream(
        prompt=str_formatter(prompt, texts=data), structure=_structure
    ):
        yield chunk


def symmetric_difference(
    a: str,
    b: str,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Get the symmetric difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        The symmetric difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.SymmetricDifference.prompt
    model = _model if _model is not None else config.Ops.SymmetricDifference.model
    response, _, _ = Engines.get(model).forward(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    )
    return response


def symmetric_difference_stream(
    a: str,
    b: str,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.Iterator[str]:
    """Get the symmetric difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        An iterator over the symmetric difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.SymmetricDifference.prompt
    model = _model if _model is not None else config.Ops.SymmetricDifference.model
    for chunk, _, _ in Engines.get(model).stream(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    ):
        yield chunk


async def async_symmetric_difference(
    a: str,
    b: str,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Get the symmetric difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        The symmetric difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.SymmetricDifference.prompt
    model = _model if _model is not None else config.Ops.SymmetricDifference.model
    response, _, _ = await Engines.get(model).aforward(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    )
    return response


async def symmetric_difference_astream(
    a: str,
    b: str,
    _prompt: str | None = None,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.AsyncIterator[str]:
    """Get the symmetric difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        An async iterator over the symmetric difference between the two texts.
    """
    prompt = _prompt if _prompt is not None else config.Ops.SymmetricDifference.prompt
    model = _model if _model is not None else config.Ops.SymmetricDifference.model
    async for chunk, _, _ in Engines.get(model).astream(
        prompt=str_formatter(prompt, a=a, b=b), structure=_structure
    ):
        yield chunk



class Union(LangEngine):
    """Operation to combine multiple texts into one unified text using a language model.
    """
    def __init__(
        self,
        sep: str | None = None,
        header: str | None = None,
        prompt: str | None = None,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        sep_val = sep if sep is not None else config.Ops.Union.sep
        header_val = header if header is not None else config.Ops.Union.header
        prompt_val = prompt if prompt is not None else config.Ops.Union.prompt
        model_val = model if model is not None else config.Ops.Union.model
        super().__init__(
            prompt=prompt_val,
            model=model_val,
            structure=structure
        )
        self.sep = sep_val
        self.header = header_val

    def forward(self, *text: str) -> str:
        """Combine multiple texts into one unified text.

        Args:
            *text: The texts to combine.

        Returns:
            The combined text.
        """
        return union(
            *text,
            _sep=self.sep,
            _header=self.header,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def aforward(self, *text: str) -> str:
        """Asynchronously combine multiple texts into one unified text.

        Args:
            *text: The texts to combine.

        Returns:
            The combined text.
        """
        return await async_union(
            *text,
            _sep=self.sep,
            _header=self.header,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    def stream(self, *text: str) -> t.Iterator[str]:
        """Stream the combined text from multiple texts.

        Args:
            *text: The texts to combine.
        Returns:
            An iterator over the combined text.
        """
        return union_stream(
            *text,
            _sep=self.sep,
            _header=self.header,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def astream(self, *text: str) -> t.AsyncIterator[str]:
        """Asynchronously stream the combined text from multiple texts.

        Args:
            *text: The texts to combine.
        Returns:
            An async iterator over the combined text.
        """
        return await union_astream(
            *text,
            _sep=self.sep,
            _header=self.header,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    

class Intersection(LangEngine):
    """Operation to get the intersection of multiple texts using a language model.
    """
    def __init__(
        self,
        prompt: str | None = None,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        prompt_val = prompt if prompt is not None else config.Ops.Intersection.prompt
        model_val = model if model is not None else config.Ops.Intersection.model
        super().__init__(
            prompt=prompt_val,
            model=model_val,
            structure=structure
        )

    def forward(self, *texts: str) -> str:
        """Get the intersection of multiple texts.

        Args:
            *texts: The texts to intersect.

        Returns:
            The intersection of the texts.
        """
        return intersect(
            *texts,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def aforward(self, *texts: str) -> str:
        """Asynchronously get the intersection of multiple texts.

        Args:
            *texts: The texts to intersect.

        Returns:
            The intersection of the texts.
        """
        return await async_intersect(
            *texts,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    def stream(self, *texts: str) -> t.Iterator[str]:
        """Stream the intersection of multiple texts.

        Args:
            *texts: The texts to intersect.
        Returns:
            An iterator over the intersection of the texts.
        """
        return intersect_stream(
            *texts,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    def astream(self, *texts: str) -> t.AsyncIterator[str]:
        """Asynchronously stream the intersection of multiple texts.

        Args:
            *texts: The texts to intersect.
        Returns:
            An async iterator over the intersection of the texts.
        """
        return intersect_astream(
            *texts,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )


class Difference(LangEngine):
    """Operation to get the difference between two texts using a language model.
    """

    def __init__(
        self,
        prompt: str | None = None,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        prompt_val = prompt if prompt is not None else config.Ops.Difference.prompt
        model_val = model if model is not None else config.Ops.Difference.model
        super().__init__(
            prompt=prompt_val,
            model=model_val,
            structure=structure
        )

    def forward(self, a: str, b: str) -> str:
        """Get the difference between two texts.

        Args:
            a: The first text.
            b: The second text.

        Returns:
            The difference between the two texts.
        """
        return difference(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def aforward(self, a: str, b: str) -> str:
        """Asynchronously get the difference between two texts.

        Args:
            a: The first text.
            b: The second text.

        Returns:
            The difference between the two texts.
        """
        return await async_difference(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    def stream(self, a: str, b: str) -> t.Iterator[str]:
        """Stream the difference between two texts.

        Args:
            a: The first text.
            b: The second text.
        Returns:
            An iterator over the difference between the two texts.
        """
        return difference_stream(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def astream(self, a: str, b: str) -> t.AsyncIterator[str]:
        """Asynchronously stream the difference between two texts.

        Args:
            a: The first text.
            b: The second text.
        Returns:
            An async iterator over the difference between the two texts.
        """
        return await difference_astream(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    

class SymmetricDifference(LangEngine):
    """Operation to get the symmetric difference between two texts using a language model.
    """
    def __init__(
        self,
        prompt: str | None = None,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        prompt_val = prompt if prompt is not None else config.Ops.SymmetricDifference.prompt
        model_val = model if model is not None else config.Ops.SymmetricDifference.model
        super().__init__(
            prompt=prompt_val,
            model=model_val,
            structure=structure
        )

    def forward(self, a: str, b: str) -> str:
        """Get the symmetric difference between two texts.

        Args:
            a: The first text.
            b: The second text.

        Returns:
            The symmetric difference between the two texts.
        """
        return symmetric_difference(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def aforward(self, a: str, b: str) -> str:
        """Asynchronously get the symmetric difference between two texts.

        Args:
            a: The first text.
            b: The second text.

        Returns:
            The symmetric difference between the two texts.
        """
        return await async_symmetric_difference(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    def stream(self, a: str, b: str) -> t.Iterator[str]:
        """Stream the symmetric difference between two texts.

        Args:
            a: The first text.
            b: The second text.
        Returns:
            An iterator over the symmetric difference between the two texts.
        """
        return symmetric_difference_stream(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
    
    async def astream(self, a: str, b: str) -> t.AsyncIterator[str]:
        """Asynchronously stream the symmetric difference between two texts.

        Args:
            a: The first text.
            b: The second text.
        Returns:
            An async iterator over the symmetric difference between the two texts.
        """
        return await symmetric_difference_astream(
            a,
            b,
            _prompt=self.prompt,
            _model=self.model,
            _structure=self.structure
        )
