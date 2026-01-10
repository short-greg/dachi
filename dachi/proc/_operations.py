from ._ai import LangModel, Engines, LangEngine
from dachi.utils.text import str_formatter
import typing as t
import pydantic


DIFFERENCE_TEMPLATE = """Describe everything that is in Text A but is not in Text B.
Text A:
{a}
Text B:
{b}"""


SYMMETRIC_DIFFERENCE_TEMPLATE = """Describe everything that is in Text A but is not in Text B, and everything that is in Text B but is not in Text A.
Text A:
{a}
Text B:
{b}"""


UNION_TEMPLATE = """Combine the following texts into one unified text, separating sections with '{sep}': 
{texts}"""


INTERSECTION_TEMPLATE = """Given the following texts, extract and return only the content that is common to all texts.
Texts:
{texts}"""


def difference(
    a: str, 
    b: str, 
    _prompt: str=DIFFERENCE_TEMPLATE, 
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
    response, _, _ =Engines.get(_model).forward(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    )
    return response


def difference_stream(a: str, b: str, _prompt: str=DIFFERENCE_TEMPLATE, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> t.Iterator[str]:
    """Get the difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        An iterator over the difference between the two texts.
    """
    for chunk, _, _ in Engines.get(_model).stream(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    ):
        yield chunk


async def async_difference(a: str, b: str, _prompt: str=DIFFERENCE_TEMPLATE, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> str:
    """Get the difference between two texts using a language model.

    Args:
        a: The first text.
        b: The second text.
        _prompt: The prompt to use for the language model.

    Returns:
        The difference between the two texts.
    """
    response, _, _ = await Engines.get(_model).aforward(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    )
    return response


async def difference_astream(
    a: str, 
    b: str, 
    _prompt: str=DIFFERENCE_TEMPLATE, 
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
    async for chunk, _, _ in Engines.get(_model).astream(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
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
    _sep: str='\n', 
    _header: str="**TEXT {}**\n", 
    _prompt: str=UNION_TEMPLATE, 
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
    data = _add_header(list(text), _header, _sep)
    response, _, _ = Engines.get(_model).forward(
        prompt=str_formatter(_prompt, texts=data, sep=_sep), structure=_structure
    )
    return response


def union_stream(*text: str, _sep: str='\n', _header: str="**TEXT {}**\n", _prompt: str=UNION_TEMPLATE, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> t.Iterator[str]:
    """Combine multiple texts into one unified text using a language model.

    Args:
        *text: The texts to combine.
        _separator: The separator to use between texts.
        _prompt: The prompt to use for the language model.

    Returns:
        An iterator over the combined text.
    """
    data = _add_header(list(text), _header, _sep)
    for chunk, _, _ in Engines.get(_model).stream(
        prompt=str_formatter(_prompt, texts=data, sep=_sep),
        structure=_structure
    ):
        yield chunk


async def async_union(*text: str, _sep: str='\n', _header: str="**TEXT {}**\n", _prompt: str=UNION_TEMPLATE, _model: LangModel | None | str = None, _structure: t.Dict | None | pydantic.BaseModel = None) -> str:
    """Combine multiple texts into one unified text using a language model.

    Args:
        *text: The texts to combine.
        _separator: The separator to use between texts.
        _prompt: The prompt to use for the language model.

    Returns:
        The combined text.
    """
    data = _add_header(list(text), _header, _sep)
    response, _, _ = await Engines.get(_model).aforward(
        prompt=str_formatter(_prompt, texts=data, sep=_sep), structure=_structure
    )
    return response


async def union_astream(
    *text: str, 
    _sep: str='\n', 
    _header: str="**TEXT {}**\n", 
    _prompt: str=UNION_TEMPLATE, 
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
    data = _add_header(list(text), _header, _sep)
    async for chunk, _, _ in Engines.get(_model).astream(
        prompt=str_formatter(_prompt, texts=data, sep=_sep),
        structure=_structure
    ):
        yield chunk




def intersect(
    *texts: str,
    _sep: str='\n',
    _header: str="**TEXT {}**\n",
    _prompt: str=INTERSECTION_TEMPLATE,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        The intersection of the texts.
    """
    data = _add_header(list(texts), _header, _sep)
    response, _, _ = Engines.get(_model).forward(
        prompt=str_formatter(_prompt, texts=data), structure=_structure
    )
    return response


def intersect_stream(
    *texts: str,
    _sep: str='\n',
    _header: str="**TEXT {}**\n",
    _prompt: str=INTERSECTION_TEMPLATE,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.Iterator[str]:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        An iterator over the intersection of the texts.
    """
    data = _add_header(list(texts), _header, _sep)
    for chunk, _, _ in Engines.get(_model).stream(
        prompt=str_formatter(_prompt, texts=data), structure=_structure
    ):
        yield chunk

async def async_intersect(
    *texts: str,
    _sep: str='\n',
    _header: str="**TEXT {}**\n",
    _prompt: str=INTERSECTION_TEMPLATE,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> str:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        The intersection of the texts.
    """
    data = _add_header(list(texts), _header, _sep)
    response, _, _ = await Engines.get(_model).aforward(
        prompt=str_formatter(_prompt, texts=data), structure=_structure
    )
    return response


async def intersect_astream(
    *texts: str,
    _sep: str='\n',
    _header: str="**TEXT {}**\n",
    _prompt: str=INTERSECTION_TEMPLATE,
    _model: LangModel | None | str = None,
    _structure: t.Dict | None | pydantic.BaseModel = None
) -> t.AsyncIterator[str]:
    """Get the intersection of multiple texts using a language model.

    Args:
        *texts: The texts to intersect.

    Returns:
        An async iterator over the intersection of the texts.
    """
    data = _add_header(list(texts), _header, _sep)
    async for chunk, _, _ in Engines.get(_model).astream(
        prompt=str_formatter(_prompt, texts=data), structure=_structure
    ):
        yield chunk


def symmetric_difference(
    a: str,
    b: str,
    _prompt: str=DIFFERENCE_TEMPLATE,
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
    response, _, _ = Engines.get(_model).forward(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    )
    return response


def symmetric_difference_stream(
    a: str,
    b: str,
    _prompt: str=DIFFERENCE_TEMPLATE,
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
    for chunk, _, _ in Engines.get(_model).stream(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    ):
        yield chunk


async def async_symmetric_difference(
    a: str,
    b: str,
    _prompt: str=DIFFERENCE_TEMPLATE,
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
    response, _, _ = await Engines.get(_model).aforward(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    )
    return response


async def symmetric_difference_astream(
    a: str,
    b: str,
    _prompt: str=DIFFERENCE_TEMPLATE,
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
    async for chunk, _, _ in Engines.get(_model).astream(
        prompt=str_formatter(_prompt, a=a, b=b), structure=_structure
    ):
        yield chunk



class Union(LangEngine):
    """Operation to combine multiple texts into one unified text using a language model.
    """
    def __init__(
        self,
        sep: str = '\n',
        header: str = "**TEXT {}**\n",
        prompt: str = UNION_TEMPLATE,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        super().__init__(
            prompt=prompt,
            model=model,
            structure=structure
        )
        self.sep = sep
        self.header = header

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
        prompt: str = INTERSECTION_TEMPLATE,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        super().__init__(
            prompt=prompt,
            model=model,
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
        prompt: str = DIFFERENCE_TEMPLATE,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        super().__init__(
            prompt=prompt,
            model=model,
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
        prompt: str = SYMMETRIC_DIFFERENCE_TEMPLATE,
        model: None | str = None,
        structure: t.Dict | None | pydantic.BaseModel = None
    ):
        super().__init__(
            prompt=prompt,
            model=model,
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
