# 1st party
import typing
import typing
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import get_type_hints
from typing import Self
import typing

from ._core import (
    Struct, Renderable
)
from ._utils import generic_class
from ._ai import Message

# 3rd party
import pydantic

# local
from ._utils import (
    generic_class
)


S = typing.TypeVar('S', bound=Struct)


class Media:

    descr: str
    data: str



class StructList(Struct, typing.Generic[S]):
    """
    """

    structs: typing.List[S]

    def __getitem__(self, key) -> typing.Any:
        """

        Args:
            key (_type_): 

        Returns:
            typing.Any: 
        """
        return self.structs[key]
    
    def __setitem__(self, key: typing.Optional[int], value: S) -> typing.Any:
        """Set a value in the 

        Args:
            key (str): The key for the value to set
            value : The value to set

        Returns:
            S: the value that was set
        """
        if key is None:
            self.structs.append(value)
        else:
            self.structs[key] = value
        return value
    
    @classmethod
    def load_records(cls, records: typing.List[typing.Dict]) -> 'StructList[S]':
        """Load the struct list from records

        Args:
            records (typing.List[typing.Dict]): The list of records to load

        Returns:
            StructList[S]: The list of structs
        """
        structs = []
        struct_cls: typing.Type[Struct] = generic_class(S)
        for record in records:
            structs.append(struct_cls.load(record))
        return StructList[S](
            structs=structs
        )


class Description(Struct, Renderable, ABC):
    """Provide context in the prompt template
    """
    name: str = pydantic.Field(description='The name of the description.')

    @abstractmethod
    def render(self) -> str:
        pass


class Ref(Struct):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    desc: Description

    @property
    def name(self) -> str:
        """Get the name of the ref

        Returns:
            str: The name of the ref
        """
        return self.desc.name

    def render(self) -> str:
        """Generate the text rendering of the ref

        Returns:
            str: The name for the ref
        """
        return self.desc.name


class MediaMessage(Message):

    def __init__(self, source: str, media: typing.List[Media]):
        """

        Args:
            source (str): 
            media (typing.List[Media]): The media to use
        """
        super().__init__(
            source=source,
            data={
                'media': media
            }
        )

    def render(self) -> str:
        """Render the media message

        Returns:
            str: The rendered message
        """
        return f'{self.source}: Media [{self.media}]'


class Term(Struct):
    """Use to define a term used in the AI system
    for repeated usage
    """

    name: str
    definition: str
    meta: typing.Dict[str, typing.Union[str, typing.List[str]]]

    def __init__(self, name: str, definition: str, **meta: str):

        super().__init__(
            name=name, definition=definition, meta=meta
        )

    def __getitem__(
        self, key: typing.Union[str, typing.Iterable[str]]
    ) -> typing.Union[str, typing.List[str]]:
        """Get one or more items from meta

        Args:
            key (typing.Union[str, typing.Iterable[str]]): The key to retrieve with

        Returns:
            typing.Union[str, typing.List[str]]: The value retrieved by key
        """
        if isinstance(key, str):
            return self.meta[key]
        
        return [
            v for k, v in self.meta.items() if k in key
        ]

    def __setitem__(self, key: typing.Union[str, typing.List[str]], val: typing.Union[str, typing.List[str]]):
        """Set the meta data for the term

        Args:
            key (typing.Union[str, typing.List[str]]): the key or keys to the meta data
            val (typing.Union[str, typing.List[str]]): the data to set

        Returns:
            val
        """
        if isinstance(key, str):
            self.meta[key] = val
        else:
            for k, v in zip(key, val):
                self.meta[k] = v
        return val
    
    def render(self) -> str:
        
        base = f"{self.name}: {self.definition}"
        if len(self.meta) == 0:
            return base
        
        meta = '\n-'.join(f'{k}: {v}' for k, v in self.meta.items())
        return f'{base}\n{meta}'


class Glossary(Struct):

    terms: typing.Dict[str, Term]

    def __init__(self, terms: typing.List[Term] = None):
        """

        Args:
            terms (typing.List[Term], optional): . Defaults to None.
        """
        terms = terms or []
        super().__init__(
            terms={term.name: term for term in terms}
        )

    def __getitem__(
        self, key: typing.Union[str, typing.Iterable[str]]
    ) -> typing.Union[str, typing.List[str]]:
        """

        Args:
            key (typing.Union[str, typing.Iterable[str]]): 

        Returns:
            typing.Union[str, typing.List[str]]: 
        """
        if isinstance(key, str):
            return self.terms[key]
        
        return [
            v for k, v in self.terms.items() if k in key
        ]

    def join(
        self, val: typing.Union[Term, typing.List[Term]]
    ):
        """

        Args:
            val (typing.Union[Term, typing.List[Term]]): 

        Returns:
            ...
        """
        if isinstance(val, Term):
            self.terms[val.name] = val
        else:
            for term_i in val:
                self.terms[term_i.name] = val
        return val
    
    def add(self, name: str, definition: str, **meta) -> Self:

        self.terms[name] = Term(name, definition, **meta)
        return self
    
    def exclude(self, *meta: str) -> 'Glossary':

        terms = []
        for name, term in self.terms.items():
            cur_meta = {
                k: v for k, v in term.meta.items() if k not in meta
            }
            terms.append(Term(name, **cur_meta))
        return Glossary(terms)

    def include(self, *meta: str) -> 'Glossary':

        terms = []
        for name, term in self.terms.items():
            cur_meta = {
                k: v for k, v in term.meta.items() if k in meta
            }
            terms.append(Term(name, **cur_meta))
        return Glossary(terms)

    def render(self) -> str:
        
        return '\n'.join(term.render() for _, term in self.terms.items())
