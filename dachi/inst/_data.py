# 1st party
import typing
from typing import Self

# 3rd party
import pydantic
import typing as t
import pandas as pd

# Local
from dachi.core import Renderable, render


class Record(Renderable):
    """Use to create a pairwise object
    """
    def __init__(self, indexed: bool=False, **kwargs):
        """Create a pairwise object

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        super().__init__()
        self._data = pd.DataFrame(kwargs)
        self.indexed = indexed

    def extend(self, **kwargs) -> t.Self:
        """Extend the pairwise object with new items

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        self._data = pd.concat(
            [self._data, pd.DataFrame(kwargs)],
            ignore_index=True
        )

    def append(self, **kwargs) -> t.Self:
        """Append the pairwise object with new items

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        kwargs = {k: [v] for k, v in kwargs.items()}
        self._data = pd.concat(
            [self._data, pd.DataFrame(kwargs)],
            ignore_index=True
        )
    
    def join(self, **kwargs) -> 'Record':
        """Join the pairwise object with new items

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        data = self._data.copy()
        data = data.assign(**kwargs)
        record = Record()
        record._data = data
        return record
    
    def clear(self):
        """Reset the record
        """
        self._data = pd.DataFrame()
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the dataframe for the pairwise object

        Returns:
            pd.DataFrame: The dataframe for the pairwise object
        """
        return pd.DataFrame(self._items)
    
    def __getitem__(self, key: str):

        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if the record has the column specified by key

        Args:
            key (str): The key to check

        Returns:
            bool: Whether contained
        """
        return key in self._data.columns.values
    
    def __len__(self) -> int:
        """Get the length of the pairwise object

        Returns:
            int: The length of the pairwise object
        """
        return len(self._data.index)
    
    def render(self) -> str:
        """Render the pairwise object

        Returns:
            str: The rendered string
        """
        return render(self._data.to_dict(orient='records'))

    def top(self, field: str, largest: bool=True):

        if field not in self._data.columns:
            raise KeyError(f"Field '{field}' not found in the dataframe.")

        if largest:
            idx = self._data[field].idxmax()
        else:
            idx = self._data[field].idxmin()

        return self._data.loc[idx]


class Term(pydantic.BaseModel, Renderable):
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
        """Render the DataList

        Returns:
            str: The data list rendered as a string
        """
        base = f"{self.name}: {self.definition}"
        if len(self.meta) == 0:
            return base
        
        meta = '\n-'.join(f'{k}: {v}' for k, v in self.meta.items())
        return f'{base}\n{meta}'


class Glossary(pydantic.BaseModel, Renderable):
    """A glossary contains a list of terms
    """
    terms: typing.Dict[str, Term]

    def __init__(self, terms: typing.List[Term] = None):
        """Create a glossary of terms

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
        """Get an item from the glossary
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
        """Join the glossary with terms

        Args:
            val (typing.Union[Term, typing.List[Term]]): The value to add

        Returns: The value joined
            ...
        """
        if isinstance(val, Term):
            self.terms[val.name] = val
        else:
            for term_i in val:
                self.terms[term_i.name] = val
        return val
    
    def add(self, term: Term) -> Self:
        """Add a term to the glossary

        Args:
            name (str): The name of the term to add
            definition (str): The definition of the term

        Returns:
            Self: The glossary
        """
        self.terms[term.name] = term
        return self

    def remove(self, name: str) -> Self:
        """Add a term to the glossary

        Args:
            name (str): The name of the term to add
            definition (str): The definition of the term

        Returns:
            Self: The glossary
        """
        del self.terms[name]
        return self

    def exclude(self, *meta: str) -> 'Glossary':
        """Filter by excluding meta fields from the glossary

        Returns:
            Glossary: The filtered glossary
        """
        terms = []
        for name, term in self.terms.items():
            cur_meta = {
                k: v for k, v in term.meta.items() if k not in meta
            }
            terms.append(Term(name, **cur_meta))
        return Glossary(terms)

    def include(self, *meta: str) -> 'Glossary':
        """Filter by including meta fields from the glossary

        Returns:
            Glossary: The filtered glossary
        """
        terms = []
        for name, term in self.terms.items():
            cur_meta = {
                k: v for k, v in term.meta.items() if k in meta
            }
            terms.append(Term(name, **cur_meta))
        return Glossary(terms)

    def update(self, term: Term) -> 'Glossary':

        self.terms[term.name] = term
        return self

    def render(self) -> str:
        """Convert the glossary into a string

        Returns:
            str: The name of the string
        """
        return '\n'.join(term.render() for _, term in self.terms.items())

    def items(self) -> typing.Iterator[typing.Tuple[str, Term]]:
        """Looop over the items in the Glossary

        Returns:
            typing.Iterator[typing.Tuple[str, Term]]: The iterator for the terms
        """
        return self.terms.items()


# class Ref(pydantic.BaseModel, Renderable):
#     """Reference to another description.
#     Useful when one only wants to include the 
#     name of a description in part of the prompt
#     """
#     desc: Description

#     @property
#     def name(self) -> str:
#         """Get the name of the ref

#         Returns:
#             str: The name of the ref
#         """
#         return self.desc.name

#     def render(self) -> str:
#         """Generate the text rendering of the ref

#         Returns:
#             str: The name for the ref
#         """
#         return self.desc.name

