from abc import ABC, abstractmethod
from typing import Self
import typing
from ..process import Module, T
from enum import Enum
import numpy as np


# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve
# 4) Include
# 5) Exclude


# Use Key to replace column
class Key(object):
    
    def __init__(self, name: str):
        """
        Args:
            name (str): Name of the column
        """
        self.name = name
    
    def __eq__(self, other) -> 'Comp':
        """Check the eqquality of two columns

        Args:
            other : The other Col com compare with

        Returns:
            Comp: The comparison for equality
        """
        return Comp(self, other, lambda lhs, rhs: lhs == rhs)
    
    def __lt__(self, other) -> 'Comp':
        """Check whether column is less than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than
        """
        return Comp(self, other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other) -> 'Comp':
        """Check whether column is less than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than or equal
        """

        return Comp(self, other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other) -> 'Comp':
        """Check whether column is greater than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than
        """
        return Comp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other) -> 'Comp':
        """Check whether column is greater than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than or equal to
        """
        return Comp(self, other, lambda lhs, rhs: lhs >= rhs)


class CompF(Enum):
    
    OR = 'or'
    AND = 'and'
    NOT = 'not'
    GT = 'gt'
    GE = 'ge'
    LT = 'lt'
    LE = 'le'
    EQ = 'eq'


class Val(object):

    def __init__(self, val) -> None:
        self.val = val


# instead of passing a function
# the store probably needs to know how to prcess it?

# TODO: extend the ability to use T

class Comp(object):
    """Comparison used to filter the "Store"
    """

    def __init__(
        self, lhs: typing.Union['Comp', 'Key', T, typing.Any], 
        rhs: typing.Union['Comp', 'Key', T, typing.Any], 
        f: CompF
    ) -> None:
        """Comparison a comparison between a left hand element and a right-hand
        element

        Args:
            lhs (typing.Union[Comp;, Key;, typing.Any]): _description_
            rhs (typing.Union[Comp;, Key;, typing.Any]): _description_
            f (typing.Callable[[typing.Any, typing.Any], bool]): _description_
        """
        if (
            not isinstance(lhs, Comp) 
            and not isinstance(lhs, Key)
        ):
            lhs = Val(lhs)

        if (
            not isinstance(rhs, Comp) 
            and not isinstance(rhs, Key)
        ):
            rhs = Val(rhs)
        
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def __and__(self, other) -> 'Comp':

        return Comp(self, other, CompF.AND)

    def __or__(self, other) -> 'Comp':

        return Comp(self, other, CompF.OR)
    
    def __invert__(self) -> 'Comp':

        return Comp(None, self, CompF.NOT)


class QF(object):
    pass


class Query(ABC):

    # def annotate
    # def join

    @abstractmethod
    def limit(self) -> Self:
        pass

    @abstractmethod
    def where(self, comp: Comp) -> Self:
        pass

    @abstractmethod
    def join(self, query: 'Query', left: str, right: str, comp: Comp) -> Self:
        pass

    @abstractmethod
    def select(self, **kwargs: typing.Union[str, QF]):
        pass

    @abstractmethod
    def retrieve(self) -> typing.Any:
        pass

    @abstractmethod
    def order_by(self) -> Self:
        pass


class Store(ABC):
    """Store 
    """

    @abstractmethod
    def limit(self) -> Query:
        """Limit the number of results

        Returns:
            Query: _description_
        """
        pass

    @abstractmethod
    def where(self, comp: Comp) -> Query:
        pass

    @abstractmethod
    def retrieve(self) -> typing.Any:
        pass

    @abstractmethod
    def join(self, other: 'Query', left: str, right: str, comp: Comp) -> Self:
        pass

    @abstractmethod
    def select(self, **kwargs: typing.Union[str, QF]):
        pass

    @abstractmethod
    def order_by(self, keys: typing.List[str]) -> Self:
        pass


class Rep(object):

    def __init__(self, ids, vectors: np.ndarray):

        super().__init__()
        self.ids = ids
        self.vectors = vectors


class VectorStore(Store):

    @abstractmethod
    def like(self) -> Query:
        pass

    @abstractmethod
    def rep(self) -> Rep:
        pass


class VectorQuery(Query):

    @abstractmethod
    def like(self) -> 'VectorQuery':
        pass

    @abstractmethod
    def rep(self) -> Rep:
        pass


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()


retrieve = Retrieve()
