from abc import ABC, abstractmethod
from typing import Self
import typing
from .._core import Module
from ..graph import T
from enum import Enum
import numpy as np
from dataclasses import dataclass
import pandas as pd


class _Types(Enum):

    UNCHANGED = 'UNCHANGED'

UNCHANGED = _Types.UNCHANGED


def coalesce(change, cur) -> typing.Any:

    if change is UNCHANGED:
        return cur
    return change


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
        return Comp(self, other, CompF.EQ) # lambda lhs, rhs: lhs == rhs)
    
    def __lt__(self, other) -> 'Comp':
        """Check whether column is less than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than
        """
        return Comp(self, other,  CompF.LT)# lambda lhs, rhs: lhs < rhs)

    def __le__(self, other) -> 'Comp':
        """Check whether column is less than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than or equal
        """

        return Comp(self, other, CompF.LE) # lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other) -> 'Comp':
        """Check whether column is greater than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than
        """
        return Comp(self, other, CompF.GT) # lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other) -> 'Comp':
        """Check whether column is greater than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than or equal to
        """
        return Comp(self, other,  CompF.GE) #  lambda lhs, rhs: lhs >= rhs)


class CompF(Enum):
    
    OR = 'or'
    AND = 'and'
    NOT = 'not'
    XOR = 'xor'
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


class QF(ABC):
    
    @abstractmethod
    def __call__(self, store: 'Store') -> 'Store':
        pass


@dataclass
class Join(object):

    query: 'Query'
    left: str
    right: str
    comp: Comp=None
    alias: typing.Dict[str, str]=None
    how: str='inner'

# Join(query, left, right, comp)

class Query(ABC):

    # def annotate
    # def join

    def __init__(
        self, store: 'Store', 
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        joins: typing.List[Join]=None,
        where: Comp=None,
        order_by: typing.List[str]=None,
        limit: int=None, 
    ):
        """

        Args:
            store (DFStore): 
            limit (int, optional): . Defaults to None.
            where (Comp, optional): . Defaults to None.
        """
        self._store = store
        self._limit = limit
        self._select = select
        self._where = where or None
        self._order_by = order_by or None
        self._joins = joins or []

    def limit(self, n: int) -> Self:

        return self.spawn(
            limit=n
        )

    def where(self, comp: Comp) -> Self:
        where = self._where & comp
        return self.spawn(
            where=where
        )

    def like(self, like: typing.Union[typing.List, typing.Any], n: int) -> 'VectorQuery':
        if not isinstance(self._store, Vectorized):
            raise RuntimeError(
                'The store does not have a vectorized '
                'operation so cannot use like'
            )
        return self.spawn(
            like=Like(like, n)
        )

    def rep(self) -> 'Rep':
        if not isinstance(self._store, Vectorized):
            raise RuntimeError(
                'The store does not have a vectorized '
                'operation so cannot use rep'
            )
        return self._store.rep(
            # same as retrieve
        )

    def spawn(
        self, 
        select: typing.Dict[str, typing.Union[QF, str]]=UNCHANGED, 
        joins: typing.List[Join]=UNCHANGED, 
        where: Comp=UNCHANGED, 
        order_by: typing.List[str]=UNCHANGED,
        limit: int=UNCHANGED
    ):
        return self.__class__(
            self._store, 
            select=coalesce(select, select), 
            joins=coalesce(joins, self._joins), 
            where=coalesce(where, self._where),
            order_by=coalesce(order_by, self._order_by),
            coalesce=coalesce(limit, self._limit), 
        )

    def select(self, **kwargs: typing.Union[str, QF]) -> Self:

        for k, v in kwargs.items():
            if isinstance(v, QF):
                # TODO: Confirm it is a valid function
                pass
        
        kwargs = {
            **self.kwargs(),
            'select': kwargs
        }
        return Query(self._store, **kwargs)
    
    def join(
        self, query: 'Query', left: str, 
        right: str, comp: Comp, 
        alias: typing.Dict[str, str]=None, how: str='inner'
    ) -> Self:
        
        if not isinstance(self._store, Joinable):
            raise RuntimeError(
                'Cannot join since the store has '
                'no join operation'
            )
        joins = [*self._joins, Join(query, left, right, comp, alias, how)]
        return self.spawn(
            self._store, joins=joins
        )
                
    def retrieve(self) -> pd.DataFrame:
        """Run all of the operations

        Returns:
            pd.DataFrame: 
        """
        return self._store.retrieve(
            self._select, self._joins, self._where,
            self._order_by, self._limit
        )

    def order_by(self, keys: typing.List[str]) -> Self:

        return self.spawn(
            order_by=keys
        )


class Store(ABC):
    """Store 
    """

    def limit(self, n: int) -> Query:
        return Query(
            self, limit=n
        )

    def where(self, comp: Comp) -> Query:
        """Use to filter the dataframe

        Args:
            comp (Comp): The comparison to use

        Returns:
            Query: The query to filter the dataframe with
        """
        return Query(self, where=comp)

    @abstractmethod
    def retrieve(
        self,
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        joins: typing.List[Join]=None,
        where: Comp=None,
        order_by: typing.List[str]=None,
        limit: int=None, 
    ) -> typing.Any:
        pass

    def join(
        self, query: 'Query', left: str, 
        right: str, comp: Comp, 
        alias: typing.Dict[str, str]=None, how: str='inner'
    ) -> Self:
        kwargs = self.kwargs()
        kwargs['joins'] = [
            *self._joins, Join(query, left, right, comp, alias, how)]
        return Query(
            self, **kwargs
        )

    def select(self, **kwargs: typing.Union[str, QF]) -> Query:

        for k, v in kwargs.items():
            if isinstance(v, QF):
                # TODO: Confirm it is a valid function
                pass
        
        kwargs = {
            **self.kwargs(),
            'select': kwargs
        }
        return Query(self._store, **kwargs)

    def order_by(self, keys: typing.List[str]) -> Self:
        return Query(
            self, order_by=keys
        )


class Rep(object):

    def __init__(self, ids, vectors: np.ndarray):

        super().__init__()
        self.ids = ids
        self.vectors = vectors


class Vectorized(ABC):
    """Mixin to handle "Vector" Stores
    """

    @abstractmethod
    def like(self) -> Query:
        pass

    @abstractmethod
    def rep(self) -> Rep:
        pass


class Joinable(ABC):
    
    @abstractmethod
    def join(self, join: Join) -> Query:
        pass


@dataclass
class Like(object):

    like: typing.List
    n: int


# class VectorQuery(Query):

#     def __init__(
#         self, store: Vectorized, select: typing.Dict[str, str | QF] = None, 
#         joins: typing.List[Join] = None, where: Comp = None, 
#         order_by: typing.List[str] = None, limit: int = None,
#         like: Like=None
#     ):
#         """

#         Args:
#             store (VectorStore): 
#             select (typing.Dict[str, str  |  QF], optional): . Defaults to None.
#             joins (typing.List[Join], optional): . Defaults to None.
#             where (Comp, optional): . Defaults to None.
#             order_by (typing.List[str], optional): . Defaults to None.
#             limit (int, optional): . Defaults to None.
#             like (Like, optional): . Defaults to None.
#         """
#         super().__init__(store, select, joins, where, order_by, limit)
#         self._like = like

#     def spawn(
#         self, 
#         select: typing.Dict[str, typing.Union[QF, str]]=UNCHANGED, 
#         joins: typing.List[Join]=UNCHANGED, 
#         where: Comp=UNCHANGED, 
#         order_by: typing.List[str]=UNCHANGED,
#         limit: int=UNCHANGED,
#         like: Like=None
#     ):
#         return self.__class__(
#             self._store, 
#             select=coalesce(select, select), 
#             joins=coalesce(joins, self._joins), 
#             where=coalesce(where, self._where),
#             order_by=coalesce(order_by, self._order_by),
#             coalesce=coalesce(limit, self._limit), 
#             like=coalesce(like, self._like)
#         )
    
#     def like(self, like: typing.Union[typing.List, typing.Any], n: int) -> 'VectorQuery':

#         return self.spawn(
#             like=Like(like, n)
#         )

#     def rep(self) -> Rep:
#         return self._store.rep(
#             # same as retrieve
#         )


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()


retrieve = Retrieve()


# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve
# 4) Include
# 5) Exclude
