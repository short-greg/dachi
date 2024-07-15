from typing import Self
import typing

from dachi.store._core import QF
from ..process import Module, T
from ._core import Query, Store, Comp, Key
import pandas as pd

# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve

class DFQuery(Query):
    """Queries the values of a dataframe
    """

    # def annotate
    # def join

    def __init__(
        self, store: 'DFStore', 
        limit: int=None, 
        where: Comp=None,
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        order_by: typing.List[str]=None
    ):
        """

        Args:
            store (DFStore): 
            limit (int, optional): . Defaults to None.
            where (Comp, optional): . Defaults to None.
        """
        self._store = store
        self._limit = limit
        self._select = select or {}
        self._where = where or None
        self._order_by = order_by or None

    def limit(self, n: int) -> Self:
        return DFQuery(
            self._store, n, self._where, self._order_by
        )

    def where(self, comp: Comp) -> Self:
        where = self._where & comp
        return DFQuery(
            self._store, self._limit, where, self._order_by
        )
    
    def join(self, other: 'Query', left_on: str, right_on: str, comp: Comp, alias: str=None, how: str='inner') -> Self:
        
        # TODO: Add the alias
        other = other.df()
        df = self._store.df.merge(
            other, how, left_on=left_on, right_on=right_on
        )
        filter = self._filter(comp, df)
        return df[filter]
    
    def _sort(self, df: pd.DataFrame):
        ascending = []
        sort_by = []
        for criterion in self._order_by:
            if criterion.startswith('+'):
                sort_by.append(criterion[1:])
                ascending.append(True)
            elif criterion.startswith('-'):
                sort_by.append(criterion[1:])
                ascending.append(False)
        return df.sort_values(
            by=sort_by, 
            ascending=ascending
        )
    
    def select(self, **kwargs: typing.Union[str, QF]):

        for k, v in kwargs.items():
            if isinstance(v, QF):
                # TODO: Confirm it is a valid function
                pass
        
        return DFQuery(
            self._store, self._limit, 
            self._where, self._order_by, kwargs
        )
                
    def _filter(self, comp: Comp, df: pd.DataFrame=None) -> pd.DataFrame:
        """

        Args:
            comp (Comp): 

        Returns:
            pd.DataFrame: 
        """
        df = df if df is not None else self._store.df

        if isinstance(comp.lhs, Comp):
            lhs = self._filter(comp.lhs, df)
        elif isinstance(comp.lhs, Key):
            lhs = df[comp.lhs.name]
        else:
            lhs = comp.lhs

        if isinstance(comp.rhs, Comp):
            rhs = self._filter(comp.rhs, df)
        elif isinstance(comp.rhs, Key):
            rhs = df[comp.rhs.name]
        else:
            rhs = comp.rhs

        return comp.compare(lhs, rhs)

    def retrieve(self) -> pd.DataFrame:
        """Run all of the operations

        Returns:
            pd.DataFrame: 
        """

        store = self._store.df
        if self._where is not None:
            store = store[self._filter(self._where)]

        # TODO: Add in the selection
        if self._order_by is not None:
            store = self._sort(store)
        if self._limit is not None:
            store = store.iloc[:self._limit]
        return store

    def order_by(self, keys: typing.List[str]) -> Self:
        return DFQuery(
            self._store, self._limit, self._where, keys
        )


class DFStore(Store):

    def __init__(self, df: pd.DataFrame):
        """

        Args:
            df (pd.DataFrame): 
        """
        super().__init__()
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def limit(self, n: int) -> Query:
        return DFQuery(
            self, limit=n
        )

    def where(self, comp: Comp) -> Query:
        """Use to filter the dataframe

        Args:
            comp (Comp): The comparison to use

        Returns:
            Query: The query to filter the dataframe with
        """
        return DFQuery(self, where=comp)

    def retrieve(self) -> pd.DataFrame:
        return self._df

    def order_by(self, keys: typing.List[str]) -> Self:
        return DFQuery(
            self, order_by=keys
        )


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()
