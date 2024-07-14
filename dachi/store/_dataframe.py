from typing import Self
import typing
from ..process import Module, T
from ._core import Query, Store, Comp, Key
import pandas as pd

# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve
# 4) Include
# 5) Exclude


class DFQuery(Query):
    """Queries the values of a dataframe
    """

    # def annotate
    # def join

    def __init__(
        self, store: 'DFStore', 
        limit: int=None, 
        where: Comp=None
    ):
        """_summary_

        Args:
            store (DFStore): 
            limit (int, optional): . Defaults to None.
            where (Comp, optional): . Defaults to None.
        """
        self._store = store
        self._limit = limit
        self._where = where or None

    def limit(self, n: int) -> Self:
        return DFQuery(
            self._store, n, self._where
        )

    def where(self, comp: Comp) -> Self:
        where = self._where & comp
        return DFQuery(
            self._store, self._limit, where
        )
    
    def _filter(self, comp: Comp) -> pd.DataFrame:
        """

        Args:
            comp (Comp): 

        Returns:
            pd.DataFrame: 
        """
        if isinstance(comp.lhs, Comp):
            lhs = self._filter(comp.lhs)
        elif isinstance(comp.lhs, Key):
            lhs = self._store.df[comp.lhs.name]
        else:
            lhs = comp.lhs

        if isinstance(comp.rhs, Comp):
            rhs = self._filter(comp.rhs)
        elif isinstance(comp.rhs, Key):
            rhs = self._store.df[comp.rhs.name]
        else:
            rhs = comp.rhs

        return self._store.df[comp.compare(lhs, rhs)]

    def retrieve(self) -> pd.DataFrame:
        """Run all of the operations

        Returns:
            pd.DataFrame: _description_
        """

        if self._where is not None:
            store = self._filter(self._where)
        else:
            store = self._store.df
        if self._limit is not None:
            store = store.iloc[:self._limit]
        return store


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
            self, n, None
        )

    def where(self, comp: Comp) -> Query:
        """Use to filter the dataframe

        Args:
            comp (Comp): The comparison to use

        Returns:
            Query: The query to filter the dataframe with
        """
        return DFQuery(self, None, comp)

    def retrieve(self) -> pd.DataFrame:
        return self._df


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()
