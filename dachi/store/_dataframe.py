from abc import ABC, abstractmethod
from typing import Self
import typing
from ..process import Module
from ._core import Query, Store
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

    # def annotate
    # def join

    def __init__(self, store: 'DFStore', limit: int=None):
        
        self._store = store
        self._limit = limit

    def limit(self) -> Self:
        pass

    def include(self) -> Self:
        pass

    def exclude(self) -> Self:
        pass

    def retrieve(self) -> pd.DataFrame:

        df = self._store.df
        return df.iloc[:self._limit]


class DFStore(Store):

    def __init__(self, df: pd.DataFrame):

        super().__init__()
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def limit(self) -> Query:
        pass

    def include(self) -> Query:
        pass

    def exclude(self) -> Query:
        pass

    def retrieve(self) -> pd.DataFrame:
        return self._df


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()
