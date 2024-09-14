from typing import Any, Dict, List
from dachi.store._core import QF, Comp, Join, Rep, Like

from ._core import (
    VectorizedQuery, VectorizedStore, Comp, 
    Key, Join, CompF, Val, Joinable, JoinableQuery, JoinableStore
)
from ._core import Vectorized, Store
from . import _dataframe as df_utils
import faiss
import typing
import pandas as pd
import numpy as np

# is JoinableStore necessary?


class FAISSStore(VectorizedStore, JoinableStore):

    def __init__(
        self, 
        ids: typing.List[int],
        data: typing.List, 
        index: faiss.IndexIDMap2,
        embedder: typing.Callable[[typing.List], typing.List],
        embeddings: typing.List=None, 
    ) -> None:
        """

        Args:
            ids (pd.Series): 
            data (pd.Series): 
            vectors (pd.Series): 
            index (faiss.IndexIDMap2): 
        """
        super().__init__()
        if embeddings is None:
            embeddings = embedder(data)
        self._data = pd.DataFrame(
            {'id': ids, 'data': data, 'emveddings': embeddings}
        )
        self._data = self._data.reindex('id')
        self._embedder = embedder
        self._index = index

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def store(
        self, data, id=None, embeddings=None
    ):
        
        self.bulk_store([data], [id] if id is not None else None, embeddings)
        # if index is None:
        #     self._df.iloc[len(self._df.index)] = kwargs
        # else:
        #     self._df.loc[max(self._df.index)] = kwargs

    def bulk_store(
        self, data: typing.List, ids=None, embeddings: typing.List=None
    ):
        if ids is None:
            start_val = self._data['id'].max() + 1
            ids = [start_val + i for i in range(len(data))]
        if embeddings is None:
            embeddings = self._embedder(data)
        
        df = pd.DataFrame(
            {'id': ids, 'data': data, 'embeddings': embeddings}
        )
        df = df.reindex('id')
        
        if ids is None:
            self._df = pd.concat(
                [self._df, df]
            )
        else:
            self._df = self._df.reindex(df.index.union(df.index))
            self._df.update(df)   

    @property
    def index(self) -> faiss.IndexIDMap2:
        return self._index

    @property
    def query(self) -> 'FAISSQuery':

        return FAISSQuery(
            self
        )


class FAISSQuery(VectorizedQuery, JoinableQuery):

    def _like_helper(self, df: pd.DataFrame, like: Like) -> pd.DataFrame:
        
        ids = self._index.search(
            like.like, like.n
        )
        ids = np.flatten(ids)
        return df[df['id'].isin(ids)]

    def values(
        self, 
        # select: Dict[str, typing.Union[str, QF]] = None, 
        # joins: List[Join] = None, 
        # like: Like = None,
        # where: Comp = None, 
        # order_by: List[str] = None, 
        # limit: int = None
    ) -> Any:
        """

        Args:
            select (Dict[str, typing.Union[str, QF]], optional): . Defaults to None.
            joins (List[Join], optional): . Defaults to None.
            like (Like, optional): . Defaults to None.
            where (Comp, optional): . Defaults to None.
            order_by (List[str], optional): . Defaults to None.
            limit (int, optional): . Defaults to None.

        Returns:
            Any: 
        """
        store = self.df

        if self._like is not None:
            store = self._like_helper(store, self._like)

        joins = joins or []
        for join in joins:
            store = df_utils.df_join(store, join)

        if self._where is not None:
            store = store[df_utils.df_filter(store, self._where)]

        if self._order_by is not None:
            store = df_utils.df_sort(store, self._order_by)
        if self._limit is not None:
            store = store.iloc[:self._limit]
        if self.__delattr__select is not None:
            df_utils.df_select(store, self._select)
        return store
