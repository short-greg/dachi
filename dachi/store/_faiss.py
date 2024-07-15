from typing import Any, Dict, List
from dachi.store._core import QF, Comp, Join, Rep, Like
from ._core import VectorStore
from . import _dataframe as df_utils
import faiss
import typing
import pandas as pd
import numpy as np


class FAISSStore(VectorStore):

    def __init__(
        self, 
        ids: pd.Series,
        data: pd.Series, 
        vectors: pd.Series, 
        index: faiss.IndexIDMap2
    ) -> None:
        """

        Args:
            ids (pd.Series): 
            data (pd.Series): 
            vectors (pd.Series): 
            index (faiss.IndexIDMap2): 
        """
        super().__init__()
        self._data = pd.DataFrame(
            {'id': ids, 'data': data, 'vector': vectors}
        )
        self._index = index
    
    def _like_helper(self, df: pd.DataFrame, like: Like) -> pd.DataFrame:
        
        ids = self._index.search(
            like.like, like.n
        )
        ids = np.flatten(ids)
        return df[df['id'].isin(ids)]

    def retrieve(
        self, 
        select: Dict[str, typing.Union[str, QF]] = None, 
        joins: List[Join] = None, 
        like: Like = None,
        where: Comp = None, 
        order_by: List[str] = None, 
        limit: int = None
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
        store = self._df

        if like is not None:
            store = self._like_helper(store, like)

        joins = joins or []
        for join in joins:
            store = df_utils.df_join(store, join)

        if where is not None:
            store = store[df_utils.df_filter(store, where)]

        if order_by is not None:
            store = df_utils.df_sort(store, order_by)
        if limit is not None:
            store = store.iloc[:limit]
        if select is not None:
            df_utils.df_select(store, select)
        return store
