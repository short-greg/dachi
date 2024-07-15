from typing import Self
import typing

from dachi.store._core import QF
from ..process import Module, T
from ._core import Query, Store, Comp, Key, Join
import pandas as pd

# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve


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

    def _join_helper(self, df: pd.DataFrame, join: Join) -> Self:
        # TODO: Add the alias
        other: pd.DataFrame = other.df()
        if join.alias is not None:
            alias = {
                column: f'{alias}.{column}' for column in other.columns
            }
            other = other.rename(
                columns=alias
            )
        df = df.merge(
            other, join.how, left_on=join.left_on, 
            right_on=join.right_on
        )
        filter = self._filter(join.comp, df)
        return df[filter]
    
    def _sort(self, df: pd.DataFrame, order_by: typing.List[str]):
        ascending = []
        sort_by = []
        for criterion in order_by:
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
    
    def _select_helper(self, df: pd.DataFrame, select: typing.Union[None, typing.Dict]=None) -> pd.DataFrame:
        
        df_new = pd.DataFrame()
        if select is None:
            return df
        for k, v in select.items():

            if isinstance(v, QF):
                df_new[k] = v(df)
            elif v in df.columns.values:
                df_new[k] = df[v]
            elif v in df.index.values:
                df_new[k] = df.index[k]
            
        return df_new

    def _filter(
        self, df: pd.DataFrame, comp: Comp, 
    ) -> pd.DataFrame:
        """

        Args:
            comp (Comp): 

        Returns:
            pd.DataFrame: 
        """

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

    def retrieve(
        self,
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        joins: typing.List[Join]=None,
        where: Comp=None,
        order_by: typing.List[str]=None,
        limit: int=None, 
    ) -> pd.DataFrame:
        store = self._df

        for join in joins:
            store = self._join_helper(store, join)

        if where is not None:
            store = store[self._filter(store, where)]

        # TODO: Add in the selection
        if order_by is not None:
            store = self._sort(store, order_by)
        if limit is not None:
            store = store.iloc[:limit]
        if select is not None:
            self._select_helper(store, select)
        return store

    # def where(self, comp: Comp) -> Query:
    #     """Use to filter the dataframe

    #     Args:
    #         comp (Comp): The comparison to use

    #     Returns:
    #         Query: The query to filter the dataframe with
    #     """
    #     return Query(self, where=comp)

    # def order_by(self, keys: typing.List[str]) -> Self:
    #     return DFQuery(
    #         self, order_by=keys
    #     )


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()
