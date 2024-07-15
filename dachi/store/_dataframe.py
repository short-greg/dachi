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

def df_sort(df: pd.DataFrame, order_by: typing.List[str]):
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

def df_filter(
    df: pd.DataFrame, comp: Comp, 
) -> pd.DataFrame:
    """

    Args:
        comp (Comp): 

    Returns:
        pd.DataFrame: 
    """

    if isinstance(comp.lhs, Comp):
        lhs = df_filter(comp.lhs, df)
    elif isinstance(comp.lhs, Key):
        lhs = df[comp.lhs.name]
    else:
        lhs = comp.lhs

    if isinstance(comp.rhs, Comp):
        rhs = df_filter(comp.rhs, df)
    elif isinstance(comp.rhs, Key):
        rhs = df[comp.rhs.name]
    else:
        rhs = comp.rhs

    return comp.compare(lhs, rhs)

def df_join(df: pd.DataFrame, join: Join) -> Self:
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
    filter = df_filter(join.comp, df)
    return df[filter]

def df_select(df: pd.DataFrame, select: typing.Union[None, typing.Dict]=None) -> pd.DataFrame:
    
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
            store = df_join(store, join)

        if where is not None:
            store = store[df_filter(store, where)]

        # TODO: Add in the selection
        if order_by is not None:
            store = df_sort(store, order_by)
        if limit is not None:
            store = store.iloc[:limit]
        if select is not None:
            df_select(store, select)
        return store
