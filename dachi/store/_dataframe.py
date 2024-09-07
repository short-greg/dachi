from typing import Self
import typing

from ._core import QF
from ._core import JoinableStore, Comp, Key, Join, CompF, Val, Joinable
import pandas as pd


def df_sort(df: pd.DataFrame, order_by: typing.List[str]):
    ascending = []
    sort_by = []
    for criterion in order_by:
        if criterion.startswith('-'):
            sort_by.append(criterion[1:])
            ascending.append(False)
        elif criterion.startswith('+'):
            sort_by.append(criterion[1:])
            ascending.append(True)
        else:
            sort_by.append(criterion)
            ascending.append(True)

    return df.sort_values(
        by=sort_by, 
        ascending=ascending
    )


def df_compare(compf: CompF, val1, val2):
    
    if compf == CompF.OR:
        return val1 | val2
    if compf == CompF.AND:
        return val1 & val2
    if compf == CompF.XOR:
        return val1 ^ val2
    if compf == CompF.LE:
        return val1 <= val2
    if compf == CompF.LT:
        return val1 < val2
    if compf == CompF.GE:
        return val1 >= val2
    if compf == CompF.GT:
        return val1 > val2
    if compf == CompF.EQ:
        return val1 == val2
    if compf == CompF.NOT:
        assert val2 is None
        return ~val1
    
    raise RuntimeError


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
    elif isinstance(comp.lhs, Val):
        lhs = comp.lhs.val
    else:
        lhs = comp.lhs

    if isinstance(comp.rhs, Comp):
        rhs = df_filter(comp.rhs, df)
    elif isinstance(comp.rhs, Key):
        rhs = df[comp.rhs.name]
    elif isinstance(comp.rhs, Val):
        rhs = comp.rhs.val
    else:
        rhs = comp.rhs

    compared = df_compare(comp.f, lhs, rhs)
    print(compared)
    return compared

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


class DFStore(JoinableStore):

    def __init__(
        self, df: pd.DataFrame, 
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        where: Comp=None,
        order_by: typing.List[str]=None,
        limit: int=None
    ):
        """

        Args:
            df (pd.DataFrame): 
        """
        super().__init__(
            select, where, order_by, limit, 
        )
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def store(
        self, index=None, **kwargs
    ):
        if index is None:
            self._df.iloc[len(self._df.index)] = kwargs
        else:
            self._df.loc[max(self._df.index)] = kwargs

    def bulk_store(
        self, index=None, **kwargs
    ):
        df = pd.DataFrame(kwargs, index)
        if index is None:
            self._df = pd.concat(
                [self._df, df]
            )
        else:
            self._df = self._df.reindex(df.index.union(df.index))
            self._df.update(df)

    def retrieve(
        self,
        select: typing.Dict[str, typing.Union[str, QF]]=None,
        joins: typing.List[Join]=None,
        where: Comp=None,
        order_by: typing.List[str]=None,
        limit: int=None, 
    ) -> pd.DataFrame:
        store = self._df

        joins = joins or []
        for join in joins:
            store = df_join(store, join)

        if where is not None:
            store = store[df_filter(store, where)]

        if order_by is not None:
            store = df_sort(store, order_by)
        if limit is not None:
            store = store.iloc[:limit]
        if select is not None:
            store = df_select(store, select)
        return store

# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve
