# # 1st party
# from abc import abstractmethod, ABC
# from dataclasses import dataclass, fields, MISSING, field
# import typing
# from typing_extensions import Self
# from functools import wraps
# import inspect
# import math
# from dataclasses import InitVar

# # 3rd party
# from pydantic import BaseModel
# import pandas as pd
# import faiss
# import numpy as np
# from pydantic import Field

# from ._rep import Rep

# class Filter(object):

#     @abstractmethod
#     def query(self, df: pd.DataFrame, rep_map: Rep) -> pd.Series:
#         pass

#     def __call__(self, df: pd.DataFrame, rep_map: Rep) -> pd.DataFrame:
#         """

#         Args:
#             df (pd.DataFrame): The dataframe to retrieve from
#             rep_map (RepMap): The repmap to retrieve from

#         Returns:
#             pd.DataFrame: 
#         """
#         return df[self.query(df, rep_map)]

#     def __xor__(self, other) -> 'Comp':

#         return Comp(self, other, lambda lhs, rhs: lhs ^ rhs)

#     def __and__(self, other):

#         return Comp(self, other, lambda lhs, rhs: lhs & rhs)

#     def __or__(self, other):

#         return Comp(self, other, lambda lhs, rhs: lhs | rhs)
    
#     def __invert__(self):

#         return Comp(None, self, lambda lhs, rhs: ~rhs)


# class Comp(Filter):

#     def __init__(
#         self, lhs: typing.Union['Comp', 'Col', typing.Any], 
#         rhs: typing.Union['Comp', 'Col', typing.Any], 
#         f: typing.Callable[[typing.Any, typing.Any], bool]
#     ) -> None:
#         """_summary_

#         Args:
#             lhs (typing.Union[BinComp;, Col;, typing.Any]): _description_
#             rhs (typing.Union[BinComp;, &#39;Col&#39;, typing.Any]): _description_
#             f (typing.Callable[[typing.Any, typing.Any], bool]): _description_
#         """
        
#         if not isinstance(lhs, Filter) and not isinstance(lhs, Col):
#             lhs = Val(lhs)

#         if not isinstance(rhs, Filter) and not isinstance(rhs, Col):
#             rhs = Val(rhs)
        
#         self.lhs = lhs
#         self.rhs = rhs
#         self.f = f

#     def query(self, df: pd.DataFrame, rep_map: Rep) -> pd.Series:
#         """

#         Args:
#             df (pd.DataFrame): 

#         Returns:
#             The filter by comparison: 
#         """
#         print(type(self.lhs))
#         lhs = self.lhs.query(df, rep_map)
#         print('Result: ', type(lhs))
#         rhs = self.rhs.query(df, rep_map)
#         print('executing ', type(lhs), type(rhs))
#         return self.f(lhs, rhs)


# class BaseQuery(ABC):

#     @abstractmethod
#     def filter(self, comp: Comp) -> Self:
#         pass

#     @abstractmethod
#     def select(self, **kwargs) -> 'DerivedQuery':
#         pass
    
#     @abstractmethod
#     def __iter__(self) -> typing.Iterator['BaseConcept']:
#         pass
        
#     @abstractmethod
#     def df(self) -> pd.DataFrame:
#         pass

#     @abstractmethod
#     def join(self, query: 'BaseQuery', alias: str, on_: str, how: str='inner') -> 'DerivedQuery':
#         pass

#     def inner(self, query: 'BaseQuery', alias: str, on_: str) -> Self:
#         return self.join(query, alias, on_, 'inner')

#     def left(self, query: 'BaseQuery', alias: str, on_: str) -> Self:
#         return self.join(query, alias, on_, 'left')

#     def right(self, query: 'BaseQuery', alias: str, on_: str) -> Self:
#         return self.join(query, alias, on_, 'right')

