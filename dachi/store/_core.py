from abc import ABC, abstractmethod
from typing import Self
import typing
from ..process import Module

# 1) "Store" [dataframe, etc]
#   the store needs to have 
# 2) "Query"
#   Query is specific to the store
#   Can specify anything on it
# 3) Retrieve
# 4) Include
# 5) Exclude


class Query(ABC):

    # def annotate
    # def join

    @abstractmethod
    def limit(self) -> Self:
        pass

    @abstractmethod
    def include(self) -> Self:
        pass

    @abstractmethod
    def exclude(self) -> Self:
        pass

    @abstractmethod
    def retrieve(self) -> typing.Any:
        pass


class Store(ABC):

    @abstractmethod
    def query(self) -> Query:
        pass

    @abstractmethod
    def retrieve(self) -> typing.Any:
        pass


class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()


retrieve = Retrieve()
