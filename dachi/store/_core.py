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


# Use Key to replace column
class Key(object):
    pass


class Comp(object):
    pass


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
    def limit(self) -> Query:
        pass

    @abstractmethod
    def include(self) -> Query:
        pass

    @abstractmethod
    def exclude(self) -> Query:
        pass

    @abstractmethod
    def retrieve(self) -> typing.Any:
        pass


class VectorStore(Store):

    @abstractmethod
    def like(self) -> Query:
        pass


class VectorQuery(Query):

    @abstractmethod
    def like(self) -> 'VectorQuery':
        pass



class Retrieve(Module):

    def forward(self, query: typing.Union[Query, Store]) -> typing.Any:
        return query.retrieve()


retrieve = Retrieve()
