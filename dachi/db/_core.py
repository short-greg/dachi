import pydantic
from abc import (
    abstractmethod, ABC
)
from functools import wraps
from typing import Callable
import pandas as pd
import typing

T = typing.TypeVar('T')


def conceptmethod(func: Callable[..., T]) -> Callable[..., T]:

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if not isinstance(cls, type):
            raise TypeError("This method can only be called on the class, not on instances.")
        return func(cls, *args, **kwargs)

    return classmethod(wrapper)


def abstractconceptmethod(func: Callable[..., T]) -> Callable[..., T]:
    func = abstractmethod(func)

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if not isinstance(cls, type):
            raise TypeError("This method can only be called on the class, not on instances.")
        return func(cls, *args, **kwargs)

    return classmethod(wrapper)


class BaseConcept(pydantic.BaseModel):

    Manager: typing.ClassVar['ConceptManager'] = None

    @conceptmethod
    def create(cls):
        pass

    @conceptmethod
    def update(cls):
        pass

    def save(self):
        self.Manager.add_row(self)

    @conceptmethod
    def get(cls, pk):
        concept = cls.Manager.get(cls, pk)
        return cls(
            concept.to_dict()
        )


class ConceptManager(ABC):

    @abstractmethod
    def add_concept(self, concept_name, concept_columns):
        pass

    @abstractmethod
    def add_row(self, concept: BaseConcept) -> BaseConcept:
        pass


class DFConceptManager(ConceptManager):

    def __init__(self):

        Outer = self
        class Concept(BaseConcept):
            Manager: typing.ClassVar['ConceptManager'] = Outer
        
            @conceptmethod
            def create(self):
                if Outer.concept_exists(self):
                    raise RuntimeError
                Outer.add_concept(self)
        
        self.Concept = Concept
        print(self.Concept)
        self._concepts: typing.Dict[str, pd.DataFrame] = {}

    def concept_exists(self, concept: typing.Type[BaseConcept]):

        return concept.__name__ in self._concepts

    def add_concept(self, concept: typing.Type[BaseConcept]) -> BaseConcept:
        schema = concept.schema()
        
        columns = {}
        for field_name, details in schema.get('properties').items():
            columns[field_name] = details.get('type')
        
        self._concepts[concept.__name__] = pd.DataFrame(
            columns=columns
        )
        
    def add_row(self, concept: typing.Type[BaseConcept]):
        
        df = self._concepts[concept.__name__]
        df.loc[df.index.max() + 1] = concept.dict()
        return df.loc[df.index.max() + 1]

    def get_row(self, concept: typing.Type[BaseConcept], pk: int) -> pd.Series:

        row = self._concepts[concept.__name__].loc[pk]
        return concept(
            **row.to_dict()
        )



# filter( Join(..., on=ColumnX, where=NestCol() == Col()))
