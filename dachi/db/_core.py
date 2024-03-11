import pydantic
from abc import (
    abstractmethod, ABC
)
from functools import wraps
from typing import Callable
import pandas as pd
import typing
from functools import singledispatchmethod
import faiss

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
        self.Manager.add_concept_entry(self)

    @conceptmethod
    def get(cls, pk):
        concept = cls.Manager.get(cls, pk)
        return cls(
            concept.to_dict()
        )


class Code(object):

    pass


class Source(object):

    pass


# 1) The table
# 2) The 

class BaseRep(pydantic.BaseModel):

    Manager: typing.ClassVar['ConceptManager'] = None
    source: Source = None
    code: Code = None
    # Want to be able to add labels
    # # 

    @conceptmethod
    def create(cls):
        pass

    @conceptmethod
    def update(cls):
        pass

    def save(self):
        self.Manager.add_concept_entry(self)

    @conceptmethod
    def get(cls, pk):
        concept = cls.Manager.get(cls, pk)
        return cls(
            concept.to_dict()
        )


class RepField(object):

    pass


class ForeignIdx(RepField):

    pass


class TextIdx(RepField):

    pass


class ConceptManager(ABC):

    @abstractmethod
    def add_concept(self, concept_name, concept_columns):
        pass

    @abstractmethod
    def add_concept_entry(self, concept: BaseConcept) -> BaseConcept:
        pass


class DFConceptManager(ConceptManager):

    def __init__(self):

        Outer = self

        class Rep(BaseRep):
            Manager: typing.ClassVar['ConceptManager'] = Outer
        
            @conceptmethod
            def create(self):
                if Outer.concept_exists(self):
                    raise RuntimeError
                Outer.add_rep(self)
            
        self.Rep = Rep
        class Concept(BaseConcept):
            Manager: typing.ClassVar['ConceptManager'] = Outer
            Rep: typing.ClassVar[BaseRep] = self.Rep
        
            @conceptmethod
            def create(self):
                if Outer.concept_exists(self):
                    raise RuntimeError
                Outer.add_concept(self)
        
        self.Concept = Concept
        self._reps: typing.Dict[str, pd.DataFrame] = {}
        self._indices: typing.Dict[str, pd.DataFrame] = {}
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

    def add_rep(self, rep: typing.Type[BaseRep]) -> BaseRep:
        schema = rep.schema()
        
        columns = {}

        # 

        for field_name, details in schema.get('properties').items():
            columns[field_name] = details.get('type')
        
        # TODO: make this more dynamic
        # must specify d for IndexFlat
        self._indices[rep.__name__] = faiss.IndexFlatL2()
        self._reps[rep.__name__] = pd.DataFrame(
            columns=columns
        )

    def add_concept_entry(self, concept: typing.Type[BaseConcept]):
        
        df = self._concepts[concept.__name__]
        df.loc[df.index.max() + 1] = concept.dict()
        return df.loc[df.index.max() + 1]

    def add_rep_entry(self, rep: typing.Type[BaseRep]):
        
        df = self._reps[rep.__name__]
        df.loc[df.index.max() + 1] = rep.dict()
        return df.loc[df.index.max() + 1]

    def get_concept_entry(self, concept: typing.Type[BaseConcept], pk: int) -> pd.Series:

        row = self._concepts[concept.__name__].loc[pk]
        return concept(
            **row.to_dict()
        )

    def get_rep_entry(self, rep: typing.Type[BaseRep], pk: int) -> pd.Series:

        row = self._reps[rep.__name__].loc[pk]
        return rep(
            **row.to_dict()
        )


# filter( Join(..., on=ColumnX, where=NestCol() == Col()))
