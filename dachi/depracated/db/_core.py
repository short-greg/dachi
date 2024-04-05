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
import numpy as np

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

    # @conceptmethod
    # def update(cls):
    #     pass

    def save(self):
        self.Manager.save_or_add_concept_entry(self)

    @conceptmethod
    def get(cls, pk):
        concept = cls.Manager.get(cls, pk)
        return cls(
            concept.to_dict()
        )
    
    def as_(self, rep: str) -> 'BaseConceptRep':
        
        return self.Manager.as_(self, rep)


class Code(object):

    pass


class Source(object):

    pass


# 1) The table
# 2) The

T = typing.TypeVar('T')


# Whats the point of the field?

# x: str
# y = Rep()

class Field(object):

    def __init__(self, default=None, default_factory=None):

        self.default = default
        self.default_factory = default_factory


class Rep(Field):
    pass


class ValRep(Rep, typing.Generic[T]):

    def __init__(self, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]):

        super().__init__()
        self.vk = faiss.IndexIDMap2(vk)
        self.emb = emb
        self._df = pd.DataFrame()

    def add(self, id, val: T) -> 'ValIdx[T]':
        
        vb = self.emb(val)
        self.vk.add_with_ids(vb, id)
        self._df.loc[id, 0:len(vb)] = vb
        return ValIdx(
            val=val, id=id, idx=self
        )

    def get(self, id) -> np.ndarray:
        
        return self._df.loc[id].values

    def remove(self, id):
        
        self.vk.remove_ids([id])
        self._df = self._df.drop(index=[id])

    def like(
        self, val: typing.List[T], k: int=1, subset=None
    ) -> typing.List[int]:

        # TODO: limit to the subset
        x = self.emb(val)
        D, I = self.vk.search(x=x, k=k)
        return I


class TableRep(Rep):

    def __init__(self, vk: faiss.Index):

        self.vk = faiss.IndexIDMap2(vk)
        self._df = pd.DataFrame()

    def add(self, id, enc: np.ndarray) -> 'TableIdx':
        
        self.vk.add_with_ids(enc, id)
        self._df.loc[id, 0:len(enc)] = enc
        return ValIdx(
            id=id, idx=self
        )

    def get(self, id) -> np.ndarray:
        
        return self._df.loc[id].values

    def remove(self, id):
        
        self.vk.remove_ids([id])
        self._df = self._df.drop(index=[id])

    def like(
        self, enc: np.ndarray, k: int=1, subset=None
    ) -> typing.List[int]:

        # TODO: limit to the subset
        D, I = self.vk.search(x=enc, k=k)
        return I


class Idx(pydantic.BaseModel, ABC):

    id: int

    def __init__(self, rep: Rep, **data):
        self.rep = rep
        super().__init__(**data)


class ValIdx(Idx, typing.Generic[T]):

    val: T
    id: int

    def __init__(self, rep: ValRep[T], **data):
        self.rep = rep
        super().__init__(**data)


class TableIdx(Idx):

    id: int

    def __init__(self, rep: TableRep, **data):
        self.rep = rep
        super().__init__(**data)


class BaseConceptRep(pydantic.BaseModel):

    Manager: typing.ClassVar['ConceptManager'] = None
    Concept: typing.ClassVar[typing.Type['BaseConcept']] = None
    vk: ValIdx
    concept: 'BaseConcept'

    def __init__(self, **data):

        if type(data['concept']) != self.Concept:
            raise pydantic.ValidationError(
                'The concept passed in must be of '
                f'type {self.Concept} not {type(data["concept"])}'
            )
        super().__init__(**data)

    def __getattr__(self, key: str) -> typing.Any:

        if hasattr(self.concept, key):
            return getattr(self.concept, key)
        raise AttributeError(
            f'Concept {type(self)} has not attribute {key}')

    @conceptmethod
    def create(cls):
        
        cls.Manager.add_rep(cls)

    def save(self):
        self.Manager.save_or_add_rep_entry(self)

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
    def add_concept_entry(self, concept: BaseConcept) -> BaseConcept:
        pass

    @abstractmethod
    def concept_exists(self, concept: typing.Type[BaseConcept]) -> bool:
        pass

    @abstractmethod
    def add_concept(self, concept: typing.Type[BaseConcept]) -> BaseConcept:
        pass

    @abstractmethod
    def add_rep(self, rep: typing.Type[BaseConceptRep]) -> BaseConceptRep:
        pass

    @abstractmethod
    def save_or_add_concept_entry(self, concept: typing.Type[BaseConcept]):
        pass

    @abstractmethod
    def save_or_add_rep_entry(self, rep: typing.Type[BaseConceptRep]):
        pass

    @abstractmethod
    def get_concept_entry(self, concept: typing.Type[BaseConcept], pk: int) -> pd.Series:
        pass

    @abstractmethod
    def get_rep_entry(self, rep: typing.Type[BaseConceptRep], pk: int) -> pd.Series:
        pass


class DFConceptManager(ConceptManager):

    def __init__(self):

        Outer = self

        class Rep(BaseConceptRep):
            Manager: typing.ClassVar['ConceptManager'] = Outer
        
            @conceptmethod
            def create(self):
                if Outer.concept_exists(self):
                    raise RuntimeError
                Outer.add_rep(self)
            
        self.Rep = Rep
        class Concept(BaseConcept):
            Manager: typing.ClassVar['ConceptManager'] = Outer
            # Rep: typing.ClassVar[BaseConceptRep] = self.Rep
        
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
    
    def add_concept_entry(self, concept: BaseConcept) -> BaseConcept:
        
        pass
    
    def add_rep_entry(self, rep: BaseConceptRep) -> BaseConceptRep:
        pass

    def add_concept(self, concept: typing.Type[BaseConcept]) -> BaseConcept:
        schema = concept.schema()
        
        columns = {}
        for field_name, details in schema.get('properties').items():
            columns[field_name] = details.get('type')
        
        self._concepts[concept.__name__] = pd.DataFrame(
            columns=columns
        )

    def add_rep(self, rep: typing.Type[BaseConceptRep]) -> BaseConceptRep:
        schema = rep.schema()
        
        columns = {}

        for field_name, details in schema.get('properties').items():
            columns[field_name] = details.get('type')

        indices = []
        for name, value in rep.__annotations__.items():
            if isinstance(value, Idx):
                indices.append(name)
        
        self._indices[rep.__name__] = indices
        self._reps[rep.__name__] = pd.DataFrame(
            columns=columns
        )

    def save_or_add_concept_entry(self, concept: BaseConcept):

        concept_name = concept.__class__.__name__
        df = self._concepts[concept_name]
        df.loc[df.index.max() + 1] = concept.dict()
        return df.loc[df.index.max() + 1]

    def save_or_add_rep_entry(self, rep: BaseConceptRep):

        rep_name = rep.__class__.__name__
        df = self._reps[rep_name]
        
        # TODO: 
        df.loc[df.index.max() + 1] = rep.dict()
        return df.loc[df.index.max() + 1]

    def get_concept_entry(self, concept: typing.Type[BaseConcept], pk: int) -> pd.Series:

        row = self._concepts[concept.__name__].loc[pk]
        return concept(
            **row.to_dict()
        )

    def get_rep_entry(self, rep: typing.Type[BaseConceptRep], pk: int) -> pd.Series:

        row = self._reps[rep.__name__].loc[pk]
        return rep(
            **row.to_dict()
        )


# filter( Join(..., on=ColumnX, where=NestCol() == Col()))

# = FlatL2(n=64, .., ...)
# loop over all of the fields


# Loop over all the fields
# if it is a VectorRep
#  add the value
    
# Buyer(vk=.., )
    

# import pydantic

# class Y(pydantic.BaseModel):

#     x: str

#     def y(self):
#         return self.x

# class X:

#     def y(self):
#         return "x"

# class G(X, Y):
    
#     g: int


# g = G(g=3, x=2)

# print(g.y())
# print(g.g)
# print(g.x)