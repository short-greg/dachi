# from ._struct import BaseStruct
from pydantic import BaseModel
import pandas as pd
import typing
from functools import wraps
from abc import abstractmethod
import faiss
import numpy as np

from typing_extensions import Self
# TODO: ad in a base class

T = typing.TypeVar('T')


def conceptmethod(func: typing.Callable[..., T]) -> typing.Callable[..., T]:

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if not isinstance(cls, type):
            raise TypeError("This method can only be called on the class, not on instances.")
        return func(cls, *args, **kwargs)

    return classmethod(wrapper)


def abstractconceptmethod(func: typing.Callable[..., T]) -> typing.Callable[..., T]:
    func = abstractmethod(func)

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        if not isinstance(cls, type):
            raise TypeError("This method can only be called on the class, not on instances.")
        return func(cls, *args, **kwargs)

    return classmethod(wrapper)


class RepFactory(typing.Generic[T]):

    def __init__(self, rep: typing.Type[T], *args, **kwargs):
        """

        Args:
            rep (typing.Type[V]): _description_
        """
        self.rep = rep
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self) -> T:
        """

        Returns:
            V: 
        """
        return self.rep


class RepIdx(typing.Generic[T]):

    def __init__(self, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]):

        super().__init__()
        self.vk = faiss.IndexIDMap2(vk)
        self.emb = emb

    def add(self, id, val: T):
        
        vb = self.emb(val)
        self.vk.add_with_ids(vb, id)
    
    def columns(self, names: typing.List[str]=None):
        
        if names is None:
            return self._columns

    def remove(self, id):
        
        self.vk.remove_ids([id])

    def like(
        self, val: typing.List[T], k: int=1, subset=None
    ) -> typing.List[int]:

        # TODO: limit to the subset
        x = self.emb(val)
        D, I = self.vk.search(x=x, k=k)
        return I
    
    # def F(cls, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]) -> RepFactory[Self]:

    @classmethod
    def F(cls, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]) -> RepFactory:

        return RepFactory[Self](
            vk, emb=emb
        )


class Concept(BaseModel):

    manager: typing.ClassVar['ConceptManager'] = None

    id: int = None

    # y_idx: typing.Optional[np.array] = None

    class __rep__:
        
        @classmethod
        def fields(cls):
            
            for key, val in cls.__dict__.items():
                if isinstance(val, RepFactory):
                    yield key, val

    @classmethod
    def columns(cls, dtypes: bool=False) -> typing.List[str]:
        schema = cls.schema()

        print(
            schema['properties'].values()
        )

        columns = list(schema['properties'].keys())
        if dtypes:
            types = [cls.__annotations__[c] if c in cls.__annotations__ else None for c in columns]

            return columns, types
        
        return columns

    @conceptmethod
    def reps(cls) -> typing.List[str]:
        
        return [key for key, _ in cls.__rep__]

    @conceptmethod
    def build(cls):
        cls.manager.add_concept(cls)

    @conceptmethod
    def get(cls, id) -> 'Concept':

        df = cls.manager.get(
            cls
        )
        return cls(
            df.loc[id][cls.columns()]
        )

    def save(self):
        self.manager.add_row(self)

    @conceptmethod
    def to_df(cls) -> pd.DataFrame:
        return cls.manager.get_data(
            cls, cls.columns()
        )

    def to_dict(self) -> typing.Dict:
        return self.dict()

    def to_series(self) -> pd.DataFrame:
        return pd.Series(self.to_dict())

    @conceptmethod
    def filter(cls, comp: 'Comp'):
        return cls.manager.subset(cls, comp)

    @conceptmethod
    def exclude(cls, comp: 'Comp'):
        return cls.manager.inverse_subset(cls, comp)

    @classmethod
    def model_name(cls):

        return cls.__name__


class Val(object):

    def __init__(self, val) -> None:
        self.val = val

    def __call__(self, df: pd.DataFrame):
        return self.val


class Comp(object):

    def __init__(self, lhs: typing.Union['Comp', 'Col', typing.Any], rhs: typing.Union['Comp', 'Col', typing.Any], f: typing.Callable[[typing.Any, typing.Any], bool]) -> None:
        
        if not isinstance(lhs, Comp) and not isinstance(lhs, Col):
            lhs = Val(lhs)

        if not isinstance(rhs, Comp) and not isinstance(rhs, Col):
            rhs = Val(rhs)
        
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def __call__(self, df: pd.DataFrame) -> typing.Any:

        lhs = self.lhs(df)
        rhs = self.rhs(df)
        return self.f(lhs, rhs)
    
    def __eq__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs == rhs)
    
    def __lt__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs >= rhs)

    def __xor__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs ^ rhs)

    def __and__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs & rhs)

    def __or__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs | rhs)


class Col(object):

    def __init__(self, name: str):

        self.name = name
    
    def __eq__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs == rhs)
    
    def __lt__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs >= rhs)

    def __xor__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs ^ rhs)

    def __and__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs & rhs)

    def __or__(self, other):

        return Comp(self, other, lambda lhs, rhs: lhs | rhs)

    def __call__(self, df: pd.DataFrame) -> typing.Any:

        return df[self.name]


class ConceptQuery(object):

    def __init__(self, concept_cls: Concept, comp: Comp):

        self.concept = concept_cls
        self.comp = comp

    def __call__(self) -> pd.DataFrame:

        df = self.concept.manager.get_data(self.concept)
        return df[self.comp(df)]


class ConceptManager(object):

    def __init__(self):

        self._concepts = {}
        self._field_indices: typing.Dict[str, typing.List] = {}
        self._concept_reps: typing.Dict[str, typing.List] = {}
        self._ids = {}

    def add_concept(self, concept: typing.Type[Concept]):

        columns, dtypes = concept.columns(dtypes=True)

        df = pd.DataFrame(
            columns=columns
        )
        df = df.astype(dict(zip(columns, dtypes)))

        indices = {key: ind() for key, ind in concept.__rep__.fields()}
        self._concepts[concept.model_name()] = df
        self._field_indices[concept.model_name()] = indices
        self._ids[concept.model_name()] = 0

    def get_data(self, concept: typing.Type[Concept]):

        return self._concepts[concept.model_name()]
    
    def add_row(self, concept: Concept):

        try:
            df = self._concepts[concept.model_name()]
        except KeyError:
            raise KeyError(
                f'No concept named {concept.model_name()}. '
                'Has it been built with Concept.build()?')
        if concept.id is None:
            concept.id = self._ids[concept.model_name()]

            # Find a better way to handle this
            self._ids[concept.model_name()] += 1
        
        df.loc[concept.id] = concept.to_dict()
        # self._concepts[concept.model_name()] = df

concept_manager = ConceptManager()


# class RepBase(object):
#     pass

#     @abstractclassmethod
#     def F(cls, *args, **kwargs) -> 'RepFactory':
#         pass


# V = typing.TypeVar('V', bound=RepBase)


# class A(BaseModel):
#     _value: str = PrivateAttr(default="Initial value")
#     id: int

#     @property
#     def value(self):
#         return self._value

#     def clone(self):
#         obj = self.copy(update={"id": self.id + 1})
#         obj._value = "Updated value"
#         return obj

# if __name__ == '__main__':
#     a1 = A(id=1)
#     a2 = a1.clone()
#     print([a1, a2])
#     print(a1.value)
#     print(a2.value)

# rep_set
# rep_ref
# rep_ref
# 

# concept = Concept()
# 1) add all of the base fields
# 2) add the reference to the rep - RepRef 
# 3) self._rep_ref()
# 4) If not passed in will retrieve the reference
#  ... Does not add the vector to the index unless
#  .. saved


# class Rep(object):
#     pass


# class ValRep(Rep, typing.Generic[T]):

#     def __init__(self, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]):

#         super().__init__()
#         self.vk = faiss.IndexIDMap2(vk)
#         self.emb = emb
#         self._columns: typing.List[str] = ['val', 'idx']

#     def add(self, id, val: T):
        
#         vb = self.emb(val)
#         self.vk.add_with_ids(vb, id)
#         self._df.loc[id, 0:len(vb)] = vb
    
#     def columns(self, names: typing.List[str]=None):
        
#         if names is None:
#             return self._columns

#     def remove(self, id):
        
#         self.vk.remove_ids([id])

#     def like(
#         self, val: typing.List[T], k: int=1, subset=None
#     ) -> typing.List[int]:

#         # TODO: limit to the subset
#         x = self.emb(val)
#         D, I = self.vk.search(x=x, k=k)
#         return I


# class TableRep(Rep):

#     def __init__(self, vk: faiss.Index):

#         self.vk = faiss.IndexIDMap2(vk)
#         self._df = pd.DataFrame()

#     def add(self, id, enc: np.ndarray) -> 'TableIdx':
        
#         self.vk.add_with_ids(enc, id)
#         self._df.loc[id, 0:len(enc)] = enc
#         return ValIdx(
#             id=id, idx=self
#         )

#     def get(self, id) -> np.ndarray:
        
#         return self._df.loc[id].values

#     def remove(self, id):
        
#         self.vk.remove_ids([id])
#         self._df = self._df.drop(index=[id])

#     def like(
#         self, enc: np.ndarray, k: int=1, subset=None
#     ) -> typing.List[int]:

#         # TODO: limit to the subset
#         D, I = self.vk.search(x=enc, k=k)
#         return I


# class Idx(pydantic.BaseModel, ABC):

#     id: int

#     def __init__(self, rep: Rep, **data):
#         self.rep = rep
#         super().__init__(**data)


# class ValIdx(Idx, typing.Generic[T]):

#     val: T
#     id: int

#     def __init__(self, rep: ValRep[T], **data):
#         self.rep = rep
#         super().__init__(**data)


# class TableIdx(Idx):

#     id: int

#     def __init__(self, rep: TableRep, **data):
#         self.rep = rep
#         super().__init__(**data)

