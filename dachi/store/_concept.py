# from ._struct import BaseStruct
from pydantic import BaseModel
import pandas as pd
import typing
from functools import wraps
from abc import abstractmethod
import faiss
import numpy as np
import inspect
from dataclasses import dataclass, fields, MISSING, field
from pydantic import Field
from typing_extensions import Self
from abc import ABC
# TODO: ad in a base class
import pydantic

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


def null_emb(x):
    return x


class RepIdx(typing.Generic[T]):

    def __init__(
        self, col: str, 
        vk: faiss.Index, 
        emb: typing.Callable[[T], np.ndarray]
    ):

        super().__init__()
        self.col = col
        self.vk = faiss.IndexIDMap2(vk)
        self.emb = emb or null_emb

    def add(self, id, val: T):
        vb = self.emb([val])
        self.vk.add_with_ids(vb, [id])
    
    def remove(self, id):
        id_selector = faiss.IDSelectorBatch(1, faiss.swig_ptr(np.array([id], dtype=np.int64)))
        self.vk.remove_ids(id_selector)

    def __len__(self) -> int:
        return self.vk.ntotal

    def like(
        self, vals: typing.Union[T, typing.List[T]], k: int=1, subset=None
    ) -> 'Similarity':

        # TODO: limit to the subset
        x = self.emb(vals)
        D, I = self.vk.search(x=x, k=k)
        return Similarity(D, I)
    
    # def F(cls, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]) -> RepFactory[Self]:

    @classmethod
    def F(cls, col: str, vk: typing.Type[faiss.Index], emb: typing.Callable[[T], np.ndarray], *idx_args, **idx_kwargs) -> 'RepFactory[RepIdx]':

        return RepFactory[Self](
            cls, col, vk, emb, *idx_args, **idx_kwargs
        )

    @classmethod
    def field(cls, col: str, vk: typing.Type[faiss.Index], emb: typing.Callable[[T], np.ndarray], *idx_args, **idx_kwargs) -> 'RepFactory[RepIdx]':
        
        return field(
            default_factory=cls.F(col, vk, emb, *idx_args, **idx_kwargs)
        )

QR = typing.TypeVar('QR', bound=RepIdx)

class RepFactory(typing.Generic[QR]):

    def __init__(self, rep_cls: typing.Type[RepIdx], col: str, f: typing.Callable[[str], QR], emb, *args, **kwargs):
        """
        Args:
            rep (typing.Type[V]): _description_
        """
        self.rep_cls = rep_cls
        self.col = col
        self.f = f
        self.emb = emb
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self) -> T:
        """

        Returns:
            V: 
        """
        return self.rep_cls(
            self.col, self.f(*self.args, **self.kwargs), self.emb
        )

@dataclass
class RepMap(object):

    def __getitem__(self, key: str):
        
        value = getattr(self, key)
        if not isinstance(value, RepIdx):
            raise AttributeError(f'RepMap has no attribute named {key}')
        return value


class Concept(BaseModel):

    __manager__: typing.ClassVar['ConceptManager'] = None

    id: int = None

    # y_idx: typing.Optional[np.array] = None

    @dataclass
    class __rep__(RepMap):
        pass

    @classmethod
    def columns(cls, dtypes: bool=False) -> typing.List[str]:
        schema = cls.model_json_schema()

        columns = list(schema['properties'].keys())
        if dtypes:
            defaults, default_factories = [], []
            for name, field in cls.model_fields.items():
                defaults.append(field.default)
                default_factories.append(field.default_factory)
            types = [cls.__annotations__[c] if c in cls.__annotations__ else None for c in columns]

            return columns, types, defaults, default_factories
        
        return columns

    @classmethod
    def manager(cls) -> 'ConceptManager':
        return cls.__manager__

    @conceptmethod
    def reps(cls) -> typing.List[str]:
        
        return [key for key, _ in cls.__rep__]

    @conceptmethod
    def build(cls):
        cls.__manager__.add_concept(cls)

    @conceptmethod
    def get(cls, id) -> 'Concept':

        df = cls.__manager__.get(
            cls
        )
        return cls(
            df.loc[id][cls.columns()]
        )

    def save(self):
        self.__manager__.add_row(self)

    @conceptmethod
    def to_df(cls) -> pd.DataFrame:
        return cls.__manager__.get_data(
            cls, cls.columns()
        )

    def to_dict(self) -> typing.Dict:
        return self.model_dump()

    def to_series(self) -> pd.DataFrame:
        return pd.Series(self.to_dict())

    @conceptmethod
    def filter(cls, comp: 'BinComp'):
        return ConceptQuery(
            cls, comp
        )

    @conceptmethod
    def exclude(cls, comp: 'BinComp'):
        return ConceptQuery(
            cls, ~comp
        )

    @classmethod
    def model_name(cls):
        
        return cls.__name__
    
    @classmethod
    def ctype(cls) -> typing.Type['Concept']:
        return cls


class RepMixin(object):

    @conceptmethod
    def like(cls, comp: 'BinComp'):
        return ConceptQuery(
            cls, comp
        )

    @conceptmethod
    def build(cls):
        cls.manager().add_rep(cls)

    @classmethod
    def model_name(cls) -> str:
        concept_cls = None
        for base in cls.__bases__:
            if inspect.isclass(base) and issubclass(base, Concept):
                concept_cls = base
                return concept_cls.model_name()
        return None

    @classmethod
    def rep_name(cls):
        
        return cls.__name__


class Val(object):

    def __init__(self, val) -> None:
        self.val = val

    def query(self, df: pd.DataFrame):
        return self.val


class BaseComp(object):

    def query(self, df: pd.DataFrame, rep_map: RepMap) -> pd.Series:
        pass

    def __call__(self, df: pd.DataFrame, rep_map: RepMap) -> pd.DataFrame:
        pass


class BinComp(object):

    def __init__(
        self, lhs: typing.Union['BinComp', 'Col', typing.Any], 
        rhs: typing.Union['BinComp', 'Col', typing.Any], 
        f: typing.Callable[[typing.Any, typing.Any], bool]
    ) -> None:
        
        if not isinstance(lhs, BinComp) and not isinstance(lhs, Col):
            lhs = Val(lhs)

        if not isinstance(rhs, BinComp) and not isinstance(rhs, Col):
            rhs = Val(rhs)
        
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def query(self, df: pd.DataFrame) -> pd.Series:
        """

        Args:
            df (pd.DataFrame): 

        Returns:
            The filter by comparison: 
        """
        lhs = self.lhs.query(df)
        rhs = self.rhs.query(df)
        return self.f(lhs, rhs)

    def __call__(self, df: pd.DataFrame) -> typing.Any:

        return df[self.query(df)]
    
    def __xor__(self, other):

        return BinComp(self, other, lambda lhs, rhs: lhs ^ rhs)

    def __and__(self, other):

        return BinComp(self, other, lambda lhs, rhs: lhs & rhs)

    def __or__(self, other):

        return BinComp(self, other, lambda lhs, rhs: lhs | rhs)
    
    def __invert__(self):

        return BinComp(None, self, lambda lhs, rhs: ~rhs)


# class Rep(object):

#     def __init__(self, name: str):
#         """Create a reference to a representation in the Concept

#         Args:
#             name (str): The name of the representation
#         """
#         self.name = name
    
#     def get(self, rep_map: RepMap) -> RepIdx:
#         """Retrieve the RepIdx from the RepMap

#         Args:
#             rep_map (RepMap): The RepMap to retrieve from

#         Returns:
#             RepIdx: The RepIdx 
#         """
#         return rep_map[self.name]


class BaseSim(ABC):

    @abstractmethod
    def query(self, rep_map: RepMap, df: pd.DataFrame) -> pd.Series:
        pass

    def __call__(self, rep_map: RepMap, df: pd.DataFrame) -> pd.DataFrame:

        similarity = self.query(rep_map, df)
        series = pd.Series(
            np.full((len(df.index)), False, np.bool_),
            similarity.indices
        )
        series[similarity.indices] = True

        return df.loc[self.query(rep_map, df)]

    def __mul__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x * y
        )
        
    def __add__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x + y
        )

    def __sub__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x - y
        )

    # How to limit the "similarity"

class Sim(BaseSim):

    def __init__(self, name: str, val, k: int):

        # val could also be a column
        # or a compare
        self.name = name
        self.val = val
        self.k = k

    def query(self, rep_map: RepMap, df: pd.DataFrame) -> 'Similarity':

        indices = df.index.tolist()

        if isinstance(self.val, Col):
            val = self.val.query(df).index
        elif isinstance(self.val, typing.List):
            val = self.val
        else:
            val = [self.val]
        rep_idx = self.rep.get(rep_map)
        
        return rep_idx.like(val, self.k, subset=indices)
        # How to get the rep index subset
        # return series
        # series = pd.Series(
        #     np.full((len(indices)), False, np.bool_),
        #     indices
        # )
        # series[rep_idx.like(val, self.k, subset=indices)] = True
        # # How to get the rep index subset


class AggSim(BaseSim):

    def __init__(self, lhs, rhs, f: typing.Callable[[typing.Any, typing.Any], 'Similarity']):

        # val could also be a column
        # or a compare
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def query(self, rep_map: RepMap, df: pd.DataFrame) -> 'Similarity':

        if isinstance(self.lhs, BaseSim):
            lhs = self.lhs.query(rep_map, df)
        else:
            lhs = self.lhs
        
        if isinstance(self.rhs, BaseSim):
            rhs = self.rhs.query((rep_map, df))
        else:
            rhs = self.rhs
        return self.f(lhs, rhs)

# TODO: Change this... Have the similarity
# contain all indices + a "chosen"
# if not "chosen" will be 0


@dataclass
class Similarity(object):

    value: np.ndarray
    indices: np.ndarray

    def align(self, other: 'Similarity'):
        all_indices = np.union1d(self.indices, other.indices)

        # Hold all of the similarities
        new_sim_self = np.zeros_like(all_indices, dtype=self.value.dtype)
        new_sim_other = np.zeros_like(all_indices, dtype=other.value.dtype)

        # 
        self_pos = np.searchsorted(all_indices, self.indices)
        other_pos = np.searchsorted(all_indices, other.indices)

        new_sim_self[self_pos] = self.value
        new_sim_other[other_pos] = other.value
        return new_sim_self, new_sim_other, all_indices
    
    def _op(self, other, f) -> Self:

        if isinstance(other, Similarity):
            v_self, v_other, indices = self.align(self, other)
            return Similarity(f(v_self, v_other), indices)
        return Similarity(
            f(self.value, other), indices
        )
    
    def _comp_op(self, other, f) -> Self:

        if isinstance(other, Similarity):
            v_self, v_other, indices = self.align(self, other)
            result = f(v_self, v_other)
        else:
            indices = self.indices
            result = f(self.value, other)
        return Similarity(
            result, indices
        )

    def __len__(self) -> int:
        return self.value.shape[0]
    
    def __mul__(self, other) -> Self:

        return self._op(
            other, lambda x, y: x * y
        )
        
    def __add__(self, other) -> Self:

        return self._op(
            other, lambda x, y: x + y
        )

    def __sub__(self, other) -> Self:

        return self._op(
            other, lambda x, y: x - y
        )
    
    # TODO: Add more such as "less than", max etc

# like( )  <= 
# Sim() <= I want this to return numerical values

# This makes it a comparison
# Sim() returns a "Similarity"
# like(Sim() + Sim(), N=10))

# # I want to make it something like this
# like(0.5 * Sim('R', Comp) + 0.5 Sim()

# Sim() + 


class Col(object):
    """A column in the model
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): Name of the column
        """
        self.name = name
    
    def __eq__(self, other) -> 'BinComp':
        """Check the eqquality of two columns

        Args:
            other (_type_): The other Col com compare with

        Returns:
            Comp: The comparison for equality
        """

        return BinComp(self, other, lambda lhs, rhs: lhs == rhs)
    
    def __lt__(self, other) -> 'BinComp':
        """Check whether column is less than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for equality
        """

        return BinComp(self, other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other) -> 'BinComp':

        return BinComp(self, other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other) -> 'BinComp':

        return BinComp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other) -> 'BinComp':

        return BinComp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other) -> 'BinComp':

        return BinComp(self, other, lambda lhs, rhs: lhs >= rhs)

    def query(self, df: pd.DataFrame) -> typing.Any:

        return df[self.name]


C = typing.TypeVar('C', bound=Concept)

# Think about how to handle this
class ConceptQuery(typing.Generic[C]):

    def __init__(self, concept_cls: typing.Type[C], comp: BinComp):

        self.concept = concept_cls
        self.comp = comp

    def filter(self, comp: BinComp) -> Self:

        return ConceptQuery[C](
            self.comp & comp
        )
    
    def __iter__(self) -> typing.Iterator[C]:

        sub_df = self.comp(self.concept.__manager__.get_data(self.concept))

        for _, row in sub_df.iterrows():
            yield self.concept(**row.to_dict())


class ConceptManager(object):

    def __init__(self):

        self._concepts = {}
        self._field_indices: typing.Dict[str, typing.List] = {}
        self._concept_reps: typing.Dict[str, typing.List] = {}
        self._ids = {}

    def add_concept(self, concept: typing.Type[Concept]):

        columns, dtypes, _, _ = concept.columns(dtypes=True)

        df = pd.DataFrame(
            columns=columns
        )
        df = df.astype(dict(zip(columns, dtypes)))

        self._concepts[concept.model_name()] = df
        self._field_indices[concept.model_name()] = concept.__rep__()
        self._ids[concept.model_name()] = 0

    def add_rep(self, rep: typing.Type[RepMixin]):

        if not issubclass(rep, Concept):
            raise ValueError('Cannot build Rep unless mixed with a concept.')
        columns, dtypes, defaults, default_factories = rep.columns(True)
        df = self.get_data(rep.model_name())

        for c, dtype, default, default_factory, in zip(columns, dtypes, defaults, default_factories):
            if default_factory is not None and default is None:
                df[c] = default_factory()
            elif default is not None:
                df[c] = default
            else:
                df[c] = None
        # add columns for the representation
        df = df.astype(dict(zip(columns, dtypes)))
        self._concepts[rep.model_name()] = df
        self._field_indices[rep.rep_name()] = rep.__rep__()

    def get_data(self, concept: typing.Type[Concept]) -> pd.DataFrame:

        return self._concepts[concept.model_name()][concept.columns()]
    
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





# class Rep(object):

#     def __init__(self, name: str):

#         self.name = name

#     def query(self, df: pd.DataFrame) -> typing.Any:

#         return df[self.name]

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


# Sim should be a type of comparison?
# Boolean comp
# Similarity Comp

