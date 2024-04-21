# 1st party
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields, MISSING, field
import typing
from typing_extensions import Self
from functools import wraps
import inspect
import math

# 3rd party
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from pydantic import Field



# TODO: Change this... Have the similarity
# contain all indices + a "chosen"
# if not "chosen" will be 0


# TODO: Add in Maximize/Minimize into similarity
#    These values must depend on the index used
#    This will affect adding and subtracting the similarities



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


def null_emb(x):
    return x


class Index(typing.Generic[T]):
    """Base class for Index. 
    """

    def __init__(
        self, col: str, 
        vk: faiss.Index, 
        emb: typing.Callable[[T], np.ndarray],
        maximize: bool=False
    ):

        super().__init__()
        self.col = col
        self.vk = faiss.IndexIDMap2(vk)
        self.emb = emb or null_emb
        self.maximize = maximize

    def add(self, id, val: T):
        vb = self.emb([val])
        self.vk.add_with_ids(vb, [id])
    
    def remove(self, id):
        id_selector = faiss.IDSelectorBatch(1, faiss.swig_ptr(np.array([id], dtype=np.int64)))
        self.vk.remove_ids(id_selector)

    def __len__(self) -> int:
        return self.vk.ntotal

    def max_values_per_index(self, values_2d, indices_2d):
        # Flatten the arrays
        indices = indices_2d.flatten()
        values = values_2d.flatten()

        # Get the unique indices and the inverse mapping
        unique_indices, inverse = np.unique(indices, return_inverse=True)

        # We will use a large negative value to ensure it doesn't interfere with the max calculation
        # Ensure that the max cannot naturally be this low; adjust as necessary.
        # The size of the output array is determined by the max index plus one
        large_negative_value = np.full(unique_indices.shape, -np.inf)

        # Use bincount to sum up the values placed at positions specified by 'inverse'
        # The weights are the actual values to aggregate
        # max_values = np.bincount(inverse, weights=values, minlength=len(unique_indices))

        # However, since bincount sums, we must ensure each position is initialized properly
        # First, we use np.maximum.at to place the max in an initialized array of large negatives
        np.maximum.at(large_negative_value, inverse, values)

        return large_negative_value, unique_indices

    def like(
        self, vals: typing.Union[T, typing.List[T]], k: int=1, subset=None
    ) -> 'Similarity':

        if self.vk.ntotal == 0:
            return Similarity()
        # TODO: limit to the subset
        x = self.emb(vals)
        D, I = self.vk.search(x=x, k=k)
        D, I = self.max_values_per_index(D, I)

        # If to maximize the distance take
        # the negative to get the "similarity"
        if self.maximize:
            D = -D
        return Similarity(D, I)
    
    # def F(cls, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]) -> RepFactory[Self]:

    @classmethod
    def F(cls, col: str, vk: typing.Type[faiss.Index], emb: typing.Callable[[T], np.ndarray], *idx_args, **idx_kwargs) -> 'IdxFactory[Index]':

        return IdxFactory[Self](
            cls, col, vk, emb, *idx_args, **idx_kwargs
        )

    @classmethod
    def field(cls, col: str, vk: typing.Type[faiss.Index], emb: typing.Callable[[T], np.ndarray], *idx_args, **idx_kwargs) -> 'IdxFactory[Index]':
        
        return field(
            default_factory=cls.F(col, vk, emb, *idx_args, **idx_kwargs)
        )

QR = typing.TypeVar('QR', bound=Index)


class IdxFactory(typing.Generic[QR]):

    def __init__(self, rep_cls: typing.Type[Index], col: str, f: typing.Callable[[str], QR], emb, *args, **kwargs):
        """Factory for creating Indexes to use in a concept

        Args:
            rep_cls (typing.Type[Index]): The class to create the index for
            col (str): The name of the column the index uses
            f (typing.Callable[[str], QR]): The function to create the actual index
            emb (_type_): The embedding function
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
class IdxMap(object):
    """Class to define the indexes that are stored
    """

    def __iter__(self) -> typing.Iterator[typing.Tuple[str, 'Index']]:

        for k, v in self.__dict__.items():

            if isinstance(v, Index):
                yield k, v


    def add(self, row: typing.Dict[str, typing.Any]):

        for k, index in self:
            value = row[index.col]
            index.add(
                row['id'], value
            )

    def __getitem__(self, key: str):
        
        value = getattr(self, key)
        if not isinstance(value, Index):
            raise AttributeError(f'RepMap has no attribute named {key}')
        return value


class Concept(BaseModel):

    __manager__: typing.ClassVar['ConceptManager'] = None

    id: int = None

    # y_idx: typing.Optional[np.array] = None

    @dataclass
    class __rep__(IdxMap):
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

    def query(self, df: pd.DataFrame, rep_map: IdxMap):
        return self.val


class BaseComp(object):

    @abstractmethod
    def query(self, df: pd.DataFrame, rep_map: IdxMap) -> pd.Series:
        pass

    def __call__(self, df: pd.DataFrame, rep_map: IdxMap) -> pd.DataFrame:
        """

        Args:
            df (pd.DataFrame): The dataframe to retrieve from
            rep_map (RepMap): The repmap to retrieve from

        Returns:
            pd.DataFrame: 
        """
        return df[self.query(df, rep_map)]

    def __xor__(self, other) -> 'BinComp':

        return BinComp(self, other, lambda lhs, rhs: lhs ^ rhs)

    def __and__(self, other):

        return BinComp(self, other, lambda lhs, rhs: lhs & rhs)

    def __or__(self, other):

        return BinComp(self, other, lambda lhs, rhs: lhs | rhs)
    
    def __invert__(self):

        return BinComp(None, self, lambda lhs, rhs: ~rhs)


class BinComp(BaseComp):

    def __init__(
        self, lhs: typing.Union['BinComp', 'Col', typing.Any], 
        rhs: typing.Union['BinComp', 'Col', typing.Any], 
        f: typing.Callable[[typing.Any, typing.Any], bool]
    ) -> None:
        """_summary_

        Args:
            lhs (typing.Union[BinComp;, Col;, typing.Any]): _description_
            rhs (typing.Union[BinComp;, &#39;Col&#39;, typing.Any]): _description_
            f (typing.Callable[[typing.Any, typing.Any], bool]): _description_
        """
        
        if not isinstance(lhs, BaseComp) and not isinstance(lhs, Col):
            lhs = Val(lhs)

        if not isinstance(rhs, BaseComp) and not isinstance(rhs, Col):
            rhs = Val(rhs)
        
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def query(self, df: pd.DataFrame, rep_map: IdxMap) -> pd.Series:
        """

        Args:
            df (pd.DataFrame): 

        Returns:
            The filter by comparison: 
        """
        print(type(self.lhs))
        lhs = self.lhs.query(df, rep_map)
        print('Result: ', type(lhs))
        rhs = self.rhs.query(df, rep_map)
        print('executing ', type(lhs), type(rhs))
        return self.f(lhs, rhs)


class BaseSim(ABC):

    @abstractmethod
    def query(self, df: pd.DataFrame, rep_map: IdxMap) -> 'Similarity':
        pass
    
    def __call__(self, df: pd.DataFrame, rep_map: IdxMap) -> 'Similarity':

        return self.query(df, rep_map)

    def __mul__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x * y
        )

    def __rmul__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x * y
        )        

    def __add__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x + y
        )

    def __radd__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x + y
        )
    
    def __sub__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: x - y
        )
    
    def __rsub__(self, other) -> Self:

        return AggSim(
            self, other, lambda x, y: y - x
        )
    # How to limit the "similarity"


class Sim(BaseSim):

    def __init__(self, name: str, val, k: int):
        """

        Args:
            name (str): 
            val (): 
            k (int): 
        """
        # val could also be a column
        # or a compare
        self.name = name
        self.val = val
        self.k = k

    def query(self, df: pd.DataFrame, rep_map: IdxMap) -> 'Similarity':
        """
        Args:
            rep_map (RepMap): 
            df (pd.DataFrame): 

        Returns:
            Similarity: 
        """
        indices = df.index.tolist()

        if isinstance(self.val, Col):
            val = self.val.query(df).index
        elif isinstance(self.val, typing.List):
            val = self.val
        else:
            val = [self.val]

        rep_idx = rep_map[self.name]
        
        return rep_idx.like(val, self.k, subset=indices)


class AggSim(BaseSim):

    def __init__(self, lhs, rhs, f: typing.Callable[[typing.Any, typing.Any], 'Similarity']):
        """Aggregate the similarity

        Args:
            lhs: The left hand side of the aggregation
            rhs: The right hand side of the aggregation
            f (typing.Callable[[typing.Any, typing.Any], Similarity;]): 
        """

        # val could also be a column
        # or a compare
        self.lhs = lhs
        self.rhs = rhs
        self.f = f

    def query(self, df: pd.DataFrame, idx_map: IdxMap) -> 'Similarity':
        """
        Args:
            idx_map (IdxMap): 
            df (pd.DataFrame): 

        Returns:
            Similarity: 
        """
        if isinstance(self.lhs, BaseSim):
            lhs = self.lhs.query(df, idx_map)
        else:
            lhs = self.lhs
        
        if isinstance(self.rhs, BaseSim):
            rhs = self.rhs.query(df, idx_map)
        else:
            rhs = self.rhs
        s = self.f(lhs, rhs)
        return s


class Like(BaseComp):

    def __init__(self, sim: BaseSim, k: int=None):
        """

        Args:
            sim (BaseSim): The similarities to use
        """
        self.sim = sim
        self.k = k

    def query(self, df: pd.DataFrame, rep_map: IdxMap) -> pd.Series:
        """

        Args:
            rep_map (RepMap): The RepMap for the concept
            df (pd.DataFrame): The DataFrame for the concept

        Returns:
            pd.Series: The 
        """
        similarity = self.sim(df, rep_map)

        # TODO: retrieve the top k

        similarity = similarity.topk(self.k)

        series = pd.Series(
            np.full((len(df.index)), False, np.bool_),
            df.index
        )
        print(similarity.indices)
        series[similarity.indices] = True
        return series

    def __call__(self, df: pd.DataFrame, rep_map: IdxMap) -> pd.DataFrame:
        
        
        return df[self.query(df, rep_map)]


@dataclass
class Similarity(object):

    value: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def align(self, other: 'Similarity') -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align the indices of the two similarities

        Args:
            other (Similarity): The other similarity to align with

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: The similarities 
                for the left and right hand sides and new indices
        """
        all_indices = np.union1d(self.indices, other.indices)

        # Hold all of the similarities
        new_sim_self = np.zeros_like(all_indices, dtype=self.value.dtype)
        new_sim_other = np.zeros_like(all_indices, dtype=other.value.dtype)

        # 
        self_pos = np.searchsorted(all_indices, self.indices)
        other_pos = np.searchsorted(all_indices, other.indices)

        new_sim_self[self_pos] = self.value
        new_sim_other[other_pos] = other.value
        print(all_indices)
        return new_sim_self, new_sim_other, all_indices
    
    def topk(self, k: int=None) -> Self:

        if k is None or k >= len(self.value):
            return self
        value = self.value * -1

        k = min(k, len(value))
        ind = np.argpartition(value, k, axis=0)
        ind = np.take(ind, np.arange(k), axis=0) # k non-sorted indices
        value = np.take_along_axis(value, ind, axis=0) # k non-sorted values

        # sort within k elements
        ind_part = np.argsort(value, axis=0)
        ind = np.take_along_axis(ind, ind_part, axis=0)
        value *= -1
        value = np.take_along_axis(value, ind_part, axis=0) 

        return Similarity(value, self.indices[ind])

    def _op(self, other, f) -> Self:
        """Execute an operation between two similarities

        Args:
            other: The value to update the similarity with
            f: The function to use on two similarities

        Returns:
            Self: The resulting similarity
        """
        if isinstance(other, Similarity):
            v_self, v_other, indices = self.align(other)
            return Similarity(f(v_self, v_other), indices)
        return Similarity(
            f(self.value, other), self.indices
        )
    
    def _comp_op(self, other, f) -> Self:
        """Filter the similarity based on a condition

        Args:
            other: The value to compare with
            f: The function to compare with

        Returns:
            Self: The updated similarity after having filtered
        """
        if isinstance(other, Similarity):
            v_self, v_other, indices = self.align(self, other)
            chosen = f(v_self, v_other)
            return Similarity(
                v_self[chosen], indices[chosen]
            )
        chosen = f(self.value, other)
        return Similarity(
            self.value[chosen], self.indices[chosen]
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The number of similarities
        """
        return self.value.shape[0]
    
    def __mul__(self, other) -> Self:
        """Multiply a value with the simlarity 
        Args:
            other: A similarity or a scalar value

        Returns:
            Self: The similarity multiplied with other
        """
        return self._op(
            other, lambda x, y: x * y
        )

    def __rmul__(self, other) -> Self:
        """Multiply a value with the simlarity 
        Args:
            other: A similarity or a scalar value

        Returns:
            Self: The similarity multiplied with other
        """
        return self._op(
            other, lambda x, y: x * y
        )

    def __add__(self, other) -> Self:
        """Add a value to the simlarity 

        Args:
            other: A similarity or a scalar value

        Returns:
            Self: The similarity multiplied with other
        """
        return self._op(
            other, lambda x, y: x + y
        )

    def __radd__(self, other) -> Self:
        """Add a value to the simlarity 

        Args:
            other: A similarity or a scalar value

        Returns:
            Self: The similarity multiplied with other
        """
        return self._op(
            other, lambda x, y: x + y
        )

    def __sub__(self, other) -> Self:
        """Subtract a value from the simlarity 

        Args:
            other: A similarity or a scalar value

        Returns:
            Self: The similarity multiplied with other
        """
        return self._op(
            other, lambda x, y: x - y
        )

    def __rsub__(self, other) -> Self:
        """Subtract a value from the simlarity 

        Args:
            other: A similarity or a scalar value

        Returns:
            Self: The similarity multiplied with other
        """
        return self._op(
            other, lambda x, y: y - x
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
            Comp: The comparison for less than
        """

        return BinComp(self, other, lambda lhs, rhs: lhs < rhs)

    def __le__(self, other) -> 'BinComp':
        """Check whether column is less than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for less than or equal
        """

        return BinComp(self, other, lambda lhs, rhs: lhs <= rhs)

    def __gt__(self, other) -> 'BinComp':
        """Check whether column is greater than another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than
        """
        return BinComp(self, other, lambda lhs, rhs: lhs > rhs)

    def __ge__(self, other) -> 'BinComp':
        """Check whether column is greater than or equal to another value

        Args:
            other: The value to compare against

        Returns:
            Comp: The comparison for greater than or equal to
        """
        return BinComp(self, other, lambda lhs, rhs: lhs >= rhs)

    def query(self, df: pd.DataFrame, rep_map: IdxMap) -> typing.Any:
        """Retrieve the column

        Args:
            df (pd.DataFrame): The DataFrame to retrieve from

        Returns:
            typing.Any: The 
        """
        return df[self.name]


C = typing.TypeVar('C', bound=Concept)

# Think about how to handle this
class ConceptQuery(typing.Generic[C]):

    def __init__(self, concept_cls: typing.Type[C], comp: BaseComp):
        """

        Args:
            concept_cls (typing.Type[C]): 
            comp (BaseComp): 
        """
        self.concept = concept_cls
        self.comp = comp

    def filter(self, comp: BinComp) -> Self:

        return ConceptQuery[C](
            self.comp & comp
        )
    
    def __iter__(self) -> typing.Iterator[C]:

        concept = self.concept.__manager__.get_data(self.concept)
        idx = self.concept.__manager__.get_rep(self.concept)

        sub_df = self.comp(
            concept, idx
        )

        for _, row in sub_df.iterrows():
            yield self.concept(**row.to_dict())
        
    def df(self) -> pd.DataFrame:

        concept = self.concept.__manager__.get_data(self.concept)
        idx = self.concept.__manager__.get_rep(self.concept)
        return self.comp(
            concept, idx
        )


class ConceptManager(object):

    def __init__(self):

        self._concepts = {}
        self._field_reps: typing.Dict[str, IdxMap] = {}
        self._concept_reps: typing.Dict[str, typing.List] = {}
        self._ids = {}

    def add_concept(self, concept: typing.Type[Concept]):

        columns, dtypes, _, _ = concept.columns(dtypes=True)

        df = pd.DataFrame(
            columns=columns
        )
        df = df.astype(dict(zip(columns, dtypes)))

        self._concepts[concept.model_name()] = df
        self._field_reps[concept.model_name()] = concept.__rep__()
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
        self._field_reps[rep.rep_name()] = rep.__rep__()

    def get_rep(self, concept: typing.Type[Concept]) -> pd.DataFrame:

        return self._field_reps[concept.model_name()]

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
        
        rep = self._field_reps[concept.model_name()]
        df.loc[concept.id] = concept.to_dict()
        rep.add(concept.to_dict())

        # self._concepts[concept.model_name()] = df


concept_manager = ConceptManager()
