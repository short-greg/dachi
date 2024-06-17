# 1st party
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields, MISSING, field
import typing
from typing_extensions import Self
from functools import wraps
import inspect
import math
from dataclasses import InitVar

# 3rd party
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from pydantic import Field


T = typing.TypeVar('T')


def null_emb(x):
    return x


class RepLookup(typing.Generic[T]):
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
        self._base_vk = vk
        self._vk = faiss.IndexIDMap2(vk)
        self.emb = emb or null_emb
        self.maximize = maximize

    def add(self, id, val: T):
        vb = self.emb([val])
        self._vk.add_with_ids(vb, [id])
    
    def remove(self, id):
        id_selector = faiss.IDSelectorBatch(1, faiss.swig_ptr(np.array([id], dtype=np.int64)))
        self._vk.remove_ids(id_selector)

    def __len__(self) -> int:
        return self._vk.ntotal

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
    ) -> 'Sim':

        if self._vk.ntotal == 0:
            return Sim()
        # TODO: limit to the subset
        x = self.emb(vals)
        D, I = self._vk.search(x=x, k=k)
        D, I = self.max_values_per_index(D, I)

        # If to maximize the distance take
        # the negative to get the "similarity"
        if self.maximize:
            D = -D
        return Sim(D, I)
    
    # def F(cls, vk: faiss.Index, emb: typing.Callable[[T], np.ndarray]) -> RepFactory[Self]:

    @classmethod
    def F(cls, col: str, vk: typing.Type[faiss.Index], emb: typing.Callable[[T], np.ndarray], *idx_args, **idx_kwargs) -> 'RepFactory[RepLookup]':

        return RepFactory[Self](
            cls, col, vk, emb, *idx_args, **idx_kwargs
        )

    @classmethod
    def field(cls, col: str, vk: typing.Type[faiss.Index], emb: typing.Callable[[T], np.ndarray], *idx_args, **idx_kwargs) -> 'RepFactory[RepLookup]':
        
        return field(
            default_factory=cls.F(col, vk, emb, *idx_args, **idx_kwargs)
        )


QR = typing.TypeVar('QR', bound=RepLookup)


@dataclass
class Sim(object):

    value: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def align(self, other: 'Sim') -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        return Sim(value, self.indices[ind])

    def _op(self, other, f) -> Self:
        """Execute an operation between two similarities

        Args:
            other: The value to update the similarity with
            f: The function to use on two similarities

        Returns:
            Self: The resulting similarity
        """
        if isinstance(other, Sim):
            v_self, v_other, indices = self.align(other)
            return Sim(f(v_self, v_other), indices)
        return Sim(
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
        if isinstance(other, Sim):
            v_self, v_other, indices = self.align(self, other)
            chosen = f(v_self, v_other)
            return Sim(
                v_self[chosen], indices[chosen]
            )
        chosen = f(self.value, other)
        return Sim(
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


class RepFactory(typing.Generic[QR]):

    def __init__(self, rep_cls: typing.Type[RepLookup], col: str, f: typing.Callable[[str], QR], emb, *args, **kwargs):
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
        
    def __call__(self) -> RepLookup:
        """

        Returns:
            V: 
        """
        return self.rep_cls(
            self.col, self.f(*self.args, **self.kwargs), self.emb
        )


@dataclass
class Rep(object):
    """Class to define the indexes that are stored
    """
    base_rep: InitVar['Rep'] = None

    def __post_init__(self, base_rep: 'Rep'=None):

        self.__base_rep__ = base_rep

    def __iter__(self) -> typing.Iterator[typing.Tuple[str, 'RepLookup']]:

        for k, v in self.__dict__.items():

            if isinstance(v, RepLookup):
                yield k, v

        if self.__base_rep__ is not None:
            for k, v in self.__base_rep__:
                if k not in self.__dict__:
                    yield k, v

    def __contains__(self, k: str):

        return (
            k in self.__dict__ 
        ) or (
            self.__base_rep__ is not None and
            k in self.__base_rep__
        )

    def add(self, row: typing.Dict[str, typing.Any]):

        for k, index in self:
            value = row[index.col]
            index.add(
                row['id'], value
            )

    def drop(self, id):

        for k, index in self:
            index.remove(id)

    def __getattr__(self, key: str):

        if self.__base_rep__ is not None:
            return getattr(self.__base_rep__, key)
        raise AttributeError(f'There is no attribute named {key} in IndexMap {str(self)}')

    def __getitem__(self, key: typing.Union[str, typing.Series, np.ndarray]):
        
        if isinstance(key, pd.Series):
            key = key.values

        value = getattr(self, key)
        if not isinstance(value, RepLookup):
            raise AttributeError(f'RepMap has no attribute named {key}')
        return value


# This needs to limit those that are chosen too

class DerivedRep(object):
    """Class to define the indexes that are stored
    """

    def __init__(
        self, base_rep: typing.Union['Rep', 'DerivedRep'], 
        alias_map: typing.Dict[str, str], ids: np.ndarray
    ):

        self._rep = base_rep
        self._alias_map = alias_map
        self._rev_alias_map = {alias: k for k, alias in alias_map.items()}
        self._ids = ids

    def __iter__(self) -> typing.Iterator[typing.Tuple[str, 'RepLookup']]:

        for k, v in self._rep:
            if k in self._alias_map:
                return self._alias_map[k], v

    def __contains__(self, k: str):

        return k in self._rev_alias_map

    def __getattr__(self, key: str):

        return getattr(self._rep, key)

    def __getitem__(self, key: typing.Union[str, typing.Series, np.ndarray]):
        
        if isinstance(key, pd.Series):
            key = key.values

        # TODO: Need to limit!

        value = getattr(self, key)
        if not isinstance(value, RepLookup):
            raise AttributeError(f'RepMap has no attribute named {key}')
        return value

