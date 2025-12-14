# 1st party
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Callable, Dict
import typing as t
from abc import ABC, abstractmethod

# 3rd party
import pydantic
from pydantic import Field

#  Local
from ._base import CORE_TYPE


class ShareableItem(pydantic.BaseModel, t.Generic[CORE_TYPE]):
    """Serializable leaf object with a ``data`` field."""

    data: CORE_TYPE | None
    _callbacks: t.List[
        Callable[[CORE_TYPE | None, CORE_TYPE | None], None]
    ] = pydantic.PrivateAttr(default_factory=list)

    def get(self) -> CORE_TYPE | None:
        return self.data

    def set(self, value: CORE_TYPE | None):
        
        old_val = self.data
        super().__setattr__('data', value)
        self.update_data_hook(old_val, value)
        return value

    def __setattr__(self, name, value):
        
        if name == "data":
            return self.set(value)
        return super().__setattr__(name, value)

    def empty(self) -> bool:

        return self.data is None

    def update_data_hook(self, old_val: CORE_TYPE | None, val: CORE_TYPE | None) -> CORE_TYPE | None:
        # override for any hooks / logic here for data
        # e.g. log, trigger dirty flag, coerce type
        for callback in self._callbacks:
            callback(old_val, val)

    def __hash__(self):
        return id(self) 
    
    @classmethod
    def to_schema(cls) -> dict:
        """Converts the shareable item into a spec. Simply a wrapper for model_json_schema

        Returns:
            dict: _description_
        """
        return cls.model_json_schema()
    
    def to_spec(self) -> dict:
        """Converts the Shareable item to a 
        specification. Simply a wrapper for 
        model_dump.

        Returns:
            dict: The specification as a dictionary.
        """
        return self.model_dump()
    
    @classmethod
    def from_spec(cls, spec: dict) -> ShareableItem[CORE_TYPE]:
        """Reconstruct a ShareableItem from its specification.
        Args:
            spec (dict): The specification dictionary.
        Returns:
            ShareableItem[J]: The reconstructed ShareableItem instance.
        """
        return cls.model_validate(spec)

    def has_callback(self, callback: Callable[[CORE_TYPE | None, CORE_TYPE | None], None]) -> bool:
        return callback in self._callbacks

    def register_callback(self, callback: Callable[[CORE_TYPE | None, CORE_TYPE | None], None]) -> None:
        """Register a callback to be called when the data is updated."""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[CORE_TYPE | None, CORE_TYPE | None], None]) -> bool:
        """Unregister a previously registered callback. 
        If callback does not exist will return False"""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def __eq__(self, other):
        if isinstance(other, ShareableItem):
            return self.data == other.data
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __call__(self, value: CORE_TYPE):
        self.data = value
        return self.data

    # Arithmetic dunder methods
    def __add__(self, other):
        result_data = self.data + (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __sub__(self, other):
        result_data = self.data - (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __mul__(self, other):
        result_data = self.data * (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __truediv__(self, other):
        result_data = self.data / (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __floordiv__(self, other):
        result_data = self.data // (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __mod__(self, other):
        result_data = self.data % (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __pow__(self, other):
        result_data = self.data ** (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    # Reverse arithmetic dunder methods
    def __radd__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) + self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rsub__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) - self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rmul__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) * self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rtruediv__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) / self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rfloordiv__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) // self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rmod__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) % self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __rpow__(self, other):
        result_data = (other.data if isinstance(other, ShareableItem) else other) ** self.data
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    # In-place arithmetic dunder methods
    def __iadd__(self, other):
        result_data = self.data + (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __isub__(self, other):
        result_data = self.data - (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __imul__(self, other):
        result_data = self.data * (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __itruediv__(self, other):
        result_data = self.data / (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __ifloordiv__(self, other):
        result_data = self.data // (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __imod__(self, other):
        result_data = self.data % (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __ipow__(self, other):
        result_data = self.data ** (other.data if isinstance(other, ShareableItem) else other)
        new_instance = self.model_copy()
        new_instance.data = result_data
        return new_instance

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, ShareableItem) else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, ShareableItem) else other)

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, ShareableItem) else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, ShareableItem) else other)

    def load(self, data):
        """
        Rebuild a ShareableItem from a spec or dict.
        """
        print('Loading ', data, self.__class__)
        loaded = self.__class__.model_validate(data)
        print('Loaded: ', self.data.__class__)
        self.data = loaded.data

    def dump(self) -> dict:
        """
        Dump the ShareableItem to a dictionary.
        """

        return self.model_dump()

    def __repr__(self):
        
        return f"{self.__class__.__name__}(data={repr(self.data)})"
    
    def __str__(self):
        return str(self.data)


class Param(ShareableItem[CORE_TYPE]):
    """Trainable parameter; ``training`` may be toggled to freeze it."""

    _fixed: bool = False
    
    def set(self, data):
        if self._fixed:
            raise RuntimeError(
                'Cannot set parameter that is fixed.'
            )
        data = super().set(data)
        return data

    def is_fixed(self) -> bool:
        """
        Check if the parameter is fixed.
        """
        return self._fixed
    
    def fix(self):
        """
        Fix the parameter, making it unmodifiable.
        """
        self._fixed = True

    def unfix(self):
        """
        Unfix the parameter, making it modifiable.
        """
        self._fixed = False   


class Runtime(ShareableItem[CORE_TYPE]):
    """Mutable runtime state (e.g. counters, RNG seeds, rolling averages).

    Example:

    attr = Attr[float](data=0.0)
    """
    pass


class Shared(ShareableItem[CORE_TYPE]):
    """Pointer‑like wrapper whose value should *not* enter ``state_dict``.
    
    Example:

    shared = Shared[float](data=0.0)
    """
    pass


class ParamSet(pydantic.BaseModel):
    """ParamSet is a collection of parameters.
    """
    params: t.List[Param] = Field(default_factory=list)

    def update(self, param_set: Dict):
        """Update the parameters from a dictionary.

        Args:
            param_set: Dictionary with param_0, param_1, etc. keys
            flat: If True, expects flat dict like {"param_0": "value"}.
                  If False (default), expects schema-compliant structure.
        """
        updated = ParamSet.model_validate(
            param_set
        )
        for old_param, new_param in zip(self, updated):
            old_param.set(new_param.data)

    def to_schema(self) -> dict:
        """Get the JSON-schema for this ParamSet."""
        # create a schema dict by looping over all params. DO NOT USE model_json_schema
        schema: dict = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for i, param in enumerate(self.params):
            param_key = f"param_{i}"
            schema["properties"][param_key] = param.model_json_schema()
            schema["required"].append(param_key)
        return schema

    def from_spec(self, schema: dict) -> "ParamSet":
        """Reconstruct a ParamSet from its JSON-schema."""
        # reconstruct by looping over all params
        params = []
        for i in range(len(self.params)):
            param_key = f"param_{i}"
            param_schema = schema["properties"][param_key]
            param = self.params[i].from_spec(param_schema)
            params.append(param)
        return ParamSet(params=params)
    
    def to_spec(self) -> dict:
        """Convert the ParamSet to a specification dictionary."""
        spec: dict = {}
        for i, param in enumerate(self.params):
            param_key = f"param_{i}"
            spec[param_key] = param.to_spec()
        return spec

    def __iter__(self) -> t.Iterator[PARAM]:
        return iter(self.params)
    
    def __len__(self) -> int:
        return len(self.params)
    
    # @pydantic.field_validator('params', mode='before')
    # def validate_regions(cls, v):
    #     """Validate and convert regions to ModuleList

    #     Args:
    #         v: The regions input (list, ModuleList)

    #     Returns:
    #         Param[CORE_TYPE]: The regions as a ModuleList
    #     """
    #     # Accept any ModuleList regardless of type parameter
    #     # Accept ModuleList and convert

    #     # get the annotation args for the generic for ModuleList 
        
    #     base_param = cls.model_fields['params'].annotation.__pydantic_generic_metadata__['args'][0]

    #     if isinstance(v, list):
    #         converted = Param[base_param](data=v)
    #         return converted
    #     if isinstance(v, Tuple):
    #         converted = Param[base_param](data=v.data)
    #         return converted

    #     return v



PARAM = t.TypeVar("P", bound=Param)


class Trainable:
    """
    """

    @abstractmethod
    def parameters(
        self,
        *,
        recurse: bool = True,
        _seen: t.Optional[set[int]] = None,
        with_annotations: bool = False,
    ) -> t.Iterator[Param | tuple[Param, t.Any]]:
        """Yield all trainable parameters.
        Args:
            recurse: If True, recursively yield parameters from submodules.
            _seen: Internal set to track seen parameters and avoid duplicates.
            with_annotations: If True, yield (Param, annotation) tuples.
        Yields:
            Iterator[Param | tuple[Param, Any]]: Trainable parameters
        """

        raise NotImplementedError

