from enum import Enum

class _Types(Enum):

    UNDEFINED = 'UNDEFINED'
    WAITING = 'WAITING'


UNDEFINED = _Types.UNDEFINED
"""Constant for UNDEFINED. usage: value is UNDEFINED"""
WAITING = _Types.WAITING
"""Constant for WAITING when streaming. usage: value is WAITING"""


primitives = (bool, str, int, float, type(None))
"""a list of primitive types"""


def is_primitive(obj) -> bool:
    """Utility to check if a value is a primitive

    Args:
        obj: Value to check

    Returns:
        bool: If it is a "primitive"
    """
    return type(obj) in primitives



"""
Decorator that turns a class into a singleton accessed via `Cls.obj`.

Compatible with normal classes *and* Pydantic BaseModel (v1 & v2).

Key points
----------
* `Cls()` raises TypeError with a helpful message.
* `Cls.obj` lazily constructs & returns the single instance.
* Each subclass automatically becomes its own singleton.
* Original validation / __init__ code *still runs* (important for Pydantic).
* Clear error chaining if the first construction fails.
"""

def singleton(cls):
    # already done?
    if getattr(cls, "__is_singleton__", False):
        return cls

    orig_meta = type(cls)                     # e.g. ModelMetaclass for Pydantic

    class _SingletonMeta(orig_meta):
        # block direct instantiation
        def __call__(self, *a, **kw):
            raise TypeError(
                f"{self.__name__} is a singleton. "
                f"Use {self.__name__}.obj instead of instantiating it."
            )

        # runs for every subclass
        def __init__(self, name, bases, ns, **kw):
            super().__init__(name, bases, ns, **kw)
            self._instance = None
            self.__is_singleton__ = True      # avoid re-decoration

        def _get_instance(self, *a, **kw):
            if self._instance is None:
                try:
                    # CALL THE ORIGINAL FACTORY so Pydantic validation happens
                    self._instance = super(_SingletonMeta, self).__call__(*a, **kw)
                except Exception as e:
                    # annotate & re-raise with original traceback
                    raise type(e)(
                        f"Error while creating the singleton instance of "
                        f"{self.__name__}: {e}"
                    ).with_traceback(e.__traceback__) from e
            return self._instance

        # make *every* subclass a singleton automatically
        def __init_subclass__(subcls, **kw):
            super().__init_subclass__(**kw)
            singleton(subcls)                 # no-op if already wrapped

    attrs = {
        "__module__" : cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__"   : cls.__doc__,
        "__is_singleton__": True,
    }
    Wrapped = _SingletonMeta(cls.__name__, (cls,), attrs)

    # descriptor so `Wrapped.obj` is an attribute
    class _ObjDescriptor:
        def __get__(self, _, owner):
            return owner._get_instance()

    Wrapped.obj = _ObjDescriptor()
    return Wrapped
