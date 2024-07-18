from abc import ABC, abstractmethod
from ._core2 import Module


class Param(object):

    def __init__(self, **p):
        """
        """
        self.p = p

    def update(self, **kwargs):
        self.p.update(kwargs)


class ParamModule(ABC):
    
    @abstractmethod
    def parameters(self) -> Param:
        pass
