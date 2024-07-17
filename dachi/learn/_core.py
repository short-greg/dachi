from typing import Any
from abc import ABC, abstractmethod
from ..converse import Module


# 1) Parameters
# 2) Score
# 3) Evaluate module

class Param()


class Learner(Module):

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def partial_fit(self, x, t=None):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def score(self) -> float:
        pass
