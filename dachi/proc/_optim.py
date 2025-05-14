import typing
from abc import ABC, abstractmethod

import pydantic

from ..proc import Param
from ..base import Renderable

from ..inst import EvaluationBatch, Evaluation
from ..store import ParamSet, update_params


class Optim(ABC):
    """
    An optimizer is a function that takes in a set of parameters and returns a value.
    The optimizer is used to update the parameters based on the evaluation.
    """
    def __init__(self, params: typing.Iterable[Param]):
        """

        Args:
            params (typing.Iterable[Param]): 
        """
        super().__init__()
        self._params = ParamSet(params)

    @abstractmethod
    def update(self, evaluation: Evaluation | EvaluationBatch):
        """

        Args:
            critique (Critique): 
        """
        pass

    def step(self, evaluation: Evaluation | EvaluationBatch):
        """

        Args:
            critique (Critique): 
        """
        update_params(
            self.param_set, 
            self.update(evaluation)
        )