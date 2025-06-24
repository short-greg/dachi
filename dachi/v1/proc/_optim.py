import typing
from abc import ABC, abstractmethod
from ..inst import EvaluationBatch, Evaluation
from ..store import ParamSet, Param, update_params
from ._process import Module

class Optim(Module):
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

    def forward(self, evaluation: Evaluation | EvaluationBatch):
        """

        Args:
            critique (Critique): 
        """
        update_params(
            self.param_set, 
            self.update(evaluation)
        )
