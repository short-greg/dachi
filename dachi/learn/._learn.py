import typing
from abc import ABC, abstractmethod

import pydantic

from ..proc import Module, Param


# Hypothesis
# Prior
# Critique
# Evaluation
# Params
# Optim
# Learner


class ParamSet(object):

    def __init__(self, params: typing.Iterable[Param]):
        """

        Args:
            params (typing.Iterable[Param]): 
        """
        self._params = list(params)


def update_params(param_set: ParamSet, update: typing.List[typing.Dict]):
    """_summary_

    Args:
        param_set (ParamSet): 
        update (typing.List[typing.Dict]): 
    """

    for p, u in zip(param_set, update):
        p.update(u)


def create_tuples(**kwargs: typing.List):
    """
    Create a list of dictionaries from keyword arguments.
    Each dictionary in the list will have keys corresponding to the keyword arguments
    and values corresponding to the values of those keyword arguments, grouped by their
    position in the input.
    Args:
        **kwargs: Arbitrary keyword arguments where each key is a string and each value is an iterable.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a combination
        of the input keyword arguments' values.
    """
    keys = kwargs.keys()
    values = zip(*kwargs.values())
    return [dict(zip(keys, value)) for value in values]


class Critique(pydantic.Module):

    description: str = pydantic.Field(
        description="A description of the "
    )
    val: typing.Union[typing.List[typing.Dict], typing.Dict] = pydantic.Field(
        "The e . A dictionary if only one Element is evaluated. Use a list if multiple Elemenents are evaluated."
    )


class Optim(ABC):
    """
    """

    def __init__(self, params: typing.Iterable[Param]):
        """

        Args:
            params (typing.Iterable[Param]): 
        """
        super().__init__()
        self._params = ParamSet(params)

    @abstractmethod
    def update(self, critique: Critique):
        """

        Args:
            critique (Critique): 
        """
        pass

    def step(self, critique: Critique):
        """

        Args:
            critique (Critique): 
        """
        update_params(
            self.param_set, 
            self.update(critique)
        )
