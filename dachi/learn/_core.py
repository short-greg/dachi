from typing import Any
import typing
from abc import ABC, abstractmethod
from .._core import Module, Data, Param, unescape_curly_braces, escape_curly_braces
from ..converse import PromptModel, Message
from ..critique import Critic, Evaluation


class Learner(Module):

    @abstractmethod
    def partial_fit(self, x, t=None):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass


def update(params: typing.Dict[int, Param], proposal: typing.Dict[int, str]) -> typing.Dict[int, Param]:

    for key, val in proposal.items():
        params[key].update(val)


class Proposer(Module, ABC):

    @abstractmethod
    def forward(self, params: typing.Dict[int, Param], evaluation: Evaluation) -> typing.Dict[int, str]:
        pass
    

class Optim(Module, ABC):

    def __init__(self, parameters: typing.Iterator[Param]):

        self.parameters = {
            i: p
            for i, p in enumerate(parameters)
        }

    @abstractmethod
    def forward(self, y: Data, t: Data) -> Any:
        pass


class Stochastic(Optim):

    def __init__(self, parameters: typing.Iterator[Param], critic: Critic, proposer: Proposer):

        super().__init__(parameters)
        self.critic = critic
        self.proposer = proposer

    def forward(self, y: Data, t: Data) -> Any:
        
        evaluation = self.critic(y, t)
        proposal = self.proposer(self.parameters, evaluation)
        return update(self.parameters, proposal)


class Proposer(Module, ABC):

    @abstractmethod
    def forward(self, params: typing.Dict[int, Param], evaluation: Evaluation) -> typing.Dict[int, str]:
        pass


class LLMProposer(Module):

    def __init__(self, llm: PromptModel):
        self.llm = llm

    def forward(
        self, params: typing.Dict[int, Param], 
        evaluation: Evaluation
    ) -> typing.Dict[int, str]:
        
        message = Message(
            role='system',
            content=unescape_curly_braces(
                f"""
                # 
                
                # Evaluation

                {escape_curly_braces(evaluation)}
                
                # Params

                {escape_curly_braces(params)}

                # Format

                {{
                    ""
                    
                }}

                """
            )
        )
        
        return self.llm(
            message
        )

