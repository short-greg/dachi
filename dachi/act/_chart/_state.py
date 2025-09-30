from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Union
from dachi.core import Attr
from abc import abstractmethod

from dachi.core import BaseModule
from ._event import Post

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class State(BaseModule):
    # ----- Declarative sugar (optional) -----
    class inputs:
        # Inputs defines what inputs are passed into the state's run() method.
        pass
    """
    # Inputs defines what inputs are passed into the state's run() method.
    class inputs:
        input1: str
        input2: int
        billboard: Billboard
    """
    class emit:
        # Emit defines what events the state can emit. Can subclass Event or be a string
        pass
    """
    class emit:
        class Event1(Payload):
            field1: str
            ...
        event: str
    """
    def __post_init__(self):
        super().__post_init__()
        self._active = Attr[bool](data=False)

    @abstractmethod
    def enter(self):
        pass

    @abstractmethod
    async def run(
        self, *, 
        post: "Post", 
        **inputs: Any
    ):
        pass

    @abstractmethod
    def exit(self) -> None:
        pass

    @abstractmethod
    def is_final(self) -> bool:
        pass


class FinalState(State):
    # FinalState has no long-running work; entering it marks the region complete.
    
    def __post_init__(self):
        super().__post_init__()

    def enter(self):
        pass

    def exit(self):
        pass

    async def run(
        self, 
        *, post: Post, 
        **inputs: Any
    ): # -> AsyncIterator[None]
        pass

    def is_final(self) -> bool:
        pass


class StepState(State):
    # Optional sugar: implement step(); the framework drives it inside run().

    @abstractmethod
    async def step(
        self, *, 
        post: Post, 
        **inputs: Any
    ) -> AsyncIterator[None]: 
        yield

    def run(self, *, post: Post, **inputs: Any):

        status = None
        for status in self.step(
            post=post, **inputs
        ):
            if self._active.get():
                break
        return status

