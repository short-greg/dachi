# 1st party
from abc import abstractmethod
import typing as t
import time
import random
import asyncio
import threading
from dataclasses import InitVar

# local
from dachi.core import BaseModule, AdaptModule
from ._core import Task, TaskStatus, State, Composite, Leaf
from contextlib import contextmanager
from dachi.core import ModuleDict, Attr, ModuleList


class Condition(Leaf):
    """A task that checks a condition
    """

    @abstractmethod
    async def condition(self) -> bool:
        """Execute between

        Returns:
            bool: The result of the condition
        """
        pass

    async def tick(self) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self._status.set(
            TaskStatus.SUCCESS 
            if await self.condition() 
            else TaskStatus.FAILURE
        )
        return self.status
    

class WaitCondition(Leaf):
    """A task that waits for a condition to be met
    """
    
    @abstractmethod
    async def condition(self) -> bool:
        """Execute between

        Returns:
            bool: The result of the condition
        """
        pass

    async def tick(self) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self._status.set(
            TaskStatus.SUCCESS 
            if await self.condition() 
            else TaskStatus.WAITING
        )
        return self.status



class Action(Leaf):
    """A task that performs some kind of action
    """

    @abstractmethod
    async def act(self) -> TaskStatus:
        """Commit an action

        Raises:
            NotImplementedError: 

        Returns:
            TaskStatus: The status of after executing
        """
        raise NotImplementedError

    async def tick(self) -> TaskStatus:
        """Execute the action

        Returns:
            TaskStatus: The resulting status
        """
        if self.status.is_done:
            return self.status
        status = await self.act()
        self._status.set(status)
        return self.status


class FixedTimer(Action):
    """A timer that will "succeed" at a fixed interval
    """
    seconds: float

    def __post_init__(self):
        super().__post_init__()
        self._start = Attr[float | None](data=None)

    async def act(self) -> TaskStatus:
        """Execute the timer

        Returns:
            TaskStatus: The TaskStatus after running
        """
        cur = time.time()
        if self._start.get() is None:
            self._start.set(cur)
        elapsed = cur - self._start.get()
        if elapsed >= self.seconds:
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING



# HERE
class RandomTimer(Action):
    """A timer that will randomly choose a time between two values
    """
    seconds_lower: float
    seconds_upper: float

    def __post_init__(
        self
    ):
        super().__post_init__()
        self._start = Attr[None | float](data=None)
        self._target = Attr[None | float](data=None)
    
    async def act(self, reset: bool=False) -> TaskStatus:
        """Execute the Timer

        Returns:
            TaskStatus: The status of the task
        """
        if reset:
            self._start.set(None)
            self._target.set(None)
        cur = time.time()
        if self._start.get() is None:
            self._start.set(cur)
            r = random.random() 
            self._target = r * self.seconds_lower + r * self.seconds_upper
        elapsed = cur - self._start.get()
        if elapsed >= self._target:
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING



class CountLimit(Action):
    """A task that counts the number of times it has been run
    """

    count: int
    on_reached: TaskStatus=TaskStatus.SUCCESS

    def __post_init__(self):
        super().__post_init__()
        self._i = Attr[int](data=0)

    async def act(self):
        
        self._i.data += 1
        print(f"Count: {self._i.data}, Limit: {self.count}")
        if self._i.data >= self.count:
            return self.on_reached
        return TaskStatus.RUNNING
    
    def reset(self):
        super().reset()
        self._i.set(0)



class WaitCondition(Leaf):
    """Check whether a condition is satisfied before
    running a task.
    """

    @abstractmethod
    async def condition(self) -> bool:
        """Execute between

        Returns:
            bool: The result of the condition
        """
        pass

    async def tick(self) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self._status.set(
            TaskStatus.SUCCESS 
            if await self.condition() 
            else TaskStatus.WAITING
        )
        return self.status
