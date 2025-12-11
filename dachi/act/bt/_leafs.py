# 1st party
from abc import abstractmethod
import typing as t
import time
import random

# local
from ._core import TaskStatus, LeafTask
from dachi.core import Runtime, PrivateRuntime


class Condition(LeafTask):
    """A task that checks a condition
    """

    @abstractmethod
    async def execute(self, *args, **kwargs) -> bool | t.Tuple[bool, t.Dict]:
        """Execute the condition logic

        Returns:
            bool or (bool, outputs_dict): The result of the condition, optionally with outputs
        """
        raise NotImplementedError

    async def tick(self, ctx) -> TaskStatus:
        """Check the condition

        Returns:
            TaskStatus or (TaskStatus, outputs_dict): Whether the condition failed or succeeded
        """
        if self.status.is_done:
            return self.status
        try:
            inputs = self.build_inputs(ctx)
        except KeyError as e:
            self._status.set(TaskStatus.FAILURE)
            return self.status
        result = await self.execute(**inputs)
        if isinstance(result, tuple):
            cond, outputs = result
            ctx.update(outputs)
        else:
            cond = result
        status = TaskStatus.SUCCESS if cond else TaskStatus.FAILURE
        self._status.set(status)
        return status


CONDITION = t.TypeVar("CONDITION", bound=Condition)    

# TODO: Update the WaitCondition class tick
# to use ctx
class WaitCondition(LeafTask):
    """A task that waits for a condition to be met
    """
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> bool | t.Tuple[bool, t.Dict]:
        """Execute the condition

        Returns:
            bool: The result of the condition
        """
        raise NotImplementedError

    async def tick(self, ctx) -> TaskStatus:
        """Check the condition

        Returns:
            TaskStatus: Whether the condition failed or succeeded
        """

        try:
            inputs = self.build_inputs(ctx)
        except KeyError as e:
            self._status.set(TaskStatus.FAILURE)
            return self.status
        res = await self.execute(**inputs)
        if isinstance(res, t.Tuple):
            res, outputs = res
            ctx.update(outputs)
        self._status.set(
            TaskStatus.SUCCESS
            if res
            else TaskStatus.WAITING
        )
        return self.status


class Action(LeafTask):
    """A task that performs some kind of action
    """

    @abstractmethod
    async def execute(self, *args, **kwargs) -> TaskStatus | t.Tuple[TaskStatus, t.Dict]:
        """Execute the action logic

        Returns:
            TaskStatus or (TaskStatus, outputs_dict): The status after executing, optionally with outputs
        """
        raise NotImplementedError

    async def tick(self, ctx) -> TaskStatus:
        """Execute the action

        Returns:
            TaskStatus or (TaskStatus, outputs_dict): The resulting status
        """
        if self.status.is_done:
            return self.status
        try:
            inputs = self.build_inputs(ctx)
        except KeyError as e:
            self._status.set(TaskStatus.FAILURE)
            return self.status
        result = await self.execute(**inputs)
        if isinstance(result, tuple):
            status, outputs = result
            ctx.update(outputs)
        else:
            status = result
        self._status.set(status)
        return status


ACTION = t.TypeVar("ACTION", bound=Action)


class FixedTimer(Action):
    """A timer that will "succeed" at a fixed interval
    """
    seconds: float
    _start: float | None = PrivateRuntime(
        None
    )

    async def execute(self) -> TaskStatus:
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

    _start: Runtime[float | None] = PrivateRuntime(None)
    _target: Runtime[float | None] = PrivateRuntime(None)

    def reset(self):
        super().reset()
        self._start.set(None)
        self._target.set(None)
    
    async def execute(self) -> TaskStatus:
        """Execute the Timer

        Returns:
            TaskStatus: The status of the task
        """
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
    _i: Runtime[int] = PrivateRuntime(0)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._i = Runtime[int](data=0)

    async def execute(self):
        
        self._i.data += 1
        print(f"Count: {self._i.data}, Limit: {self.count}")
        if self._i.data >= self.count:
            return self.on_reached
        return TaskStatus.RUNNING
    
    def reset(self):
        super().reset()
        self._i.set(0)
