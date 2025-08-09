import typing as t
from typing import Iterable
from ._core import Task, State, TaskStatus
from ..proc import Process, AsyncProcess
from dachi.core import ModuleDict, AdaptModule, BaseModule, Attr



class BranchState(State):
    """Branch state has two branches, one for success and one for failure. It wraps a process that returns a boolean value
    """

    f: Process | AsyncProcess

    async def check(self, val: t.Any) -> bool:
        """Check the value of the process

        Args:
            val (t.Any): The value to check

        Returns:
            bool: True if the value is True, otherwise False
        """
        if isinstance(val, bool):
            return val
        if val == 1:
            return True
        if val == 0:
            return False
        raise ValueError(
            f"Expected a boolean value, got {type(val)}: {val}"
        )

    async def update(self) -> t.Literal[
        TaskStatus.SUCCESS, TaskStatus.FAILURE
    ]:
        """ Update the state by executing the wrapped process and returning the status based on its result.

        Returns:
            t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE]: The status of the branch state, either SUCCESS or FAILURE.
            If the wrapped process returns True, it will return SUCCESS, otherwise it will return FAILURE.
            If the wrapped process is an AsyncProcess, it will await the process before returning the status
        """
        if isinstance(self.f, AsyncProcess):
            if await self.check(await self.f.aforward()):
                return TaskStatus.SUCCESS
        else:
            if await self.check(self.f()):
                return TaskStatus.SUCCESS
        return TaskStatus.FAILURE


class TaskState(State):
    """Wraps a behavior tree task in a state
    """

    task: Task

    async def update(self) -> t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.RUNNING]:
        """ Update the state by executing the wrapped task and returning the status based on its result.

        Args:
            reset (bool, optional): Whether to reset the state. Defaults to False.

        Returns:
            t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.RUNNING]: The status of the task.
        """
        if self.task.status.is_done:
            return self.task.status
        print('Ticking')
        return await self.task.tick()
