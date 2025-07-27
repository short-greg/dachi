import typing as t
from ._core import Task, State, TaskStatus
from ..proc import Process, AsyncProcess



class BranchState(State):
    """Branch state has two branches, one for success and one for failure. It wraps a process that returns a boolean value
    """

    f: Process | AsyncProcess

    async def update(self) -> t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE]:
        """ Update the state by executing the wrapped process and returning the status based on its result.

        Returns:
            t.Literal[TaskStatus.SUCCESS, TaskStatus.FAILURE]: The status of the branch state, either SUCCESS or FAILURE.
            If the wrapped process returns True, it will return SUCCESS, otherwise it will return FAILURE.
            If the wrapped process is an AsyncProcess, it will await the process before returning the status
        """

        if isinstance(self.f, AsyncProcess):
            if await self.f():
                return TaskStatus.SUCCESS
        else:
            if self.f():
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
        return await self.task.tick()


            # if filter_type is None or isinstance(obj, filter_type):
            #     fn(obj)
