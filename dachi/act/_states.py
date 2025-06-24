import typing
from ._core import Task, State, TaskStatus
from ..proc import Process, AsyncProcess


# TODO: Decide how to refactor

class BranchState(State):
    """Executes a function which decides on
    which branch to follow
    """

    f: Process | AsyncProcess
    success: State
    failure: State

    async def update(self):
        
        if isinstance(self.f, AsyncProcess):
            if await self.f():
                return self.success
        else:
            if self.f():
                return self.success
        return self.failure


class TaskState(State):
    """Wraps a behavior tree task in a state
    """

    task: Task
    success: State
    failure: State | None = None

    def __post_init__(
        self
    ):
        """

        Args:
            task (Task): The task to wrap
            success (State): The state to go to on success
            failure (State, optional): The state to go to on failure. If not defined will be the same as success. Defaults to None.
        """
        super().__init__()
        self.failure = self.failure if self.failure else self.success

    async def update(self) -> 'State':
        """

        Args:
            reset (bool, optional): Whether to reset the state. Defaults to False.

        Returns:
            State: The outgoing state
        """
        status = await self.task.tick()
        if status.failure:
            return self.failure
        if status.success:
            return self.success
        return TaskStatus.RUNNING
