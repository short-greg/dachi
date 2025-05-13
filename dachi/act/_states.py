import typing
from ._core import Task, State, TaskStatus


class BranchState(State):
    """Executes a function which decides on
    which branch to follow
    """

    def __init__(
        self, 
        f: typing.Callable,
        success: State,
        failure: State,
    ):
        """Executes a function which decides on
    which branch to follow

        Args:
            f (typing.Callable): The function to check
            success (State): The state to go to on success
            failure (State): The state to go to on failure
        """
        super().__init__()
        self.f = f
        self.success = success
        self.failure = failure

    def update(self, reset = False):
        
        if self.f():
            return self.success
        return self.failure


class TaskState(State):
    """Wraps a behavior tree task in a state
    """

    def __init__(
        self, 
        task: Task,
        success: State,
        failure: State=None,
    ):
        """

        Args:
            task (Task): The task to wrap
            success (State): The state to go to on success
            failure (State, optional): The state to go to on failure. If not defined will be the same as success. Defaults to None.
        """
        super().__init__()
        self.task = task
        self.success = success
        self.failure = failure if failure else success

    def update(self, reset = False) -> 'State':
        """

        Args:
            reset (bool, optional): Whether to reset the state. Defaults to False.

        Returns:
            State: The outgoing state
        """
        if reset:
            self.task.reset_status()

        status = self.task.tick()
        if status.failure:
            return self.failure
        if status.success:
            return self.success
        return TaskStatus.RUNNING
