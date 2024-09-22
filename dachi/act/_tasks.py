# 1st party
from abc import abstractmethod
import typing
import time

# 3rd party
from . import _functional
from ._core import Task, TaskStatus
from .._core._utils import Context


class Root(Task):
    """The root task for a behavior tree
    """

    def __init__(self, root: 'Task'=None):
        """Create a tree to store the tasks

        Args:
            root (Task, optional): The root task. Defaults to None.
            name (str): The name of the tree
        """
        super().__init__()
        self._root = root

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            SangoStatus: The status after tick
        """
        if self._root is None:
            return TaskStatus.SUCCESS

        return self._root.tick()

    @property
    def root(self) -> Task:
        """
        Returns:
            Task: The root task
        """
        return self._root
    
    @root.setter
    def root(self, root: 'Task'):
        """
        Args:
            root (Task): The root task
        """
        self._root = root

    def reset(self):
        """Reset the status of the task

        """
        super().reset()
        if self._root is not None:
            self._root.reset()


class Serial(Task):
    """A task consisting of other tasks executed one 
    after the other
    """

    def __init__(self, tasks: typing.Iterable[Task]):
        """Create a serial task

        Args:
            tasks (typing.Iterable[Task]): The tasks comprising the serial task
        """
        super().__init__()
        self._tasks = tasks
        self._context = Context()
        
    @property
    def tasks(self) -> typing.Iterable[Task]:
        """Get the tasks in the serial task

        Returns:
            typing.Iterable[Task]: The tasks comprising the serial task
        """
        return self._tasks

    def reset(self):
        """Reset the state
        """
        super().reset()
        for task in self._tasks:
            task.reset()


class Sequence(Serial):
    """Create a sequence of tasks to execute
    """

    def __init__(self, tasks: typing.Iterable[Task]):
        """Create a sequence of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks making up the sequence
        """
        super().__init__(tasks)
        self.f = _functional.sequence(
            self._tasks, self._context
        )

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status
        """
        self._status = self.f()
        return self._status


class Selector(Serial):
    """Create a set of tasks to select from
    """

    def __init__(self, tasks: typing.Iterable[Task]):
        """Create a selector of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks to select from
        """
        super().__init__(tasks)
        self.f = _functional.selector(
            self._tasks, self._context
        )

    def tick(self) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The resulting task status
        """
        self._status = self.f()
        return self._status


Fallback = Selector


class Parallel(Task):
    """A composite task for running multiple tasks in parallel
    """

    def __init__(self, tasks: typing.Iterable[Task]=None, fails_on: int=None, succeeds_on: int=None, success_priority: bool=True):
        """The parallel

        Args:
            tasks (typing.Iterable[Task]): 
            runner (, optional): . Defaults to None.
            fails_on (int, optional): . Defaults to None.
            succeeds_on (int, optional): . Defaults to None.
            success_priority (bool, optional): . Defaults to True.
        """
        super().__init__()
        self._tasks = tasks if tasks is not None else []
        self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        self._succeeds_on = succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)

        self._success_priority = success_priority
        self._f = _functional.parallel(
            self._tasks, self._succeeds_on, self._fails_on,
            self._success_priority
        )
        self._ticked = set()

    def _update_f(self):
        self._f = _functional.parallel(
            self._tasks, self._succeeds_on, self._fails_on,
            self._success_priority
        )

    @property
    def tasks(self) -> typing.Iterable[Task]:
        return self._tasks

    def set_condition(self, fails_on: int, succeeds_on: int):
        """Set the number of falures or successes it takes to end

        Args:
            fails_on (int): The number of failures it takes to fail
            succeeds_on (int): The number of successes it takes to succeed

        Raises:
            ValueError: If teh number of successes or failures is invalid
        """
        self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        self._succeeds_on = succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)

    def validate(self):
        
        if (self._fails_on + self._succeeds_on - 1) > len(self._tasks):
            raise ValueError('')
        if self._fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('')

    @property
    def fails_on(self) -> int:
        return self._fails_on

    @fails_on.setter
    def fails_on(self, fails_on) -> int:
        self._fails_on = fails_on
        self._update_f()
        return self._fails_on

    @property
    def succeeds_on(self) -> int:

        return self._succeeds_on        
    
    @succeeds_on.setter
    def succeeds_on(self, succeeds_on) -> int:
        self._succeeds_on = succeeds_on
        self._update_f()

        return self._succeeds_on
    
    def tick(self) -> TaskStatus:
        
        self._status = self._f()
        return self._status

    def reset(self):

        super().reset()
        for task in self._ticked:
            task.reset()
        self._context = {}


class Action(Task):

    @abstractmethod
    def act(self) -> TaskStatus:
        raise NotImplementedError

    def tick(self) -> TaskStatus:

        if self._status.is_done:
            return self._status
        self._status = self.act()
        return self._status


class Condition(Task):
    """Check whether a condition is satisfied before
    running a task.
    """

    @abstractmethod
    def condition(self) -> bool:
        pass

    def tick(self) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self._status = TaskStatus.SUCCESS if self.condition() else TaskStatus.FAILURE
        return self._status


class Decorator(Task):

    # name should retrieve the name of the decorated
    def __init__(self, task: Task) -> None:
        """Decorate a task

        Args:
            task (Task): The task to decorate
        """
        super().__init__()
        self._task = task

    @abstractmethod
    def decorate(self, status: TaskStatus) -> bool:
        pass

    @property
    def task(self) -> Task:
        """Get the task to execute

        Returns:
            Task: The decorated task
        """
        return self._task

    def tick(self) -> TaskStatus:
        """Decorate the tick for the decorated task

        Args:
            terminal (Terminal): The terminal for the task

        Returns:
            SangoStatus: The decorated status
        """

        if self._status.is_done:
            return self._status

        self._status = self.decorate(
            self.task.tick()
        )
        return self._status

    def reset(self):

        super().reset()
        self._task.reset()


class Until(Decorator):
    """Loop until a condition is met
    """
    def __init__(self, task: Task, target_status: TaskStatus= TaskStatus.SUCCESS) -> None:
        super().__init__(task)
        self.target_status = target_status

    def decorate(self, status: TaskStatus) -> TaskStatus:
        """Continue running unless the result is a success

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            return status
        if status.is_done:
            self._task.reset()
        return TaskStatus.RUNNING


class Unless(Decorator):
    """Loop while a condition is met
    """
    def __init__(self, task: Task, target_status: TaskStatus= TaskStatus.FAILURE) -> None:
        super().__init__(task)
        self.target_status = target_status

    def decorate(self, status: TaskStatus) -> TaskStatus:
        """Continue running unless the result is a failure

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            return status
        if status.is_done:
            self._task.reset()
        return TaskStatus.RUNNING


class Not(Decorator):
    """Invert the result
    """

    def decorate(self, status: TaskStatus) -> TaskStatus:
        """Return Success if status is a Failure or Failure if it is a SUCCESS

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        return status.invert()


def run_task(task: Task, interval: float=1./60) -> typing.Iterator[TaskStatus]:

    status = None
    while status == TaskStatus.RUNNING or status == TaskStatus.READY:
        status = task.tick()
        if interval is not None:
            time.sleep(interval)
        yield status


class StateMachine(Task):
    """StateMachine is a task composed of multiple tasks in
    a directed graph
    """

    def __init__(self, init_state: 'State'):
        """

        Args:
            init_state (State): The starting state for the machine
        """
        self._init_state = init_state
        self._cur_state = init_state
        self._status = TaskStatus.READY

    def tick(self) -> TaskStatus:
        """Update the state machine
        """
        if self._status.is_done:
            return self._status
        
        self._cur_state = self._cur_state.update()
        if self._cur_state == TaskStatus.FAILURE or self._cur_state == TaskStatus.SUCCESS:
            self._status = self._cur_state
        else:
            self._status = TaskStatus.RUNNING
        return self._status

    @property
    def status(self) -> TaskStatus:
        """Get the status of the state machine

        Returns:
            TaskStatus: Get the current status
        """
        return self._status

    def reset(self):

        self._cur_state = self._init_state


class State(Task):

    @abstractmethod
    def update(self) -> typing.Union['State', TaskStatus]:
        pass
