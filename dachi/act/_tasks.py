# 1st party
from abc import abstractmethod
import typing
import time
import random

# local
from . import _functional
from ._core import Task, TaskStatus, State
from ..store._data import Context
from contextlib import contextmanager


class Root(Task):
    """The root task for a behavior tree
    """

    def __init__(self, root: Task | None=None):
        
        self.root = root

    def tick(self, reset: bool=False) -> TaskStatus:
        """Update the task

        Returns:
            SangoStatus: The status after tick
        """
        if reset:
            self.root.reset_status()
        if self.root is None:
            return TaskStatus.SUCCESS
        return self.root.tick(reset)


class Serial(Task):
    """A task consisting of other tasks executed one 
    after the other
    """

    def __init__(
        self, 
        tasks: typing.List[Task]=None,
        context: Context=None
    ):
        super().__init__()
        self._context = context or Context()
        self.tasks = tasks or []


class Sequence(Serial):
    """Create a sequence of tasks to execute
    """

    def __init__(self, **data):
        """Create a sequence of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks making up the sequence
        """
        super().__init__(**data)
        self._f = _functional.sequence(
            self.tasks, self._context
        )

    def tick(self, reset: bool=False) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status
        """
        if reset:
            self.reset_status()
            self._context = Context()
            self._f = _functional.sequence(
                self.tasks, self._context
            )

        self._status = self._f(reset)
        return self._status


class Selector(Serial):
    """Create a set of tasks to select from
    """
    # _f: typing.Callable = pydantic.PrivateAttr()

    def __init__(self, **data):
        """Create a selector of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks to select from
        """
        super().__init__(**data)
        self._f = _functional.selector(
            self.tasks, self._context
        )

    def tick(self, reset: bool=False) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The resulting task status
        """
        if reset:
            self.reset_status()
            self._context = Context()
            self._f = _functional.selector(
                self.tasks, self._context
            )
        self._status = self._f(reset)
        return self._status


Fallback = Selector


class Parallel(Task):
    """A composite task for running multiple tasks in parallel
    """

    def __init__(
        self, tasks: typing.List[Task],
        fails_on: int=1, succeeds_on: int=-1,
        success_priority: bool=True
    ):
        """The parallel

        Args:
            tasks (typing.Iterable[Task]): 
            runner (, optional): . Defaults to None.
            fails_on (int, optional): . Defaults to None.
            succeeds_on (int, optional): . Defaults to None.
            success_priority (bool, optional): . Defaults to True.
        """
        super().__init__()
        self.tasks = tasks
        self._succeeds_on = succeeds_on
        self._fails_on = fails_on
        self._success_priority = success_priority
        self._update_f()
        self._f = _functional.parallel(
            self.tasks, self._succeeds_on, self._fails_on,
            self._success_priority
        )

    def _update_f(self):
        self._f = _functional.parallel(
            self.tasks, self.succeeds_on, self.fails_on,
            self.success_priority
        )

    def validate(self):
        
        if (self._fails_on + self._succeeds_on - 1) > len(self.tasks):
            raise ValueError('')
        if self._fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('')
    
    def tick(self, reset: bool=False) -> TaskStatus:
        """Tick the task

        Returns:
            TaskStatus: The status
        """
        if reset:
            self.reset_status()
            self._f = _functional.parallel(
                self.tasks, self.succeeds_on, self.fails_on,
                self.success_priority
            )
        self._status = self._f(reset)
        return self._status

    @property
    def fails_on(self) -> int:
        return self._fails_on

    @fails_on.setter
    def fails_on(self, val) -> int:
        self._fails_on = val
        self._update_f()
        return val

    @property
    def succeeds_on(self) -> int:
        """Get the number required for success

        Returns:
            int: The number of successes to succeed
        """
        return self._succeeds_on        
    
    @succeeds_on.setter
    def succeeds_on(self, val) -> int:
        """Set the number required for success

        Args:
            succeeds_on: the number required for success

        Returns:
            int: The number of successes to succeed
        """
        self._succeeds_on = val
        self._update_f()
        return val
    
    @property
    def success_priority(self) -> bool:
        """Get the number required for success

        Returns:
            int: The number of successes to succeed
        """
        return self._success_priority        
    
    @success_priority.setter
    def success_priority(self, val: bool) -> int:
        """Set the number required for success

        Args:
            succeeds_on: the number required for success

        Returns:
            int: The number of successes to succeed
        """
        self._success_priority = val
        self._update_f()
        return val
     

class Action(Task):
    """A standard task that executes 
    """

    @abstractmethod
    def act(self, reset: bool=False) -> TaskStatus:
        """Commit an action

        Raises:
            NotImplementedError: 

        Returns:
            TaskStatus: The status of after executing
        """
        raise NotImplementedError

    def tick(self, reset: bool=False) -> TaskStatus:
        """Execute the action

        Returns:
            TaskStatus: The resulting status
        """
        if reset:
            self.reset_status()
            self.reset()
        if self._status.is_done:
            return self._status
        self._status = self.act(reset)
        return self._status


class Condition(Task):
    """Check whether a condition is satisfied before
    running a task.
    """
    @abstractmethod
    def condition(self, reset: bool=False) -> bool:
        """Execute between

        Returns:
            bool: The result of the condition
        """
        pass

    def tick(self, reset: bool=False) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        if reset:
            self.reset_status()
        self._status = TaskStatus.SUCCESS if self.condition(reset) else TaskStatus.FAILURE
        return self._status


class Decorator(Task):

    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    @abstractmethod
    def decorate(self, status: TaskStatus, reset: bool=False) -> bool:
        pass

    def tick(self, reset: bool=False) -> TaskStatus:
        """Decorate the tick for the decorated task

        Args:
            terminal (Terminal): The terminal for the task

        Returns:
            SangoStatus: The decorated status
        """

        if reset:
            self.reset_status()
        if self._status.is_done:
            return self._status

        self._status = self.decorate(
            self.task(reset), reset
        )
        return self._status


class Until(Decorator):
    """Loop until a condition is met
    """

    # target_status: TaskStatus = TaskStatus.SUCCESS

    def __init__(self, task: Task, target_status: TaskStatus=TaskStatus.SUCCESS):
        super().__init__(task)
        self.target_status = target_status

    def decorate(self, status: TaskStatus, reset: bool=False) -> TaskStatus:
        """Continue running unless the result is a success

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            return status
        if status.is_done:
            self.task.reset_status()
        return TaskStatus.RUNNING


class AsLongAs(Decorator):
    """Loop while a condition is met
    """
    # target_status: TaskStatus = TaskStatus.FAILURE

    def __init__(
        self, task: Task, 
        target_status: TaskStatus=TaskStatus.SUCCESS
    ):
        super().__init__(task)
        self.target_status = target_status

    def decorate(
        self, status: TaskStatus, reset: bool=False
    ) -> TaskStatus:
        """Continue running unless the result is a failure

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            if status.is_done:
                self.task.reset_status()
        elif status.is_done:
            return status
        return TaskStatus.RUNNING


class Not(Decorator):
    """Invert the result
    """

    def decorate(self, status: TaskStatus, reset: bool=False) -> TaskStatus:
        """Return Success if status is a Failure or Failure if it is a SUCCESS

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        return status.invert()


def run_task(
    task: Task, 
    interval: typing.Optional[float]=1./60
) -> typing.Iterator[TaskStatus]:
    """Run a task until completion

    Args:
        task (Task): The task to execute
        interval (float, optional): The interval to execute on. Defaults to 1./60.

    Yields:
        Iterator[typing.Iterator[TaskStatus]]: The status
    """
    status = None
    while status == TaskStatus.RUNNING or status == TaskStatus.READY:
        status = task.tick()
        if interval is not None:
            time.sleep(interval)
        yield status


class StateMachine(Task):
    """StateMachine is a task composed of multiple tasks in a directed graph
    """

    # init_state: State
    # _cur_state: State = pydantic.PrivateAttr()

    def __init__(self, init_state: State):
        """Create the status

        Args:
            init_state (State): The starting state for the machine
        """
        super().__init__()
        self.init_state = init_state
        self._cur_state = self.init_state

    def tick(self, reset: bool=False) -> TaskStatus:
        """Update the state machine
        """
        if reset:
            self.reset_status()
        if self._status.is_done:
            return self._status
        
        self._cur_state = self._cur_state()
        if self._cur_state == TaskStatus.FAILURE or self._cur_state == TaskStatus.SUCCESS:
            self._status = self._cur_state
        else:
            self._status = TaskStatus.RUNNING
        return self._status

    def reset_status(self):
        """Reset the state machine
        """
        super().reset_status()
        self._cur_state = self.init_state


class FixedTimer(Action):
    """A timer that will "succeed" at a fixed interval
    """
    # seconds: float
    # _start: typing.Optional[float] = pydantic.PrivateAttr(default=None)

    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = seconds
        self._start = None

    def act(self, reset: bool=False) -> TaskStatus:
        """Execute the timer

        Returns:
            TaskStatus: The TaskStatus after running
        """
        if reset:
            self._start = None
        cur = time.time()
        if self._start is None:
            self._start = cur
        elapsed = cur - self._start
        if elapsed >= self.seconds:
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING


class RandomTimer(Action):
    """A timer that will randomly choose a time between two values
    """

    def __init__(self, seconds_lower: float, seconds_upper: float):
        super().__init__()
        self.seconds_lower = seconds_lower
        self.seconds_upper = seconds_upper
        self._start = None
        self._target = None
    
    def act(self, reset: bool=False) -> TaskStatus:
        """Execute the Timer

        Returns:
            TaskStatus: The status of the task
        """
        if reset:
            self._start = None
            self._target = None
        cur = time.time()
        if self._start is None:
            self._start = cur
            r = random.random() 
            self._target = r * self.seconds_lower + r * self.seconds_upper
        elapsed = cur - self._start
        if elapsed >= self._target:
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING


@contextmanager
def loop_aslongas(task: Task, status: TaskStatus, reset: bool=False):
    """A context manager for running a task functionally

    Args:
        task (Task): The task to manage
    """
    cur_status = task.status
    try:
        yield task, cur_status
    finally:
        if cur_status.is_done:
            if status != cur_status:
                return
            else:
                reset = True
    
        cur_status = task(reset)


@contextmanager
def loop_until(
    task: Task, 
    status: TaskStatus, 
    reset: bool=False
):
    """A context manager for running a task functionally

    Args:
        task (Task): The task to manage
    """
    cur_status = task.status
    try:
        yield task, cur_status
    finally:
        
        if cur_status.is_done:
            if status == cur_status:
                return
            else:
                reset = True

        cur_status = task(reset)
