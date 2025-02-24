# 1st party
from abc import abstractmethod
import typing
import time
import random


# 3rd party
import pydantic

# local
from . import _functional
from ._core import Task, TaskStatus, State
from ._data import Context


class Root(Task):
    """The root task for a behavior tree
    """

    root: Task

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            SangoStatus: The status after tick
        """
        if self.root is None:
            return TaskStatus.SUCCESS

        return self.root.tick()

    @property
    def root(self) -> Task:
        """
        Returns:
            Task: The root task
        """
        return self.root
    
    @root.setter
    def root(self, root: 'Task'):
        """
        Args:
            root (Task): The root task
        """
        self.root = root

    def reset(self):
        """Reset the status of the task

        """
        super().reset()
        if self.root is not None:
            self.root.reset()


class Serial(Task):
    """A task consisting of other tasks executed one 
    after the other
    """

    tasks: typing.Iterable[Task]
    _context: Context = pydantic.PrivateAttr(default=Context)

    @property
    def tasks(self) -> typing.Iterable[Task]:
        """Get the tasks in the serial task

        Returns:
            typing.Iterable[Task]: The tasks comprising the serial task
        """
        return self.tasks

    def reset(self):
        """Reset the state
        """
        super().reset()
        self._context = Context()
        for task in self.tasks:
            task.reset()


class Sequence(Serial):
    """Create a sequence of tasks to execute
    """

    _f: typing.Callable = pydantic.PrivateAttr()
    # context: Context = pydantic.PrivateAttr(default=Context)

    def __init__(self, **data):
        """Create a sequence of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks making up the sequence
        """
        super().__init__(**data)
        self._f = _functional.sequence(
            self.tasks, self._context
        )

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status
        """
        self.status = self._f()
        return self.status

    def reset(self):
        super().reset()
        self._f = _functional.sequence(
            self.tasks, self._context
        )


class Selector(Serial):
    """Create a set of tasks to select from
    """
    _f: typing.Callable = pydantic.PrivateAttr()

    def __init__(self, **data):
        """Create a selector of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks to select from
        """
        super().__init__(**data)
        self._f = _functional.selector(
            self.tasks, self._context
        )

    def tick(self) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The resulting task status
        """
        self.status = self._f()
        return self.status

    def reset(self):
        super().reset()
        self._f = _functional.selector(
            self.tasks, self._context
        )


Fallback = Selector


class Parallel(Task):
    """A composite task for running multiple tasks in parallel
    """

    fails_on: int = 1
    succeeds_on: int = -1
    success_priority: bool = True
    f: typing.Callable = pydantic.PrivateAttr()
    ticked: typing.Set = pydantic.PrivateAttr(default_factory=set)

    def __init__(self, **data):
        """The parallel

        Args:
            tasks (typing.Iterable[Task]): 
            runner (, optional): . Defaults to None.
            fails_on (int, optional): . Defaults to None.
            succeeds_on (int, optional): . Defaults to None.
            success_priority (bool, optional): . Defaults to True.
        """
        super().__init__(**data)
        # self._tasks = tasks if tasks is not None else []
        # self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        # self._succeeds_on = succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)

        # self._success_priority = success_priority
        self.f = _functional.parallel(
            self.tasks, self.succeeds_on, self.fails_on,
            self.success_priority
        )

    def _update_f(self):
        self.f = _functional.parallel(
            self.tasks, self.succeeds_on, self.fails_on,
            self.success_priority
        )

    @property
    def tasks(self) -> typing.Iterable[Task]:
        return self.tasks

    def set_condition(self, fails_on: int, succeeds_on: int):
        """Set the number of falures or successes it takes to end

        Args:
            fails_on (int): The number of failures it takes to fail
            succeeds_on (int): The number of successes it takes to succeed

        Raises:
            ValueError: If teh number of successes or failures is invalid
        """
        self.fails_on = fails_on
        self.succeeds_on = succeeds_on
        # self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        # self._succeeds_on = succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)

    def validate(self):
        
        if (self.fails_on + self.succeeds_on - 1) > len(self.tasks):
            raise ValueError('')
        if self.fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('')

    @property
    def fails_on(self) -> int:
        return self.fails_on

    @fails_on.setter
    def fails_on(self, fails_on) -> int:
        self.fails_on = fails_on
        self._update_f()
        return self.fails_on

    @property
    def succeeds_on(self) -> int:
        """Get the number required for success

        Returns:
            int: The number of successes to succeed
        """
        return self.succeeds_on        
    
    @succeeds_on.setter
    def succeeds_on(self, succeeds_on) -> int:
        """Set the number required for success

        Args:
            succeeds_on: the number required for success

        Returns:
            int: The number of successes to succeed
        """
        self.succeeds_on = succeeds_on
        self._update_f()
        return self.succeeds_on
    
    def tick(self) -> TaskStatus:
        """Tick the task

        Returns:
            TaskStatus: The status
        """
        self.status = self.f()
        return self.status

    def reset(self):
        """Reset the task and subtasks
        """

        super().reset()
        for task in self.ticked:
            task.reset()


class Action(Task):
    """A standard task that executes 
    """

    @abstractmethod
    def act(self) -> TaskStatus:
        """Commit an action

        Raises:
            NotImplementedError: 

        Returns:
            TaskStatus: The status of after executing
        """
        raise NotImplementedError

    def tick(self) -> TaskStatus:
        """Execute the action

        Returns:
            TaskStatus: The resulting status
        """
        if self.status.is_done:
            return self.status
        self.status = self.act()
        return self.status


class Condition(Task):
    """Check whether a condition is satisfied before
    running a task.
    """
    @abstractmethod
    def condition(self) -> bool:
        """Execute between

        Returns:
            bool: The result of the condition
        """
        pass

    def tick(self) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self.status = TaskStatus.SUCCESS if self.condition() else TaskStatus.FAILURE
        return self.status


class Decorator(Task):

    task: Task

    @abstractmethod
    def decorate(self, status: TaskStatus) -> bool:
        pass

    @property
    def task(self) -> Task:
        """Get the task to execute

        Returns:
            Task: The decorated task
        """
        return self.task

    def tick(self) -> TaskStatus:
        """Decorate the tick for the decorated task

        Args:
            terminal (Terminal): The terminal for the task

        Returns:
            SangoStatus: The decorated status
        """

        if self.status.is_done:
            return self.status

        self.status = self.decorate(
            self.task.tick()
        )
        return self.status

    def reset(self):
        """Reset the task and subtask
        """
        super().reset()
        self.task.reset()


class Until(Decorator):
    """Loop until a condition is met
    """

    target_status: TaskStatus = TaskStatus.SUCCESS

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
            self.task.reset()
        return TaskStatus.RUNNING


class Unless(Decorator):
    """Loop while a condition is met
    """
    target_status: TaskStatus = TaskStatus.SUCCESS

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
            self.task.reset()
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


def run_task(task: Task, interval: typing.Optional[float]=1./60) -> typing.Iterator[TaskStatus]:
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
    """StateMachine is a task composed of multiple tasks in
    a directed graph
    """

    init_state: State
    cur_state: State = pydantic.PrivateAttr()

    def __init__(self, **data):
        """Create the status

        Args:
            init_state (State): The starting state for the machine
        """
        super().__init__()
        self.cur_state = self.init_state

    def tick(self) -> TaskStatus:
        """Update the state machine
        """
        if self.status.is_done:
            return self.status
        
        self.cur_state = self.cur_state.update()
        if self.cur_state == TaskStatus.FAILURE or self._cur_state == TaskStatus.SUCCESS:
            self.status = self.cur_state
        else:
            self.status = TaskStatus.RUNNING
        return self.status

    def reset(self):
        """Reset the state machine
        """
        super().reset()
        self.cur_state = self.init_state


class FixedTimer(Action):
    """A timer that will "succeed" at a fixed interval
    """
    seconds: float
    start: typing.Optional[float] = pydantic.PrivateAttr(default=None)

    def act(self) -> TaskStatus:
        """Execute the timer

        Returns:
            TaskStatus: The TaskStatus after running
        """
        cur = time.time()
        if self.start is None:
            self.start = cur
        elapsed = cur - self.start
        if elapsed >= self.seconds:
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING

    def reset(self):
        """Reset the timer
        """
        super().reset()
        self.start = None


class RandomTimer(Action):
    """A timer that will randomly choose a time between two values
    """

    seconds_lower: float
    seconds_upper: float
    start: typing.Optional[float] = pydantic.PrivateAttr(default=None)
    target: typing.Optional[float] = pydantic.PrivateAttr(default=None)
    
    def act(self) -> TaskStatus:
        """Execute the Timer

        Returns:
            TaskStatus: The status of the task
        """
        cur = time.time()
        if self.start is None:
            self.start = cur
            r = random.random() 
            self.target = r * self.seconds_lower + r * self.seconds_upper
        elapsed = cur - self.start
        if elapsed >= self.target:
            return TaskStatus.SUCCESS
        return TaskStatus.RUNNING

    def reset(self):
        """Reset the task
        """
        super().reset()
        self.start = None
        self.target = None
