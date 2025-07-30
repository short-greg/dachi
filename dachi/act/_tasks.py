# 1st party
from abc import abstractmethod
import typing
import time
import random
import asyncio
import threading
from typing import Iterable
from dataclasses import InitVar

# local
from dachi.core import BaseModule, AdaptModule
from ._core import Task, TaskStatus, State
from contextlib import contextmanager
from dachi.core import ModuleDict, Attr, ModuleList


class BT(AdaptModule, Task):
    """The root task for a behavior tree
    """
    root: InitVar[Task | None] = None

    def __post_init__(self, root: Task):
        """Create a behavior tree task
        Args:
            root (Task | None): The root task for the behavior tree
        """
        super().__post_init__()
        Task.__post_init__(self)
        self.adapted = root
        self.__method_tasks__ = {}
        for name in dir(self):
            # Get the attribute from the instance
            attr = getattr(self, name)
            # Check if it's a bound method of this instance
            if (
                callable(attr)
                and hasattr(attr, "__self__")
                and attr.__self__ is self
                and getattr(attr, "is_task", False) is True
            ):
                self.__method_tasks__[name] = attr
    
    def task(
        self, 
        name: str, 
        *args, 
        **kwargs
    ) -> Task:
        """Get a task by name

        Args:
            name (str): The name of the task
            *args: The arguments to pass to the task
            **kwargs: The keyword arguments to pass to the task

        Returns:
            Task: The task with the given name
        """
        if name not in self.__method_tasks__:
            raise ValueError(f"Task {name} not found")
        return self.__method_tasks__[name].create(
            name=name, 
            args=args, 
            kwargs=kwargs,
            f=self.__method_tasks__[name],
        )
    
    async def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status after tick
        """
        print(self._adapted)
        if self._adapted is None:
            return TaskStatus.SUCCESS
        return await self._adapted()

    def root(self) -> Task | None:
        return self._adapted

    def reset(self):
        super().reset()
        if self._adapted is not None:
            self._adapted.reset()


class Serial(Task):
    """A task consisting of other tasks executed one 
    after the other
    """
    tasks: ModuleList | None = None

    def __post_init__(
        self
    ):
        """

        Args:
            tasks (typing.List[Task], optional): The tasks. Defaults to None.
            context (Context, optional): . Defaults to None.
        """
        super().__post_init__()
        if self.tasks is None:
            self.tasks = ModuleList(data=[])
        elif isinstance(self.tasks, typing.List):
            self.tasks = ModuleList(data=self.tasks)
        if self.tasks is not None and not isinstance(self.tasks, ModuleList):
            raise ValueError(
                f"Tasks must be of type ModuleList not {type(self.tasks)}"
            )
        self._idx = Attr[int](data=0)


class Sequence(Serial):
    """Create a sequence of tasks to execute
    """

    async def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status
        """
        if self.status.is_done:
            return self.status
        
        status = await self.tasks[self._idx.data]()
        if status == TaskStatus.FAILURE:
            self._status.set(TaskStatus.FAILURE)
        elif status == TaskStatus.SUCCESS:
            self._idx.data += 1
            if self._idx.data >= len(self.tasks):
                self._status.set(TaskStatus.SUCCESS)
            else:
                self._status.set(TaskStatus.RUNNING)

        return self.status
    
    def reset(self):
        
        super().reset()
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()

        self._idx.data = 0
# TODO: Decide how to handle this


class Selector(Serial):
    """Create a set of tasks to select from
    """

    def __post_init__(self):
        """Create a selector of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks to select from
        """
        super().__post_init__()
        self._idx = Attr[int](data=0)

    async def tick(self) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The resulting task status
        """

        if self.status.is_done:
            return self.status
        
        status = await self.tasks[self._idx.data]()
        if status == TaskStatus.SUCCESS:
            self._status.set(TaskStatus.SUCCESS)
        elif status == TaskStatus.FAILURE:
            self._idx.data += 1
            if self._idx.data >= len(self.tasks):
                self._status.set(TaskStatus.FAILURE)
            else:
                self._status.set(TaskStatus.RUNNING)

        return self.status
    
    def reset(self):
        super().__init__()
        self._idx = 0
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()


Fallback = Selector


class Parallel(Task):
    """A composite task for running multiple tasks in parallel
    """

    tasks: ModuleList
    fails_on: int=1
    succeeds_on: int=-1
    success_priority: bool = True
    preempt: bool = False

    def validate(self):
        """Validate the number of tasks required to succeed and fail
        Raises:
            ValueError: If the number of tasks is less than the number of fails or succeeds
            ValueError: If the number of fails or succeeds is less than 0
        """
        if self.fails_on < 0:
            fails_on = len(self.tasks) + self.fails_on + 1
        else:
            fails_on = self.fails_on
        if self.succeeds_on < 0:
            succeeds_on = len(self.tasks) + self.succeeds_on + 1
        else:
            succeeds_on = self.succeeds_on

        print(f"Fails on: {fails_on}, Succeeds on: {succeeds_on}, Tasks: {len(self.tasks)}")
        if (
            fails_on + succeeds_on - 1
        ) > len(self.tasks):
            raise ValueError(
                'The number of tasks required to succeed or fail is greater than the number of tasks'
            )
        if fails_on <= 0 or succeeds_on <= 0:
            raise ValueError(
                'The number of fails or succeeds '
                'must be greater than 0'
            )
    
    async def tick(self) -> TaskStatus:
        """Tick the task

        Returns:
            TaskStatus: The status
        """
        tasks = []
        failures = 0
        successes = 0
        async with asyncio.TaskGroup() as tg:
            
            for task in self.tasks:
                if task.status.is_done:
                    tasks.append(task.status)
                else:
                    tasks.append(
                        tg.create_task(
                            task()
                        )
                    )
        statuses = [
            task if isinstance(
                task, TaskStatus
            ) else task.result()
            for task in tasks
        ]
        failures = sum(
            1 if status.failure else 0 for status in statuses
        )
        successes = sum(
            1 if status.success else 0 for status in statuses
        )

        if not self.preempt and (
            failures + successes
        ) < len(statuses):
            self._status.set(TaskStatus.RUNNING)
        elif (
            successes >= self.succeeds_on and
            failures >= self.fails_on
        ):
            self._status.set(TaskStatus.from_bool(
                self.success_priority
            ))
        elif successes >= self.succeeds_on:
            self._status.set(TaskStatus.SUCCESS)
        
        elif failures >= self.fails_on:
            self._status.set(TaskStatus.FAILURE)
        else:
            self._status.set(TaskStatus.RUNNING)
        return self.status

    def reset(self):
        super().reset()
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()


class Action(Task):
    """A standard task that executes 
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


class Condition(Task):
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
            else TaskStatus.FAILURE
        )
        return self.status


class WaitCondition(Task):
    """This condition will return a WAITING status
    if it fails
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


class Decorator(Task):

    task: Task

    @abstractmethod
    async def decorate(self, status: TaskStatus, reset: bool=False) -> bool:
        pass

    async def tick(self) -> TaskStatus:
        """Decorate the tick for the decorated task

        Args:
            terminal (Terminal): The terminal for the task

        Returns:
            SangoStatus: The decorated status
        """
        # if reset:
        #     self.reset()
        if self.status.is_done:
            return self.status

        self._status.set(await self.decorate(
            await self.task()
        ))
        return self.status


class Until(Decorator):
    """Loop until a condition is met
    """

    target_status: TaskStatus = TaskStatus.SUCCESS

    async def decorate(
        self, 
        status: TaskStatus
    ) -> TaskStatus:
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


class AsLongAs(Decorator):
    """Loop while a condition is met
    """
    target_status: TaskStatus = TaskStatus.SUCCESS

    async def decorate(
        self, status: TaskStatus
    ) -> TaskStatus:
        """Continue running unless the result is a failure

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status == self.target_status:
            if status.is_done:
                self.task.reset()
        elif status.is_done:
            return status
        return TaskStatus.RUNNING


class Not(Decorator):
    """Invert the result
    """

    async def decorate(
        self, 
        status: TaskStatus
    ) -> TaskStatus:
        """Return Success if status is a Failure or Failure if it is a SUCCESS

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        return status.invert()


async def run_task(
    task: Task, 
    interval: typing.Optional[float]=1./60
) -> typing.AsyncIterator[TaskStatus]:
    """Run a task until completion

    Args:
        task (Task): The task to execute
        interval (float, optional): The interval to execute on. Defaults to 1./60.

    Yields:
        Iterator[typing.Iterator[TaskStatus]]: The status
    """
    status = None
    while (
        status == TaskStatus.RUNNING 
        or status == TaskStatus.READY
    ):
        status = await task.tick()
        if interval is not None:
            time.sleep(interval)
        yield status


def statefunc(func):
    """Decorator to mark a function as a state for StateMachine."""
    func._is_state = True
    return func

# TODO: How to handle "statefuncs" 
# I think it p

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


@contextmanager
async def loop_aslongas(
    task: Task, 
    status: TaskStatus=TaskStatus.SUCCESS
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
            if status != cur_status:
                return
            else: 
                task.reset()
    
        cur_status = await task()


@contextmanager
async def loop_until(
    task: Task, 
    status: TaskStatus=TaskStatus.SUCCESS
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
                task.reset()
    
        cur_status = await task()


class AutoRun(Task):
    """A decorator that will automatically rerun the task until completed.
    Useful below parallel tasks in systems that do not
    run at intervals. Not useful for systems that run at intervals like robots or games.
    """

    task: Task
    active: bool = True

    async def tick(self) -> TaskStatus:
        """Run the task if it is not done

        Returns:
            TaskStatus: The status of the task after running
        """
        if self.status.is_done:
            return self.status
        
        if self.active:
            done = False
            while not done:
                status = await self.task.tick()
                done = status.is_done
        else:
            status = await self.task.tick()
        return status


class PreemptCond(Task):
    """Use to have a condition applied with
    each tick in order to stop the execution
    of other tasks
    """

    conds: ModuleList
    task: Task

    async def tick(self) -> TaskStatus:
        """

        Args:
            reset (bool, optional): . Defaults to False.

        Returns:
            TaskStatus: 
        """
        status = TaskStatus.SUCCESS
        for cond in self.conds:
            cond.reset()
            status = await cond.tick() & status
        
        if status.failure:
            self._status.set(
                TaskStatus.FAILURE
            )
        
        else:
            self._status.set(
                await self.task.tick()
            )
        return self.status


class WaitCondition(Task):
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


class CountLimit(Action):

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


class Threaded(Task):

    task: Task

    def __post_init__(self):
        super().__post_init__()
        self._t = None

    async def tick(self):
        if self._t is None:
            self._t = threading.Thread(
                target=self.task,
                args=()
            )
        if self._t.is_alive():
            return TaskStatus.WAITING
        self._t = None
        return self.task.status
