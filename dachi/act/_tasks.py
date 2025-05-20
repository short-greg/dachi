# 1st party
from abc import abstractmethod
import typing
import time
import random
import threading

# local
from ._core import Task, TaskStatus, State
from ..core import Storable
from contextlib import contextmanager

from ..core import (
    list_state_dict,
    load_dict_state_dict,
    load_list_state_dict,
    dict_state_dict
)


class Root(Task):
    """The root task for a behavior tree
    """
    def __init__(self, root: Task | None=None):
        """Create a root task

        Args:
            root (Task | None, optional): . Defaults to None.
        """
        self.root = root

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            SangoStatus: The status after tick
        """
        if self.root is None:
            return TaskStatus.SUCCESS
        return self.root()

    def reset(self):
        super().reset()
        self.root.reset()

# how to handle this?

class FTask(Task):

    def __init__(
        self, 
        f: typing.Callable[[], typing.Callable],
        *args, 
        **kwargs
    ):
        """Create a task based on a function

        Args:
            f (typing.Callable[[], typing.Callable]): The function to execute
            *args: The arguments to pass to the function
            **kwargs: The keyword arguments to pass to the function
        """
        super().__init__()
        self._f = f
        self._args = args
        self._kwargs = kwargs
        self._cur = None

    def tick(self, reset: bool=False) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The status of the task
        """
        if reset is True:
            self._cur = None
    
        if self._cur is None:
            self._cur = self._f(
                *self._args, **self._kwargs
            )

        return self._cur()
    
    def state_dict(self):
        """"Retrieve the state dict for the object"""
        return {
            'kwargs': dict_state_dict(self._kwargs),
            'args': list_state_dict(self._args)
        }
    
    def load_state_dict(self, state_dict):
        load_list_state_dict(
            self._args, state_dict['args']
        )
        load_dict_state_dict(
            self._kwargs, state_dict['kwargs']
        )


class Serial(Task):
    """A task consisting of other tasks executed one 
    after the other
    """
    def __init__(
        self, 
        tasks: typing.List[Task]=None
    ):
        """

        Args:
            tasks (typing.List[Task], optional): The tasks. Defaults to None.
            context (Context, optional): . Defaults to None.
        """
        super().__init__()
        self._idx = 0
        self.tasks = tasks or []

    def load_state_dict(self, state_dict: typing.Dict):
        """Load the state dict for the object

        Args:
            state_dict (typing.Dict): The state dict
        """
        for k, v in self.__dict__.items():

            if k == "tasks":
                load_list_state_dict(v, state_dict[k])
            elif isinstance(v, Storable):
                v.load_state_dict(state_dict[k])
            else:
                self.__dict__[k] = state_dict[k]
        
    def state_dict(self) -> typing.Dict:
        """Retrieve the state dict for the object

        Returns:
            typing.Dict: The state dict
        """
        cur = {}

        for k, v in self.__dict__.items():
            if k == "tasks":
                cur[k] = list_state_dict(v)
            elif isinstance(v, Storable):
                cur[k] = v.state_dict()
            else:
                cur[k] = v
        return cur


class Sequence(Serial):
    """Create a sequence of tasks to execute
    """
    def __init__(self, tasks: typing.List[Task]):
        """Create a sequence of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks making up the sequence
        """
        super().__init__(tasks)
        self._tasks = tasks

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            TaskStatus: The status
        """
        
        if self._status.is_done:
            return self._status
        
        status = self._tasks[self._idx]()
        if status == TaskStatus.FAILURE:
            self._status = TaskStatus.FAILURE
        elif status == TaskStatus.SUCCESS:
            self._idx += 1
            if self._idx >= len(self._tasks):
                self._status = TaskStatus.SUCCESS
            else:
                self._status = TaskStatus.RUNNING

        return self._status
    
    def reset(self):
        
        super().reset()
        for task in self.tasks:
            if isinstance(task, Task):
                task.reset()

        self._idx = 0


class Threaded(Task):

    def __init__(
        self, 
        task: Task
    ):
        super().__init__()
        self.task = task
        self._t = None

    def tick(self):
        if self._t is None:
            self._t = threading.Thread(
                target=self.task,
                args=()
            )
        if self._t.is_alive():
            return TaskStatus.WAITING
        self._t = None
        return self.task.status


class Selector(Serial):
    """Create a set of tasks to select from
    """

    def __init__(self, tasks: typing.List[Task]):
        """Create a selector of tasks

        Args:
            tasks (typing.Iterable[Task]): The tasks to select from
        """
        super().__init__(tasks)
        self._idx = 0

    def tick(self) -> TaskStatus:
        """Execute the task

        Returns:
            TaskStatus: The resulting task status
        """

        if self._status.is_done:
            return self._status
        
        status = self._tasks[self._idx]()
        if status == TaskStatus.SUCCESS:
            self._status = TaskStatus.SUCCESS
        elif status == TaskStatus.FAILURE:
            self._idx += 1
            if self._idx >= len(self._tasks):
                self._status = TaskStatus.FAILURE
            else:
                self._status = TaskStatus.RUNNING

        return self._status
    
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

    def __init__(
        self, tasks: typing.List[Task],
        fails_on: int=1, succeeds_on: int=-1,
        success_priority: bool=True,
        # parallelizer: typing.Optional[
        #     _functional.PARALLEL
        # ]=None,
        preempt: bool=False,
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
        self._preempt = preempt
        # self._update_f()

        # self._f = _functional.parallel(
        #     self.tasks, 
        #     self._succeeds_on, 
        #     self._fails_on,
        #     self._success_priority,
        #     parallelizer=parallelizer,
        #     preempt=preempt
        # )

    # def _update_f(self):
    #     """Update the parallel function
    #     """
    #     self._f = _functional.parallel(
    #         self.tasks, 
    #         self.succeeds_on, 
    #         self.fails_on,
    #         self.success_priority
    #     )

    def validate(self):
        """Validate the number of tasks required to succeed and fail
        Raises:
            ValueError: If the number of tasks is less than the number of fails or succeeds
            ValueError: If the number of fails or succeeds is less than 0
        """
        if (
            self._fails_on + self._succeeds_on - 1
        ) > len(self.tasks):
            raise ValueError(
                'The number of tasks required to succeed or fail is greater than the number of tasks'
            )
        if self._fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('The number of fails or succeeds must be greater than 0')
    
    def tick(self) -> TaskStatus:
        """Tick the task

        Returns:
            TaskStatus: The status
        """
        statuses = []
        failures = 0
        successes = 0
        for task in self.tasks:
            if task.status.is_done:
                statuses.append(task.status)
            else:
                statuses.append(task())
            if statuses[-1] == TaskStatus.SUCCESS:
                successes += 1
            elif statuses[-1] == TaskStatus.FAILURE:
                failures += 1
        
        if not self.preempt and (
            failures + successes
        ) < len(statuses):
            self._status = TaskStatus.RUNNING
        elif (
            successes >= self._succeeds_on and
            failures >= self._fails_on
        ):
            self._status = TaskStatus.from_bool(
                self._success_priority
            )
        elif successes >= self._succeeds_on:
            self._status = TaskStatus.SUCCESS
        
        elif failures >= self._fails_on:
            self._status = TaskStatus.FAILURE
        else:
            self._status = TaskStatus.RUNNING
        return self._status

    @property
    def fails_on(self) -> int:
        """
        Returns:
            int: The number of failures required to fail
        """
        return self._fails_on

    @fails_on.setter
    def fails_on(self, val) -> int:
        """Set the number of failures required to fail

        Args:
            val: The number of failures required to fail

        Returns:
            int: The number of failures required to fail 
        """
        if val < 0:
            val = (
                len(self.tasks) + val + 1
            )
        self._fails_on = val
        if val + self._fails_on > len(self.tasks):
            raise ValueError(
                f''
            )
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
        if val < 0:
            val = (
                len(self.tasks) + self._succeeds_on + 1
            )
        if val + self._fails_on > len(self.tasks):
            raise ValueError(
                f''
            )
        self._succeeds_on = val
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
        return val

    def reset(self):
        super().reset()
        for task in self.tasks:
            if isinstance(task, Task):
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
        self._status = TaskStatus.SUCCESS if self.condition() else TaskStatus.FAILURE
        return self._status


class WaitCondition(Task):
    """This condition will return a WAITING status
    if it fails
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
        self._status = TaskStatus.SUCCESS if self.condition() else TaskStatus.WAITING
        return self._status


class Decorator(Task):

    def __init__(self, task: Task):
        """

        Args:
            task (Task): The task to decorate
        """
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
        # if reset:
        #     self.reset()
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
            self.task.reset()
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
                self.task.reset()
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
    while (
        status == TaskStatus.RUNNING 
        or status == TaskStatus.READY
    ):
        status = task.tick()
        if interval is not None:
            time.sleep(interval)
        yield status


class StateMachine(Task):
    """StateMachine is a task composed of multiple tasks in a directed graph
    """

    def __init__(self, init_state: State):
        """Create the status

        Args:
            init_state (State): The starting state for the machine
        """
        super().__init__()
        self.init_state = init_state
        self._cur_state = self.init_state

    def tick(self) -> TaskStatus:
        """Update the state machine
        """
        if self._status.is_done:
            return self._status
        
        self._cur_state = self._cur_state()
        if self._cur_state == TaskStatus.FAILURE or self._cur_state == TaskStatus.SUCCESS:
            self._status = self._cur_state
        else:
            self._status = TaskStatus.RUNNING
        return self._status

    def reset(self):
        """Reset the state machine
        """
        super().reset()
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

    def __init__(
        self, 
        seconds_lower: float, 
        seconds_upper: float
    ):
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


class PreemptCond(Task):
    """Use to have a condition applied with
    each tick in order to stop the execution
    of other tasks
    """
    def __init__(
        self, 
        conds: typing.List[Condition], 
        task: Task
    ):
        """Create a PreemptCond

        Args:
            conds (typing.Iterable[Condition]): The conditions to execute
            task (Task): The task to execute if satisfied
        """
        super().__init__()

        self.conds = conds
        self.task = task

    def tick(self) -> TaskStatus:
        """

        Args:
            reset (bool, optional): . Defaults to False.

        Returns:
            TaskStatus: 
        """
        status = TaskStatus.SUCCESS
        for cond in self.conds:
            cond.reset()
            status = cond.tick() & status
        
        if status.failure:
            self._status = TaskStatus.FAILURE
        
        else:
            self._status = self.task.tick()
        return self._status


class WaitCondition(Task):
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
        self._status = TaskStatus.SUCCESS if self.condition() else TaskStatus.WAITING
        return self._status


class CountLimit(Action):

    def __init__(
        self, 
        count: int,
        on_reached: TaskStatus=TaskStatus.SUCCESS
    ):
        super().__init__()
        self._i = 0
        self._count = count
        self._on_reached = on_reached

    def act(self):
        
        self._i += 1
        if self._i == self._count:
            return self._on_reached
        return TaskStatus.RUNNING
    
    def reset(self):
        super().__init__()
        self._i = 0


# def count_limit(
#     count: int, 
#     ctx: Context, 
#     on_reached: TaskStatus.SUCCESS
# ) -> typing.Callable:
#     """

#     Args:
#         count (int): The number to execute
#         ctx (Context): 
#         on_reached (TaskStatus.SUCCESS): The status to return when reached the count

#     Returns:
#         typing.Callable
#     """

#     ctx.i = 0
#     if not on_reached.is_done:
#         raise ValueError(
#             'On reached must be a finished status'
#         )

#     def _(reset: bool=False) -> TaskStatus:
#         if reset:
#             ctx.i = 0
#         ctx.i += 1
#         if ctx.i == count:
#             return on_reached
#         return TaskStatus.RUNNING
#     return _


# def threaded_task(
#     f, ctx: Context, shared: SharedBase=None, to_status: TOSTATUS=None, 
#     callback: typing.Callable[[Context], typing.NoReturn]=None
# ) -> CALL_TASK:
#     """Use to wrap the task in a thread"""

#     if 'task_id' in ctx and id(f) != ctx['task_id']:

#         raise RuntimeError(
#             'Task context has been initialized but '
#             'the task passed in is does not match'
#         )

#     def _f(reset: bool=False):
#         """Run the task in a thread"""
#         if 'tick_id' not in ctx or reset:
#             ctx['tick_id'] = str(uuid.uuid4())
        
#         if '_thread' not in ctx:
#             ctx['thread_status'] = TaskStatus.WAITING
#             t = threading.Thread(
#                 target=_f_task_wrapper,
#                 args=(
#                     f, ctx,
#                     id, to_status,
#                     shared,
#                     callback
#                 )
#             )

#             ctx['_thread'] = t
#             t.start()
#             return TaskStatus.WAITING
        
#         if 'error' in ctx:
#             raise ctx['error']
        
#         # if callback:
#         #     callback(ctx)
#         if ctx['_thread'].is_alive():
#             return TaskStatus.WAITING
#         elif ctx['thread_status'].is_done:
#             print('Thread is alive but task is finished')
#         return ctx['thread_status']
#     return _f