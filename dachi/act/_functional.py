# 1st party
from abc import abstractmethod
import typing
from dataclasses import dataclass
import time
from typing import Self
import asyncio

# local
from .._core import Storable
from ._status import TaskStatus
from . import _tasks
from . import _status


# require the state?

TASK = typing.Union[_tasks.Task, typing.Callable[[typing.Dict], TaskStatus]]


async def _parallel(
    tasks: typing.Iterable[TASK], 
    state: typing.Dict, success_on: int, 
    fails_on: int, success_priority: bool=True
) -> TaskStatus:
    """

    Args:
        tasks (typing.List[TASK]): 
        state (typing.Dict): 
        success_on (int): 
        fails_on (int): 
        success_priority (bool)

    Raises:
        ValueError: 

    Returns:
        TaskStatus: 
    """

    tg_tasks = []
    async with asyncio.TaskGroup() as tg:

        for task in tasks:

            threads = []
            if isinstance(task, _tasks.Task):
                threads.append(asyncio.to_thread(task.tick))
                tg_tasks.append(
                    tg.create_task()
                )
            else:
                threads.append(asyncio.to_thread(task, state))

        if success_on < 0:
            success_on = len(tg_tasks) + 1 + success_on

        if fails_on < 0:
            fails_on = len(tg_tasks) + 1 + fails_on

        if (fails_on + success_on) > len(tg_tasks):
            raise ValueError(
                f'Success + failure must be lte the number '
                f'of tasks not {fails_on + success_on}'
            )
        
        for thread in threads:
            tg_tasks.append(
                tg.create_task(thread)
            )

    failed = 0
    succeeded = 0

    for tg_task in tg_tasks:
        cur_status = tg_task.result()
        if cur_status.success:
            succeeded += 1
        if cur_status.failure:
            failed += 1

    if success_priority:
        if succeeded >= success_on:
            return TaskStatus.SUCCESS
        if failed >= fails_on:
            return TaskStatus.FAILURE
    else:
        if failed >= fails_on:
            return TaskStatus.FAILURE
        if succeeded >= success_on:
            return TaskStatus.SUCCESS
    return TaskStatus.RUNNING


def parallel(tasks: typing.Iterable[TASK], state: typing.Dict, succeeds_on: int=-1, fails_on: int=1, success_priority: bool=True) -> TaskStatus:

    return asyncio.run(_parallel(tasks, state, succeeds_on, fails_on, success_priority))


def sequence(tasks: typing.Iterable[TASK], state: typing.Dict) -> TaskStatus:
    
    idx = _status.get_or_set(state, 'idx', 0)
    status = _status.get_or_set(state, 'status', TaskStatus.RUNNING)
    if status.is_done:
        return status
    
    cur_task = tasks[idx]
    if isinstance(cur_task, _tasks.Task):
        cur_task.tick()
    else:
        child_state = _status.get_or_spawn(state, idx)
        cur_status = cur_task(child_state)
    if cur_status.success:
        state['status'] = TaskStatus.RUNNING
    else:
        state['status'] = cur_status

    return state['status']


def selector(tasks: typing.List[TASK], 
    state: typing.Dict
) -> TaskStatus:

    idx = _status.get_or_set(state, 'idx', 0)
    status = _status.get_or_set(state, 'status', TaskStatus.RUNNING)
    if status.is_done:
        return status
    
    child_state = _status.get_or_spawn(state, idx)
    cur_status = tasks[idx](child_state)

    cur_task = tasks[idx]
    if isinstance(cur_task, _tasks.Task):
        cur_task.tick()
    else:
        child_state = _status.get_or_spawn(state, idx)
        cur_status = cur_task(child_state)

    if cur_status.failure:
        state['status'] = TaskStatus.RUNNING
    else:
        state['status'] = cur_status

    return state['status']


def action(task: TASK, state: typing.Optional[typing.Dict], *args, **kwargs) -> TaskStatus:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    return task(state, *args, **kwargs)


def cond(task: TASK, state: typing.Dict, *args, **kwargs) -> TaskStatus:
    """Functional form of condition

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    return TaskStatus.from_bool(task(state, *args, **kwargs))


def unless(task: TASK, state: typing.Dict, status: TaskStatus=TaskStatus.FAILURE) -> TaskStatus:
    """Use to loop unless a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """

    if isinstance(task, _tasks.Task):
        cur_status = task.tick()
    elif isinstance(task, TaskStatus):
        cur_status = task
    else:
        cur_status = task(state)

    if cur_status != status:
        return cur_status
    return TaskStatus.RUNNING


def until(task: TASK, state: typing.Dict, status: TaskStatus=TaskStatus.SUCCESS) -> TaskStatus:
    """Use to loop until a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """

    if isinstance(task, _tasks.Task):
        cur_status = task.tick()
    elif isinstance(task, TaskStatus):
        cur_status = task
    else:
        cur_status = task(state)
    
    if cur_status == status:
        return cur_status
    return TaskStatus.RUNNING


def not_(task: TASK, state: typing.Dict) -> TaskStatus:
    """Invert the result of the task if Failure or Success

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    if isinstance(task, _tasks.Task):
        cur_status = task.tick()
    elif isinstance(task, TaskStatus):
        cur_status = task
    else:
        cur_status = task(state)
    
    return cur_status.invert()


def nest_parallel(tasks: typing.Iterable[TASK]) -> typing.Callable:
    
    def _f(state: typing.Dict):
        return parallel(tasks, state)

    return _f


def nest_sequence(tasks: typing.Iterable[TASK]) -> typing.Callable:
    
    def _f(state: typing.Dict):
        return sequence(tasks, state)

    return _f


def nest_selector(tasks: TASK) -> typing.Callable:

    def _f(state: typing.Dict):
        return selector(tasks, state)

    return _f


def nest_action(task: typing.Iterable[TASK], *args, **kwargs) -> TaskStatus:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f(state: typing.Dict):
        return action(task, state, *args, **kwargs)

    return _f


def nest_not(task: TASK) -> TaskStatus:
    """Invert the result of the task if Failure or Success

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f(state: typing.Dict):
        return not_(task, state)

    return _f

# until
# unless
