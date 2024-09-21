# 1st party
from abc import abstractmethod
import typing
import asyncio

# local
from ._core import TaskStatus, Task, get_or_set, get_or_spawn


TASK = typing.Union[Task, typing.Callable[[typing.Dict], TaskStatus]]


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

        for i, task in enumerate(tasks):

            cur_state = get_or_spawn(state, i)

            if isinstance(task, Task):
                tg_tasks.append(
                    tg.create_task(asyncio.to_thread(task.tick))
                )
            else:
                tg_tasks.append(tg.create_task(
                    (asyncio.to_thread(task, cur_state))
                ))

        if success_on < 0:
            success_on = len(tg_tasks) + 1 + success_on

        if fails_on < 0:
            fails_on = len(tg_tasks) + 1 + fails_on

        if (fails_on + success_on) > (len(tg_tasks) + 1):
            raise ValueError(
                f'Success + failure must be lte the number '
                f'of tasks not {fails_on + success_on}'
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


def parallel(
    tasks: typing.Iterable[TASK], 
    state: typing.Dict, succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True
) -> TaskStatus:

    return asyncio.run(
        _parallel(tasks, state, succeeds_on, fails_on, success_priority)
    )


def multi(
    task: TASK, state: typing.Dict, 
    n: int, succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True
) -> TaskStatus:

    tasks = [task] * n
    return asyncio.run(
        _parallel(tasks, state, succeeds_on, fails_on, success_priority)
    )


def sequence(tasks: typing.Iterable[TASK], state: typing.Dict) -> TaskStatus:
    
    idx = get_or_set(state, 'idx', 0)
    status = get_or_set(state, 'status', TaskStatus.RUNNING)
    
    if status.is_done:
        return status
    if idx >= len(tasks):
        return TaskStatus.SUCCESS
    
    cur_task = tasks[idx]
    if isinstance(cur_task, Task):
        cur_task.tick()
    else:
        child_state = get_or_spawn(state, idx)
        cur_status = cur_task(child_state)
    idx += 1
    if cur_status.success and idx == len(tasks):
        state['status'] = TaskStatus.SUCCESS
    elif cur_status.success:
        state['status'] = TaskStatus.RUNNING
    else:
        state['status'] = cur_status
    state['idx'] = idx

    return state['status']


def selector(tasks: typing.List[TASK], 
    state: typing.Dict
) -> TaskStatus:

    idx = get_or_set(state, 'idx', 0)
    status = get_or_set(state, 'status', TaskStatus.RUNNING)
    
    if status.is_done:
        return status
    if idx >= len(tasks):
        return TaskStatus.FAILURE
    
    cur_task = tasks[idx]
    if isinstance(cur_task, Task):
        cur_task.tick()
    else:
        child_state = get_or_spawn(state, idx)
        cur_status = cur_task(child_state)
    
    idx += 1
    if cur_status.failure and idx == len(tasks):
        state['status'] = TaskStatus.FAILURE
    elif cur_status.failure:
        state['status'] = TaskStatus.RUNNING
    else:
        state['status'] = cur_status
    state['idx'] = idx

    return state['status']


def action(
    task: TASK, state: typing.Optional[typing.Dict], 
    *args, **kwargs
) -> TaskStatus:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    if isinstance(task, Task):
        return task.tick()
    if state is False:
        return task(*args, **kwargs)
    return task(state, *args, **kwargs)


def cond(task: TASK, state: typing.Dict, *args, **kwargs) -> TaskStatus:
    """Functional form of condition

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    if isinstance(task, Task):
        return task.tick()
    if state is False:
        result = task(*args, **kwargs)
    else:
        result = task(state, *args, **kwargs)
    
    return TaskStatus.from_bool(result)


def unless(task: TASK, state: typing.Dict, status: TaskStatus=TaskStatus.FAILURE) -> TaskStatus:
    """Use to loop unless a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """

    if isinstance(task, Task):
        cur_status = task.tick()
    elif isinstance(task, TaskStatus):
        cur_status = task
    else:
        cur_status = task(state)

    if cur_status == status:
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
    if isinstance(task, Task):
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
    if isinstance(task, Task):
        cur_status = task.tick()
    elif isinstance(task, TaskStatus):
        cur_status = task
    elif state is None:
        cur_status = task()
    else:
        cur_status = task(state)
    
    return cur_status.invert()


def nest_multi(task: TASK, n: int) -> typing.Callable:
    
    def _f(state: typing.Dict):
        return multi(task, state, n)

    return _f


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


def nest_action(task: Task, *args, use_state: bool=True, **kwargs) -> TaskStatus:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f(state: typing.Dict):
        if use_state is False:
            state = None
        return action(task, state, *args, **kwargs)

    return _f


def nest_cond(task: Task, *args, use_state: bool=True, **kwargs) -> TaskStatus:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f(state: typing.Dict):
        if use_state is False:
            state = None
        return cond(task, state, *args, **kwargs)

    return _f


def nest_not(task: TASK, use_state: bool=True) -> TaskStatus:
    """Invert the result of the task if Failure or Success

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f(state: typing.Dict):
        if use_state is False:
            state = None
        return not_(task, state)

    return _f
