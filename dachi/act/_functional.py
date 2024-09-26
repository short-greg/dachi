# 1st party
import typing
import asyncio
from functools import partial

# local
from ._core import (
    TaskStatus, Task, ROUTE
)
from .._core._utils import Context, ContextSpawner


TASK = typing.Union[Task, typing.Callable[[typing.Dict], TaskStatus]]
CALL_TASK = typing.Callable[[],TaskStatus]

async def _parallel(
    tasks: typing.Iterable[TASK], 
    success_on: int, 
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

        for _, task in enumerate(tasks):

            tg_tasks.append(tg.create_task(
                (asyncio.to_thread(task))
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
    succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True
) -> CALL_TASK:
    """Create a parallel task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to run
        succeeds_on (int, optional): The number of Successes required to succeed. Defaults to -1.
        fails_on (int, optional): The number of Failures required to fail. Defaults to 1.
        success_priority (bool, optional): Whether success is prioritized over failure. Defaults to True.

    Returns:
        CALL_TASK: The task to call
    """
    def _f():
        return asyncio.run(
            _parallel(tasks, succeeds_on, fails_on, success_priority)
        )
    return _f


def parallelf(
    f: typing.Callable[[typing.Any], typing.Iterator[TASK]],
    *args, 
    succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True,
    **kwargs
) -> CALL_TASK:
    """Create a parallel task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to run
        succeeds_on (int, optional): The number of Successes required to succeed. Defaults to -1.
        fails_on (int, optional): The number of Failures required to fail. Defaults to 1.
        success_priority (bool, optional): Whether success is prioritized over failure. Defaults to True.

    Returns:
        CALL_TASK: The task to call
    """
    f = partial(f, *args, **kwargs)
    return parallel(f, succeeds_on, fails_on, success_priority)


def spawn(
    f, n: int,
    *args, 
    **kwargs
) -> typing.List[TASK]:
    """Convert a task into multiple tasks by spawning multiple
    states. Any "states" that are included in args or kwargs

    Args:
        f: The function to execute
        n (int): The number of tasks to spawn

    Returns:
        typing.List[TASK]: The list of tasks to run
    """

    tasks = []

    for i in range(n):
        cur_args = [
            arg[i] if isinstance(arg, ContextSpawner) else arg
            for arg in args.items()
        ]
        cur_kwargs = {
            k: arg[i] if isinstance(arg, ContextSpawner) else arg
            for k, arg in kwargs.items()
        }

        tasks.append(partial(f, *cur_args, **cur_kwargs))

    return tasks


def sequence(tasks: typing.Iterable[TASK], ctx: Context) -> CALL_TASK:
    """Run a sequence task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to execute
        state (State): The current state

    Returns:
        CALL_TASK: The task to call
    """
    def _f():

        status = ctx.get_or_set('status', TaskStatus.RUNNING)

        if status.is_done:
            return status
        
        # get the iterator
        if 'it' not in ctx:
            ctx['it'] = iter(tasks)

            try:
                cur_task = next(ctx['it'])
                ctx['cur_task'] = cur_task
            except StopIteration:
                ctx['status'] = TaskStatus.SUCCESS
                return TaskStatus.SUCCESS
            
        cur_task = ctx['cur_task']
            
        cur_status = cur_task()

        # update the sequence status
        if cur_status.running:
            ctx['status'] = TaskStatus.RUNNING
            return ctx['status']

        if cur_status.success:

            try:
                cur_task = next(ctx['it'])
                ctx['cur_task'] = cur_task
                ctx['status'] = TaskStatus.RUNNING
            except StopIteration:
                ctx['status'] = TaskStatus.SUCCESS
                return TaskStatus.SUCCESS
        else:
            ctx['status'] = cur_status

        return ctx['status']

    return _f


def sequencef(f: typing.Callable[[typing.Any], typing.Iterator[TASK]], ctx: Context, *args, **kwargs) -> CALL_TASK:
    """Run a callable sequence task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to execute
        state (State): The current state

    Returns:
        CALL_TASK: The task to call
    """
    return sequence(partial(f, *args, **kwargs), ctx)


# SELECTOR does not seem to be working correctly

def _selector(
    tasks: typing.List[TASK], 
    ctx: Context
) -> TaskStatus:

    status = ctx.get_or_set('status', TaskStatus.RUNNING)

    if status.is_done:
        return status
    
    # get the iterator
    if 'it' not in ctx:
        ctx['it'] = iter(tasks)

        try:
            cur_task = next(ctx['it'])
            ctx['cur_task'] = cur_task
        except StopIteration:
            ctx['status'] = TaskStatus.FAILURE
            return TaskStatus.FAILURE
        
    cur_task = ctx['cur_task']
        
    cur_status = cur_task()


    if cur_status.running:
        ctx['status'] = TaskStatus.RUNNING
        return ctx['status']

    if cur_status.failure:

        try:
            cur_task = next(ctx['it'])
            ctx['cur_task'] = cur_task
            ctx['status'] = TaskStatus.RUNNING
        except StopIteration:
            ctx['status'] = TaskStatus.FAILURE
            return TaskStatus.FAILURE
    else:
        ctx['status'] = cur_status

    return ctx['status']


def selector(tasks: TASK, ctx: Context) -> CALL_TASK:
    """Create a selector task

    Args:
        tasks (TASK): The tasks to run
        state (State): The state for the selector

    Returns:
        CALL_TASK: The task to call
    """
    def _f():
        return _selector(tasks, ctx)

    return _f


def selectorf(
    f: typing.Callable[[typing.Any], typing.Iterator[TASK]], 
    ctx: Context, *args, **kwargs) -> CALL_TASK:
    """Run a callable selector task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to execute
        state (State): The current state

    Returns:
        CALL_TASK: The task to call
    """
    return selector(partial(f, *args, **kwargs), ctx)


fallback = selector
fallbackf = selectorf


def actionf(task: Task, *args, router: ROUTE=None, **kwargs) -> CALL_TASK:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f():
        result = task(*args, **kwargs)
        if router:
            return router(result)
        return result

    return _f


def condf(task: typing.Callable, *args, **kwargs) -> CALL_TASK:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f():
        result = task(*args, **kwargs)
        return TaskStatus.from_bool(result)

    return _f


def not_(task: TASK) -> CALL_TASK:
    """Invert the result of the task if Failure or Success

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f():
        status = tick(task)
        return status.invert()

    return _f


def notf(f, *args, **kwargs) -> CALL_TASK:
    """Invert the result of the task if Failure or Success

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    return not_(partial(f, *args, **kwargs))


def tick(task: TASK) -> TaskStatus:
    """Run the task

    Args:
        task (TASK): The task to run

    Returns:
        TaskStatus: The resulting status from the task
    """

    if isinstance(task, Task):
        return task.tick()
    
    return task()


def unless(task: TASK, status: TaskStatus=TaskStatus.FAILURE) -> CALL_TASK:
    """Loop unless a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        TaskStatus: The status of the result
    """
    def _f():
        cur_status = task()
        
        if cur_status == status:
            return TaskStatus.RUNNING
        return cur_status
    return _f


def unlessf(f, *args, status: TaskStatus=TaskStatus.FAILURE, **kwargs) -> CALL_TASK:
    """Loop unless a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        CALL_TASK: The status of the result
    """
    return unless(partial(f, *args, **kwargs), status)


def until(task: TASK, status: TaskStatus=TaskStatus.SUCCESS) -> CALL_TASK:
    """Loop until a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        TaskStatus: The status of the result
    """

    def _f():
        cur_status = task()
        
        if cur_status == status:
            return cur_status
        return TaskStatus.RUNNING
    return _f


def untilf(f, *args, status: TaskStatus=TaskStatus.SUCCESS, **kwargs) -> CALL_TASK:
    """Loop until a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        TaskStatus: The status of the result
    """

    return until(partial(f, *args, **kwargs), status)
