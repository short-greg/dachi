# 1st party
import typing
import asyncio
from functools import partial
import threading
import time

# local
from ._core import (
    TaskStatus, Task
)
from ..data import Context, ContextSpawner, SharedBase

from ._core import TOSTATUS

from ..data import Buffer,Shared
from .._core import AIModel, AIPrompt


TASK = typing.Union[Task, typing.Callable[[typing.Dict], TaskStatus]]
CALL_TASK = typing.Callable[[],TaskStatus]


PARALLEL = typing.Callable[[typing.Iterable[Task], int, int, bool], TaskStatus]


async def _parallel(
    tasks: typing.Iterable[TASK], 
    success_on: int=-1, 
    fails_on: int=1, success_priority: bool=True
) -> TaskStatus:
    """Run in parallel

    Args:
        tasks (typing.List[TASK]): The tasks to run
        success_on (int): The number required for success
        fails_on (int): The number 
        success_priority (bool): The 

    Raises:
        ValueError: If the value is not defined

    Returns:
        TaskStatus: The resulting status
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
    success_priority: bool=True,
    parallelizer: PARALLEL=None
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
        if parallelizer:
            return parallelizer(tasks, succeeds_on, fails_on, success_priority)
        
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
    return parallel(
        f(*args, **kwargs), succeeds_on, fails_on, success_priority
    )


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

        if isinstance(cur_task, bool):
            cur_status = TaskStatus.from_bool(cur_task)
        elif cur_task == TaskStatus.FAILURE or cur_task == TaskStatus.SUCCESS:
            cur_status = cur_status
        # It must be a task
        else:
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
    return sequence(f(*args, **kwargs), ctx)


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
    
    if isinstance(cur_task, bool):
        cur_status = TaskStatus.from_bool(cur_task)
    elif cur_task == TaskStatus.FAILURE or cur_task == TaskStatus.SUCCESS:
        cur_status = cur_status
    # It must be a task
    else:
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
    return selector(f(*args, **kwargs), ctx)


fallback = selector
fallbackf = selectorf


def action(task: TASK, *args, **kwargs) -> CALL_TASK:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f():
        result = task(*args, **kwargs)
        return result

    return _f


def taskf(f, *args, out: SharedBase=None, to_status: TOSTATUS=None, **kwargs) -> CALL_TASK:
    """A generic task based on a function

    Args:
        f: The function to exectue
        out (SharedBase, optional): The output to store to. Defaults to None.
        to_status (TOSTATUS, optional): The status converter. If not set will automatically set to "Success". 
            Defaults to None.

    Returns:
        CALL_TASK: The CALL_TASK to execute
    """
    def _f():
        result = f(*args, **kwargs)
        if out is not None:
            out.set(result)
        
        if to_status is not None:
            status = to_status(result)
        elif isinstance(result, TaskStatus):
            status = result
        else:
            status = TaskStatus.SUCCESS
        return status

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
    def _f() -> TaskStatus:
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


def _run_thread(task: TASK, ctx: Context, interval: float=1./60):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    while True:
        status = task()
        ctx['thread_status'] = status
        if status.is_done:
            break
        time.sleep(interval)


def threaded(task: TASK, ctx: Context, interval: float=1./60) -> CALL_TASK:

    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(target=_run_thread, args=(task, ctx, interval))
            t.start()
            ctx['_thread'] = t
        
        return ctx['thread_status']

    if 'task_id' in ctx and id(task) != ctx['task_id']:

        raise RuntimeError(
            'Task context has been initialized but '
            'the task passed in is does not match'
        )

    return run


def _stream_model(model: AIModel, prompt: AIPrompt, ctx: Context, *args, interval: float=1./60, **kwargs):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    print('Executing thread')
    for x, dx in model.stream_forward(prompt, *args, **kwargs):
        ctx['x'] = x
        ctx['dx'].append(dx)
        time.sleep(interval)
    ctx['thread_status'] = TaskStatus.SUCCESS


def stream_model(
    buffer: Buffer, engine: AIModel, prompt: AIPrompt, ctx: Context, 
    *args, interval: float=1./60,  **kwargs
) -> CALL_TASK:
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['x'] = None
            ctx['dx'] = []
            ctx['i'] = 0
            ctx['thread_status'] = TaskStatus.RUNNING
            print('create thread')
            t = threading.Thread(
                target=_stream_model, args=(
                    engine, prompt, ctx, *args
                ), kwargs={'interval': interval, **kwargs}
            )
            t.start()
            ctx['_thread'] = t
        
        if ctx['i'] < len(ctx['dx']):
            buffer.add(*ctx['dx'][ctx['i']:])
            ctx['i'] = len(ctx['dx'])

        return ctx['thread_status']

    return run


def _run_model(model: AIModel, prompt: AIPrompt, ctx: Context, **kwargs):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    ctx['x'] = model(prompt, **kwargs)
    ctx['thread_status'] = TaskStatus.SUCCESS


def exec_model(
    shared: Shared, engine: AIModel, prompt: AIPrompt, ctx: Context, 
    **kwargs
) -> CALL_TASK:
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['x'] = None
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_run_model, args=(
                    engine, prompt, ctx
                ), kwargs=kwargs
            )
            t.start()
            ctx['_thread'] = t
        
        if ctx['thread_status'].is_done:
            shared.set(ctx['x'])

        return ctx['thread_status']

    return run


def _run_func(ctx: Context, f, *args, **kwargs):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    ctx['x'] = f(*args, **kwargs)
    ctx['thread_status'] = TaskStatus.SUCCESS


def exec_func(
    shared: Shared, ctx: Context, f: typing.Callable,
    *args,
    **kwargs
) -> CALL_TASK:
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['x'] = None
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_run_func, args=(
                    ctx, f, *args
                ), kwargs=kwargs
            )
            t.start()
            ctx['_thread'] = t
        
        if ctx['thread_status'].is_done:
            shared.set(ctx['x'])

        return ctx['thread_status']

    return run


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
