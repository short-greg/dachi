# 1st party
import typing
import asyncio
from functools import partial
import threading
import time

# local
from ._core import (
    TaskStatus, Task, State
)
from ..store._data import Context, ContextSpawner, SharedBase
from ._core import TOSTATUS
from ..store._data import Buffer,Shared
from ..asst import LLM, LLM_PROMPT, FromMsg
import itertools

TASK = typing.Union[
    Task, typing.Callable[[typing.Dict], TaskStatus]]
CALL_TASK = typing.Callable[[],TaskStatus]


PARALLEL = typing.Callable[[typing.Iterable[Task], int, int, bool], TaskStatus]

def reset_arg_ctx(*args, **kwargs):
    """Use to reset the context to the args and kwargs
    """

    for k, arg in itertools.chain(enumerate(args), kwargs.items()):
        if isinstance(arg, Task):
            arg.reset_status()
        elif isinstance(arg, Context):
            arg.reset()

async def _parallel(
    tasks: typing.Iterable[TASK] | typing.Callable[[], typing.Iterable[TASK]], 
    success_on: int=-1, 
    fails_on: int=1, success_priority: bool=True,
    reset: bool=False
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

        _tasks = tasks() if callable(tasks) else tasks
        for _, task in enumerate(_tasks):

            tg_tasks.append(tg.create_task(
                (asyncio.to_thread(task, reset=reset))
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
    tasks: typing.Iterable[TASK] | typing.Callable[[], typing.Iterable[TASK]], 
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
    fails_on = (
        fails_on if fails_on is not None else len(self._tasks)
    )
    succeeds_on = (
        succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)
    )
    def _f(reset: bool=False):
        if parallelizer:
            return parallelizer(
                tasks, succeeds_on, 
                fails_on, 
                success_priority, 
                reset=reset
            )
        
        return asyncio.run(
            _parallel(tasks, succeeds_on, fails_on, success_priority, reset=reset)
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
        partial(f, *args, **kwargs), succeeds_on, fails_on, success_priority
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


def sequence(tasks: typing.Iterable[TASK] | typing.Callable[[], typing.Iterable[TASK]], ctx: Context) -> CALL_TASK:
    """Run a sequence task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to execute
        state (State): The current state

    Returns:
        CALL_TASK: The task to call
    """
    def _f(reset: bool=False):
        if reset:
            ctx['status'] = TaskStatus.READY
            ctx['cur_task'] = None
            del ctx['it']
        status = ctx.get_or_set('status', TaskStatus.RUNNING)

        if status.is_done:
            return status
        
        # get the iterator
        if 'it' not in ctx:
            if callable(tasks):
                ctx["it"] = iter(tasks())
            else:
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
            cur_status = cur_task(reset)

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
    tasks: typing.Iterable[TASK] | typing.Callable[[], typing.Iterable[TASK]], 
    ctx: Context,
    reset: bool=False
) -> TaskStatus:

    if reset:
        ctx['status'] = TaskStatus.READY
        ctx['cur_task'] = None
        del ctx['it']

    status = ctx.get_or_set('status', TaskStatus.RUNNING)

    if status.is_done:
        return status
    
    # get the iterator
    if 'it' not in ctx:
        if callable(tasks):
            ctx["it"] = iter(tasks())
        else:
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


def selector(tasks: typing.Iterable[TASK] | typing.Callable[[], typing.Iterable[TASK]], ctx: Context) -> CALL_TASK:
    """Create a selector task

    Args:
        tasks (TASK): The tasks to run
        state (State): The state for the selector

    Returns:
        CALL_TASK: The task to call
    """
    def _f(reset: bool=False):
        return _selector(tasks, ctx, reset)

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


def action(task: TASK, *args, **kwargs) -> CALL_TASK:
    """Functional form of action

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute
        state (typing.Dict): The state of execution

    Returns:
        TaskStatus: The status of the result
    """
    def _f(reset: bool=False):
        result = task(*args, reset=reset, **kwargs)
        return result

    return _f


def threadedf(
    task: TASK, ctx: Context, 
    *args, out: SharedBase=None, 
    to_status: TOSTATUS=None, **kwargs
) -> CALL_TASK:
    """Use to wrap the task in a thread"""

    if 'task_id' in ctx and id(task) != ctx['task_id']:

        raise RuntimeError(
            'Task context has been initialized but '
            'the task passed in is does not match'
        )

    def _f(reset: bool=False):
        """Run the task in a thread"""
        def task_wrapper():
            
            result = task(
                *args, reset=reset, 
                **kwargs
            )
            if out is not None:
                out.set(result)
            if to_status is not None:
                status = to_status(result)
            ctx['thread_status'] = status

        if '_thread' not in ctx:
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(target=task_wrapper)
            ctx['_thread'] = t
            t.start()
        
        if t.is_alive():
            return TaskStatus.RUNNING
        
        return ctx['thread_status']
    return _f


def taskf(
    f, *args, out: SharedBase=None, 
    to_status: TOSTATUS=None, **kwargs) -> CALL_TASK:
    """A generic task based on a function

    Args:
        f: The function to exectue
        out (SharedBase, optional): The output to store to. Defaults to None.
        to_status (TOSTATUS, optional): The status converter. If not set will automatically set to "Success". 
            Defaults to None.

    Returns:
        CALL_TASK: The CALL_TASK to execute
    """
    def _f(reset: bool=False):
        result = f(*args, reset=reset, **kwargs)
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
    def _f(reset: bool=False):
        result = task(*args, reset=reset, **kwargs)
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
    def _f(reset: bool=False) -> TaskStatus:
        status = tick(task, reset)
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
    """Use to wrap the task in a thread"""

    def run() -> TaskStatus:
        """Run the task in a thread"""
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


def _stream_model(model: LLM, prompt: LLM_PROMPT, ctx: Context, out: FromMsg, *args, interval: float=1./60, **kwargs):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    for msg in model.stream(prompt, *args, **kwargs):
        ctx['msg'] = msg
        ctx['cur'].append(out(msg))
        time.sleep(interval)
    ctx['thread_status'] = TaskStatus.SUCCESS


def stream_model(
    buffer: Buffer, engine: LLM, prompt: LLM_PROMPT, ctx: Context, out: typing.List[str] | str,
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
    if interval <= 0.0:
        raise ValueError(f'Interval must be greater than 0.0 not {interval}')
    if not isinstance(ctx, Context):
        raise AttributeError(
            f'Context must be of type context not {type(ctx)} '
        )
    out = FromMsg(out)
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['msg'] = None
            ctx['cur'] = []
            ctx['i'] = 0
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_stream_model, args=(
                    engine, prompt, ctx, out, *args
                ), kwargs={'interval': interval, **kwargs}
            )
            t.start()
            ctx['_thread'] = t
        
        if ctx['i'] < len(ctx['cur']):
            buffer.add(*ctx['cur'][ctx['i']:])
            ctx['i'] += len(ctx['cur']) - ctx['i']

        return ctx['thread_status']

    return run


# TODO: Improve Error Handling
def _run_model(model: LLM, prompt: LLM_PROMPT, ctx: Context, out: FromMsg, **kwargs):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    ctx['msg'] = model(prompt, **kwargs)
    ctx['x'] = out(ctx['msg'])
    ctx['thread_status'] = TaskStatus.SUCCESS


def exec_model(
    shared: Shared, engine: LLM, prompt: LLM_PROMPT, ctx: Context, out: typing.List[str] | str,
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
    out = FromMsg(out)
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['msg'] = None
            ctx['x'] = None
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_run_model, args=(
                    engine, prompt, ctx, out
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


def tick(task: TASK, reset: bool=False) -> TaskStatus:
    """Run the task

    Args:
        task (TASK): The task to run

    Returns:
        TaskStatus: The resulting status from the task
    """

    if isinstance(task, Task):
        return task.tick(reset)
    
    return task(reset)


def aslongas(task: TASK, status: TaskStatus=TaskStatus.FAILURE) -> CALL_TASK:
    """Loop unless a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        TaskStatus: The status of the result
    """
    local_reset = False
    def _f(reset: bool=False):
        nonlocal local_reset
        cur_status = task(reset or local_reset)
        local_reset = False
        if cur_status == status:
            if cur_status.is_done:
                local_reset = True
            return TaskStatus.RUNNING
        return cur_status
    return _f


def aslongasf(
    f, *args, status: TaskStatus=TaskStatus.FAILURE, **kwargs
) -> CALL_TASK:
    """Loop unless a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        CALL_TASK: The status of the result
    """
    return aslongas(partial(f, *args, **kwargs), status)


def until(
    task: TASK, 
    status: TaskStatus=TaskStatus.SUCCESS
) -> CALL_TASK:
    """Loop until a condition is met

    Args:
        task (typing.Union[_tasks.Task, typing.Callable[[], TaskStatus]]): the task to execute

    Returns:
        TaskStatus: The status of the result
    """
    local_reset = False

    def _f(reset: bool=False):
        nonlocal local_reset
        cur_status = task(local_reset or reset)
        
        local_reset = False
        if cur_status == status:
            return cur_status

        if cur_status == status:
            if cur_status.is_done:
                local_reset = True
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


def statemachine(f: State | typing.Callable[[], State | TaskStatus], ctx: Context):
    """A state machine "task" allows 

    Args:
        f (State | typing.Callable[[], State  |  TaskStatus]): 
        ctx (Context): 
    """

    def _(reset: bool=False):
        if reset:
            del ctx['cur']
        
        if 'cur' not in ctx:
            if callable(f):
                ctx['cur'] = f()
            else:
                ctx['cur'] = f
        
        updated = ctx['cur']()
        if isinstance(updated, TaskStatus):
            return updated
        else:
            if callable(updated):
                ctx['cur'] = updated()
            else:
                ctx['cur'] = updated

            return TaskStatus.RUNNING

    return _


def statemachinef(f: typing.Callable[[typing.Any], State | TaskStatus], ctx: Context, *args, **kwargs) -> CALL_TASK:
    """Run a callable sequence task

    Args:
        tasks (typing.Iterable[TASK]): The tasks to execute
        state (State): The current state

    Returns:
        CALL_TASK: The task to call
    """
    
    return f(*args, ctx=ctx, **kwargs)



