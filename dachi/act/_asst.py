# 1st party
import typing
import threading
import time

import pydantic

# local
from ._core import (
    TaskStatus, Task, State
)
from ..store._data import Context, SharedBase
from ._core import TOSTATUS
from ..store._data import Buffer,Shared
from ..asst import LLM, Assistant, LLM_PROMPT, FromMsg, Op, Threaded, OutConv
import itertools
from ._tasks import Action
from ..utils import Args


TASK = typing.Union[
    Task, typing.Callable[[typing.Dict], TaskStatus]]
CALL_TASK = typing.Callable[[],TaskStatus]


PARALLEL = typing.Callable[[typing.Iterable[Task], int, int, bool], TaskStatus]


def _stream_op(
    op: Op | Threaded, buffer: Buffer, ctx: Context, args: Args,
    out: OutConv=None,
    interval: float=1./60
):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    try:
        for cur in op.stream(
            *args.args, _out=out, **args.kwargs
        ):
            ctx['cur'] = cur
            buffer.add(cur)
            time.sleep(interval)
        ctx['thread_status'] = TaskStatus.SUCCESS
    except Exception as e:
        ctx['error'] = e

def stream_op(    
    op: Op | Threaded, 
    buffer: Buffer, ctx: Context, 
    args: Args=None,
    out: OutConv=None, 
    interval: float=1./60
):
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    args = args or Args()
    if interval <= 0.0:
        raise ValueError(f'Interval must be greater than 0.0 not {interval}')
    if not isinstance(ctx, Context):
        raise AttributeError(
            f'Context must be of type context not {type(ctx)} '
        )
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['cur'] = []
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_stream_op, args=(
                    op, buffer, ctx, 
                    args,
                ), kwargs={
                    'interval': interval, 
                    'out': out,
                }
            )
            t.start()
            ctx['_thread'] = t
            return TaskStatus.RUNNING
        print(ctx['cur'])

        if 'error' in ctx:
            raise ctx['error']
        
        return ctx['thread_status']

    return run


# TODO: Improve Error Handling
def _run_op(
    op: Op | Threaded, 
    shared: SharedBase, 
    ctx: Context, 
    args: Args,
    out: OutConv=None
):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    try:
        ctx['res'] = op(
            *args.args, _out=out, **args.kwargs
        )
        shared.data = ctx['res']
        ctx['thread_status'] = TaskStatus.SUCCESS
    except Exception as e:
        ctx['error'] = e


def run_op(
    op: Op | Threaded, 
    shared: SharedBase,
    ctx: Context, 
    args: Args=None,
    out=None, 
) -> CALL_TASK:
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    args = args or Args()
    def run() -> TaskStatus:
        if '_thread' not in ctx:
            ctx['msg'] = None
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_run_op, args=(
                    op, shared, 
                    ctx, args
                ), kwargs={
                    'out': out
                }
            )
            t.start()
            ctx['_thread'] = t
            return TaskStatus.RUNNING
        if 'error' in ctx:
            print('Found an error!')
            raise ctx['error']
        return ctx['thread_status']
    return run


def _stream_assist(
    model: LLM, 
    prompt: LLM_PROMPT, 
    buffer: Buffer, 
    ctx: Context, 
    out: FromMsg, 
    args: Args,
    _interval: float=1./60, 
):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    try:
        for msg in model.stream(prompt, *args.args, **args.kwargs):
            ctx['msg'] = msg
            buffer.add(out(msg))
            time.sleep(_interval)
        ctx['thread_status'] = TaskStatus.SUCCESS
    except Exception as e:
        ctx['error'] = e


def stream_assist(
    model: LLM, 
    prompt: LLM_PROMPT, 
    buffer: Buffer, 
    ctx: Context, 
    out: typing.List[str] | str, 
    args: Args=None, 
    interval: float=1./60, 
) -> CALL_TASK:
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    args = args or Args()
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
                target=_stream_assist, args=(
                    model, prompt, 
                    buffer, ctx, out, args
                ), kwargs={
                    'interval': interval, 
                }
            )
            t.start()
            ctx['_thread'] = t
            return TaskStatus.RUNNING

        if 'error' in ctx:
            print('Raising error')
            raise ctx['error']
        return ctx['thread_status']

    return run


# TODO: Improve Error Handling
def _run_assist(
    model: LLM, prompt: LLM_PROMPT, 
    shared: SharedBase, 
    ctx: Context, out: typing.List[str] | str,
    args: Args
):
    """Run periodically to update the status

    Args:
        task (TASK): The task to run
        ctx (Context): The context
        interval (float, optional): The interval to run at. Defaults to 1./60.
    """
    try:
        ctx['msg'] = model(prompt, *args.args, **args.kwargs)
        shared.data = out(ctx['msg'])
        ctx['thread_status'] = TaskStatus.SUCCESS
    except Exception as e:
        ctx['error'] = e


def run_assist(
    model: LLM, prompt: LLM_PROMPT, 
    shared: SharedBase, 
    ctx: Context, out: typing.List[str] | str,
    args: Args=None
) -> CALL_TASK:
    """Execute the AI model in a thread

    Args:
        shared (Shared): THe shared
        engine (AIModel): The model to use
        ctx (Context): The context to use for maintaining state

    Returns:
        CALL_TASK
    """
    args = args or Args()
    out = FromMsg(out)
    def run() -> TaskStatus:
        if '_thread' not in ctx:

            ctx['msg'] = None
            ctx['x'] = None
            ctx['thread_status'] = TaskStatus.RUNNING
            t = threading.Thread(
                target=_run_assist, args=(
                    model, prompt, shared, ctx, out,
                    args
                )
            )
            t.start()
            ctx['_thread'] = t
            return TaskStatus.RUNNING

        if 'error' in ctx:
            raise ctx['error']
        return ctx['thread_status']

    return run
