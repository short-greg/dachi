from . import _functional as F
from ..data import Context, SharedBase
import typing
import functools


class CompositeFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, base_f, ctx: 
        Context=None, instance=None
    ):
        """

        Args:
            f (_type_): _description_
            base_f (_type_): _description_
            ctx (Context, optional): _description_. Defaults to None.
            instance (_type_, optional): _description_. Defaults to None.
        """
        self.f = f
        self.base_f = base_f
        self.instance = instance
        self._ctx = ctx

    def task(self, *args, _ctx: Context=None, **kwargs):
        # This method will handle "task" and correctly bind to the instance

        ctx = _get_ctx(self.instance, self._ctx, _ctx)  
        if self.instance is None:
            return self.base_f(self.f, ctx, *args, **kwargs)
        
        return self.base_f(self.f, ctx, self.instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        # This handles the original method call
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):

        task = CompositeFunc(self.f, self.base_f, self._ctx, instance)
        instance.__dict__[self.f.__name__] = task
        return task


class ParallelFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, succeeds_on=-1, fails_on=1, success_priority=True,
        instance=None
    ):
        """

        Args:
            f (_type_): _description_
            base_f (_type_): _description_
            ctx (Context, optional): _description_. Defaults to None.
            instance (_type_, optional): _description_. Defaults to None.
        """
        self.f = f
        self.instance = instance
        self.succeeds_on = succeeds_on
        self.fails_on = fails_on
        self.success_priority = success_priority

    def task(self, *args, **kwargs):

        if self.instance is None:
            return F.parallelf(
                self.f, *args, succeeds_on=self.succeeds_on, 
                fails_on=self.fails_on, success_priority=self.success_priority, **kwargs
            )
        
        return F.parallelf(
            self.f, self.instance, *args, 
            succeeds_on=self.succeeds_on, fails_on=self.fails_on, success_priority=self.success_priority, **kwargs
        )

    def __call__(self, *args, **kwargs):
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):

        task = ParallelFunc(
            self.f, self.succeeds_on, self.fails_on, 
            self.success_priority, instance
        )
        instance.__dict__[self.f.__name__] = task
        return task


class CondFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, instance=None
    ):
        """

        Args:
            f (_type_): _description_
            base_f (_type_): _description_
            ctx (Context, optional): _description_. Defaults to None.
            instance (_type_, optional): _description_. Defaults to None.
        """
        self.f = f
        self._instance = instance

    def task(self, *args, **kwargs):
        # This method will handle "task" and correctly bind to the instance

        if self._instance is None:
            return F.condf(self.f, *args, **kwargs)
        return F.condf(self.f, self._instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):

        task = CondFunc(self.f, instance)
        instance.__dict__[self.f.__name__] = task
        return task


def _get(obj, key):
    return object.__getattribute__(obj, key) if isinstance(key, str) else key
            

class TaskFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, instance=None, out: typing.Union[SharedBase, str]=None, 
        to_status: F.TOSTATUS=None
    ):
        """

        Args:
            f (_type_): _description_
            base_f (_type_): _description_
            ctx (Context, optional): _description_. Defaults to None.
            instance (_type_, optional): _description_. Defaults to None.
        """
        self.f = f
        self._instance = instance
        self._to_status = to_status
        self._out = out

    def task(self, *args, **kwargs):

        if self._instance is None:
            out = self._out
            to_status = self._to_status
            return F.taskf(self.f, *args, out=out, to_status=to_status, **kwargs)
        else:
            to_status = _get(self._instance, self._to_status)
            out = _get(self._instance, self._out)
            return F.taskf(
                self.f, self._instance, out=out, to_status=to_status, *args, **kwargs
            )

    def __call__(self, *args, **kwargs):
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):

        task = TaskFunc(self.f, instance, self._out, self._to_status)
        instance.__dict__[self.f.__name__] = task
        return task


def _get_ctx(self, _ctx, ctx):
    
    ctx = ctx or _ctx
    if ctx is None:
        raise ValueError('Context has not been defined')
    elif isinstance(ctx, str):
        return object.__getattribute__(self, ctx)
    return ctx


def sequencefunc(ctx: Context=None):

    def _(f):

        return CompositeFunc(f, F.sequencef, ctx)
    return _


def selectorfunc(ctx: Context=None):

    def _(f):
        return CompositeFunc(f, F.selectorf, ctx)
    return _


def parallelfunc(succeeds_on: int=-1, fails_on: int=1, success_priority: bool=True):

    def _(f):
        return ParallelFunc(f, succeeds_on, fails_on, success_priority)
    return _


def condfunc():

    def _(f):
        return CondFunc(f, F.condf)
    return _


def taskfunc(out: SharedBase=None, to_status: F.TOSTATUS=None):

    def _(f):
        t = TaskFunc(f, F.taskf, out, to_status)
        t.__call__ = functools.wraps(f, t.__call__)
        return t
    return _
