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

        ctx = _get_ctx(self, self._ctx, _ctx)  
        if self.instance is None:
            return self.base_f(self.f, ctx, *args, **kwargs)
        return self.base_f(self.f, ctx, self.instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        # This handles the original method call
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        # Bind the decorator to the instance (used for instance method binding)

        return CompositeFunc(self.f, instance)


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
        # This handles the original method call
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        # Bind the decorator to the instance (used for instance method binding)

        return CondFunc(self.f, instance)


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

        # if out is not None:
        #     out.set(result)
        # if to_status is not None:
        #     return to_status(result)
        # return result

    def __call__(self, *args, **kwargs):
        if self._instance is not None:
            return self.f(self._instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):

        return TaskFunc(self.f, instance, self._out, self._to_status)


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


def parallelfunc(ctx: Context=None):

    def _(f):
        return CompositeFunc(f, F.parallelf, ctx)
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


# def actfunc(data: Shared=None):

#     def _(f):
#         return TaskF(f, F.conf)
#     return _
