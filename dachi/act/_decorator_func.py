from . import _functional as F
from ..data import Context


class TaskF:

    def __init__(
        self, f, base_f, ctx: Context=None, instance=None
    ):
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
        return self.f(self.instance, *args, **kwargs)

    def __get__(self, instance, owner):
        # Bind the decorator to the instance (used for instance method binding)

        return TaskF(self.f, instance)


def _get_ctx(self, _ctx, ctx):
    
    ctx = ctx or _ctx
    if ctx is None:
        raise ValueError('Context has not been defined')
    elif isinstance(ctx, str):
        return object.__getattribute__(self, ctx)
    return ctx


def sequencefunc(ctx: Context=None):

    def _(f):

        return TaskF(f, F.sequencef, ctx)
    return _


def selectorfunc(ctx: Context=None):

    def _(f):
        return TaskF(f, F.selectorf, ctx)
    return _


def parallelfunc(ctx: Context=None):

    def _(f):
        return TaskF(f, F.parallelf, ctx)
    return _


# make it possible to define the context

# using "task" will result in using the
# functional task
