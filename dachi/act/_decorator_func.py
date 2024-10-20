from . import _functional as F
from ..data import Context, Shared


class TaskF:
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


def condfunc():

    def _(f):
        return TaskF(f, F.conf)
    return _


# def actfunc(data: Shared=None):

#     def _(f):
#         return TaskF(f, F.conf)
#     return _
