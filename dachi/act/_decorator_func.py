from . import _functional as F
from ..data import Context, SharedBase
from ..utils import get_member
import typing
import functools


def get_instance(instance, is_method, args):
    """Get the instance

    Args:
        instance: _description_
        is_method (bool): whether the function is a method
        args: The args for the insta

    Returns:
        the instance and args
    """
    if is_method and instance is None:
        return args[0], args[1:]
    
    return instance, args
    

class CompositeFunc:
    """CompositeFunc is used for decorating sequences and selectors
    """

    def __init__(
        self, f, base_f, ctx: 
        Context=None, is_method: bool=False, instance=None
    ):
        """Create a composite function

        Args:
            f: The function to execute
            base_f: The base composite function to use (i.e. selector or sequence)
            ctx (Context, optional): The context for the function. Defaults to None.
            is_method (bool): Whether it is a method or not
            instance (optional): The instance. Defaults to None.
        """
        self.f = f
        self.base_f = base_f
        self.instance = instance
        self.is_method = is_method
        self._ctx = ctx

    def task(self, *args, _ctx: Context=None, **kwargs):
        """Get the task for the function

        Args:
            _ctx (Context, optional): An override for the context. Defaults to None.

        Returns:
            The task 
        """
        # This method will handle "task" and correctly bind to the instance

        instance, args = get_instance(self.instance, self.is_method, args)
        
        ctx = _get(instance, self._ctx, _ctx)  
        if instance is None:
            return self.base_f(self.f, ctx, *args, **kwargs)
        
        return self.base_f(self.f, ctx, instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Execute the function

        Returns: The output of the function
        """
        # This handles the original method call
        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is not None:
            return self.f(instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):

        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        
        task = CompositeFunc(self.f, self.base_f, self._ctx, True, instance)
        instance.__dict__[self.f.__name__] = task
        return task


class ParallelFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, succeeds_on=-1, fails_on=1, success_priority=True,
        is_method: bool=False,
        instance=None
    ):
        """A Parallel Function decorator executes a parallel task

        Args:
            f : The function to execute
            ctx (Context, optional): The context to use. Defaults to None.
            succeeds_on (int): the number of successes before regarded as success
            fails_on (int): the number of failures before it is regarded as failing
            success_priority (int): whether it success is prioritized over failure
            is_method (bool): 
            instance (optional): The instance. Defaults to None.
        """
        self.f = f
        self.instance = instance
        self.succeeds_on = succeeds_on
        self.fails_on = fails_on
        self.is_method = is_method
        self.success_priority = success_priority

    def task(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is None:
            return F.parallelf(
                self.f, *args, succeeds_on=self.succeeds_on, 
                fails_on=self.fails_on, success_priority=self.success_priority, **kwargs
            )
        
        return F.parallelf(
            self.f, instance, *args, 
            succeeds_on=self.succeeds_on, fails_on=self.fails_on, success_priority=self.success_priority, **kwargs
        )

    def __call__(self, *args, **kwargs):
        """Execute the function

        Returns: The output of the function
        """
        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is not None:
            return self.f(instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """

        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        
        task = ParallelFunc(
            self.f, self.succeeds_on, self.fails_on, 
            self.success_priority, True, instance
        )
        instance.__dict__[self.f.__name__] = task
        return task


class CondFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, is_method: bool=False, instance=None
    ):
        """Create a conditional function

        Args:
            f: The conditional function
            instance (optional): _description_. Defaults to None.
        """
        self.f = f
        self.instance = instance
        self.is_method = is_method

    def task(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        # This method will handle "task" and correctly bind to the instance

        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is None:
            return F.condf(self.f, *args, **kwargs)
        return F.condf(self.f, instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Execute the function

        Returns: The output of the function
        """
        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is not None:
            return self.f(instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """
        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        task = CondFunc(self.f, True, instance)
        instance.__dict__[self.f.__name__] = task
        return task


def _get_str(obj, key):

    return get_member(obj, key) if isinstance(key, str) else key
            

class TaskFunc:
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, is_method: bool=False, instance=None, 
        out: typing.Union[SharedBase, str]=None, 
        to_status: F.TOSTATUS=None
    ):
        """A general decorator for a Task Function

        Args:
            f: The task function
            is_method (bool, optional): Whether it is a method or not. Defaults to False.
            instance (_type_, optional): The instance for the task. Defaults to None.
            out (typing.Union[SharedBase, str], optional): The processor for the result. Defaults to None.
            to_status (F.TOSTATUS, optional): How to convert to a status. Defaults to None.
        """
        self.f = f
        self.instance = instance
        self.is_method = is_method
        self._to_status = to_status
        self._out = out

    def task(self, *args, **kwargs):
        """Get the "task" for the function

        Returns:
            The task
        """

        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is None:
            out = self._out
            to_status = self._to_status
            return F.taskf(self.f, *args, out=out, to_status=to_status, **kwargs)
        else:
            to_status = _get_str(instance, self._to_status)
            out = _get_str(instance, self._out)
            return F.taskf(
                self.f, instance, out=out, to_status=to_status, 
                *args, **kwargs
            )

    def __call__(self, *args, **kwargs):
        instance, args = get_instance(self.instance, self.is_method, args)

        if instance is not None:
            return self.f(instance, *args, **kwargs)
        return self.f(*args, **kwargs)

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """
        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        
        task = TaskFunc(self.f, True, instance, self._out, self._to_status)
        instance.__dict__[self.f.__name__] = task
        return task


def _get(self, override, base):
    """Get the an override value if defined

    Args:
        override: The override value
        base: The base value

    Raises:
        ValueError: If neither value has been set

    Returns:
        Any
    """
    base = base or override
    if base is None:
        raise ValueError('Value has not been defined')
    elif isinstance(base, str):
        return get_member(self, base)
    return base


def sequencefunc(ctx: Context=None, is_method: bool=False):
    """Decorate a sequence function that yields tasks

    Args:
        ctx (Context, optional): The context. Defaults to None.
        is_method (bool, optional): Whether it is a method. Defaults to False.
    Returns: The task
    """
    def _(f):
        return CompositeFunc(f, F.sequencef, ctx, is_method)
    return _


def sequencemethod(ctx: Context):
    """Decorate a sequence method that yields tasks

    Args:
        ctx (Context): 

    Returns: The task
    """
    return sequencefunc(ctx, True)


def selectorfunc(ctx: Context=None):
    """Decorate a selector function that yields tasks

    Args:
        ctx (Context, optional): The context for the task. Defaults to None.

    Returns: the task
    """
    def _(f):
        return CompositeFunc(f, F.selectorf, ctx)
    return _


def selectormethod(ctx: Context):
    """Decorate a selector method that yields tasks

    Args:
        ctx (Context): The context for the task

    Returns: the task
    """
    return selectorfunc(ctx, True)


fallbackfunc = selectorfunc
fallbackmethod = selectormethod


def parallelfunc(
    succeeds_on: int=-1, fails_on: int=1, 
    success_priority: bool=True, is_method: bool=False
):
    """Decorate a parallel function that yields tasks

    Args:
        succeeds_on (int, optional): Number required to succeed. Defaults to -1.
        fails_on (int, optional): Number required to fail. Defaults to 1.
        success_priority (bool, optional): Whether success prioritized over failure if equal. Defaults to True.
        is_method (bool, optional): Whether it is a method. Defaults to False.
    """

    def _(f):
        return ParallelFunc(f, succeeds_on, fails_on, success_priority, is_method)
    return _


def parallelmethod(
    succeeds_on: int=-1, fails_on: int=1, 
    success_priority: bool=True
):
    """Decorate a parallel method that yields tasks

    Args:
        succeeds_on (int, optional): Number required to succeed. Defaults to -1.
        fails_on (int, optional): Number required to fail. Defaults to 1.
        success_priority (bool, optional): Whether success prioritized over failure if equal. Defaults to True.
    """
    return parallelfunc(
        succeeds_on, fails_on, success_priority, True
    )


def condfunc(is_method: bool=False):
    """Decorate a conditional function that returns True or False

    Args:
        is_method (bool, optional): Whether it is a method. Defaults to False.
    """
    def _(f):
        return CondFunc(f, is_method)
    return _


def condmethod():
    """Decorate a conditional method
    """
    return condfunc(True)


def taskfunc(out: SharedBase=None, to_status: F.TOSTATUS=None, is_method: bool=False):
    """Decorate a general task function

    Args:
        out (SharedBase, optional): The processor for the result. Defaults to None.
        to_status (F.TOSTATUS, optional): The converter to change the result to a status. Defaults to None.
        is_method (bool, optional): Whether it is a method or not. Defaults to False.
    """
    def _(f):
        t = TaskFunc(f, is_method, F.taskf, out, to_status)
        t.__call__ = functools.wraps(f, t.__call__)
        return t
    return _


def taskmethod(out: SharedBase=None, to_status: F.TOSTATUS=None):
    """Decorate a general task method

    Args:
        out (SharedBase, optional): The processor for the result. Defaults to None.
        to_status (F.TOSTATUS, optional): The converter to change the result to a status. Defaults to None.
    """
    return taskfunc(
        out, to_status, True
    )
