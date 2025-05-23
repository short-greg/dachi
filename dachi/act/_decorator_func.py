from . import _functional as F
from ..store._data import Context, SharedBase
from ..utils import get_member
from ._core import STATE_CALL
import typing
import functools
from functools import partial


class TaskFuncBase(object):

    def __init__(
        self, instance=None, 
        is_method: bool=False
    ):
        """This is the base function for the 

        Args:
            instance (optional): The instance. Defaults to None.
            is_method (bool, optional): Whether the function is a method. Defaults to False.
        """
        
        self._is_method = is_method
        self._instance = instance

    def get_instance(self, args):
        """Get the instance

        Args:
            instance: _description_
            is_method (bool): whether the function is a method
            args: The args for the insta

        Returns:
            the instance and args
        """
        if self._is_method:
            if self._instance is None:
                return args[0], args[1:]
            return self._instance, args
        
        return None, args
        

class CompositeFunc(TaskFuncBase):
    """CompositeFunc is used for decorating sequences and selectors
    """

    def __init__(
        self, f, base_f, is_method: bool=False, 
        instance=None
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
        self._instance = instance
        self._is_method = is_method

    def __call__(self, ctx: Context, *args, **kwargs):
        """Get the task for the function

        Args:
            _ctx (Context, optional): An override for the context. Defaults to None.

        Returns:
            The task 
        """
        # This method will handle "task" and correctly bind to the instance

        instance, args = self.get_instance(args)
        
        # ctx = _get(instance, self._ctx, ctx)  
        if instance is None:
            return self.base_f(self.f, ctx, ctx, *args, **kwargs)
        
        return self.base_f(
            self.f, ctx, instance, ctx, *args, **kwargs)

    def __get__(self, instance, owner):
        """_summary_

        Args:
            instance: The instance for the function
            owner: 

        Returns:
            TaskFunc: the task
        """

        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        
        task = CompositeFunc(self.f, self.base_f, True, instance)
        instance.__dict__[self.f.__name__] = task
        return task


class StateMachineFunc(TaskFuncBase):
    """CompositeFunc is used for decorating sequences and selectors
    """

    def __init__(
        self, f, init_state: STATE_CALL, 
        is_method: bool=False, 
        instance=None
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
        self.init_state = init_state
        self._instance = instance
        self._is_method = is_method
        # self._ctx = ctx

    def __call__(self, ctx: Context, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        # ctx = _get(instance, self._ctx, ctx)
        instance, args = self.get_instance(args)

        if instance is None:
            f = partial(
                self.f, instance, ctx, *args, **kwargs)
        else:
            f = partial(self.f, instance, ctx, *args, **kwargs)

        return F.statemachinef(
            f, ctx, *args, init_state=self.init_state, **kwargs
        )

    def __get__(self, instance, owner):
        """_summary_

        Args:
            instance: The instance for the function
            owner: 

        Returns:
            TaskFunc: the task
        """
        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        
        task = StateMachineFunc(
            self.f, self.base_f, 
            True, instance
        )
        instance.__dict__[self.f.__name__] = task
        return task


class ParallelFunc(TaskFuncBase):
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, 
        succeeds_on=-1, 
        fails_on=1, 
        success_priority=True,
        preempt: bool=False,
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
        self._instance = instance
        self.succeeds_on = succeeds_on
        self.fails_on = fails_on
        self._is_method = is_method
        self.success_priority = success_priority
        self.preempt = preempt

    def __call__(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        instance, args = self.get_instance(args)
        if instance is None:
            return F.parallelf(
                self.f, *args, succeeds_on=self.succeeds_on, 
                fails_on=self.fails_on, success_priority=self.success_priority, preempt=self.preempt, **kwargs
            )
        
        return F.parallelf(
            self.f, instance, *args, 
            succeeds_on=self.succeeds_on, 
            fails_on=self.fails_on, 
            success_priority=self.success_priority, 
            preempt=self.preempt, **kwargs
        )

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """

        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        
        task = ParallelFunc(
            self.f, self.succeeds_on, self.fails_on, 
            self.success_priority, self.preempt, 
            True, instance
        )
        instance.__dict__[self.f.__name__] = task
        return task


class CondFunc(TaskFuncBase):
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
        self._instance = instance
        self._is_method = is_method

    def __call__(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        # This method will handle "task" and correctly bind to the instance

        instance, args = self.get_instance(args)

        if instance is None:
            return F.condf(self.f, *args, **kwargs)
        return F.condf(self.f, instance, *args, **kwargs)

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """
        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        task = CondFunc(self.f, True, instance)
        instance.__dict__[self.f.__name__] = task
        return task


def _get_str(obj, key):
    """_summary_

    Args:
        obj : 
        key: 

    Returns:
        str: 
    """
    return get_member(obj, key) if isinstance(key, str) else key
            

class TaskFunc(TaskFuncBase):
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
        self._instance = instance
        self._is_method = is_method
        self._to_status = to_status
        self._out = out

    def __call__(self, *args, **kwargs):
        """Get the "task" for the function

        Returns:
            The task
        """

        instance, args = self.get_instance(args)

        if instance is None:
            out = self._out
            to_status = self._to_status
            return F.taskf(self.f, *args, out=out, to_status=to_status, **kwargs)
        else:
            to_status = _get_str(instance, self._to_status)
            out = _get_str(instance, self._out)
            return F.taskf(
                self.f, instance, 
                *args, out=out, to_status=to_status, **kwargs
            )

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """
        if id(self.f) in instance.__dict__:
            return instance.__dict__[id(self.f)]
        
        task = TaskFunc(self.f, True, instance, self._out, self._to_status)
        instance.__dict__[id(self.f)] = task
        return task


def sequencefunc(is_method: bool=False):
    """Decorate a sequence function that yields tasks

    Args:
        ctx (Context, optional): The context. Defaults to None.
        is_method (bool, optional): Whether it is a method. Defaults to False.
    Returns: The task
    """
    def _(f):
        return CompositeFunc(f, F.sequencef, is_method)
    return _


def sequencemethod():
    """Decorate a sequence method that yields tasks

    Args:
        ctx (Context): 

    Returns: The task
    """
    return sequencefunc(True)


def statemachinefunc():
    """Decorate a state machine function that yields tasks

    Args:
        ctx (Context, optional): The context for the task. Defaults to None.

    Returns: the task
    """
    def _(f):
        return StateMachineFunc(f, F.statemachinef)
    return _


def statemachinemethod():
    """Decorate a state machine method that yields tasks

    Args:
        ctx (Context): The context for the task

    Returns: the task
    """
    return statemachinefunc(True)


def selectorfunc(is_method: bool=False):
    """Decorate a selector function that yields tasks

    Args:
        ctx (Context, optional): The context for the task. Defaults to None.

    Returns: the task
    """
    def _(f):
        return CompositeFunc(f, F.selectorf, is_method)
    return _


def selectormethod():
    """Decorate a selector method that yields tasks

    Args:
        ctx (Context): The context for the task

    Returns: the task
    """
    return selectorfunc(True)


fallbackfunc = selectorfunc
fallbackmethod = selectormethod


def parallelfunc(
    succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True,
    preempt: bool=False, 
    is_method: bool=False
):
    """Decorate a parallel function that yields tasks

    Args:
        succeeds_on (int, optional): Number required to succeed. Defaults to -1.
        fails_on (int, optional): Number required to fail. Defaults to 1.
        success_priority (bool, optional): Whether success prioritized over failure if equal. Defaults to True.
        is_method (bool, optional): Whether it is a method. Defaults to False.
    """

    def _(f):
        return ParallelFunc(
            f, succeeds_on, fails_on, 
            success_priority, preempt, is_method
        )
    return _


def parallelmethod(
    succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True,
    preempt: bool=False
):
    """Decorate a parallel method that yields tasks

    Args:
        succeeds_on (int, optional): Number required to succeed. Defaults to -1.
        fails_on (int, optional): Number required to fail. Defaults to 1.
        success_priority (bool, optional): Whether success prioritized over failure if equal. Defaults to True.
    """
    return parallelfunc(
        succeeds_on, fails_on, success_priority, preempt, True
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
