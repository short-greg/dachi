from . import _functional as F
from ..store._data import Context, SharedBase
from ..utils import get_member
from ._core import STATE_CALL, TaskStatus
import typing
import functools
from functools import partial
from ..core import Task
from abc import abstractmethod
from ._tasks import (
    Task, 
    Parallel, 
    Sequence,
    Selector, 
    Action,
    Condition,
    StateMachine

)


class TaskFuncBase(object):

    def __init__(
        self, instance=None, 
    ):
        """This is the base function for the 

        Args:
            instance (optional): The instance. Defaults to None.
            is_method (bool, optional): Whether the function is a method. Defaults to False.
        """
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
        if self._instance is None:
            return args[0], args[1:]
        return self._instance, args
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> TaskStatus:
        pass
        

class SequenceFunc(TaskFuncBase):
    """CompositeFunc is used for decorating sequences and selectors
    """

    def __init__(
        self, f, 
        instance=None
    ):
        """Create a composite function

        Args:
            f: The function to execute
            base_f: The base composite function to use (i.e. selector or sequence)
            ctx (Context, optional): The context for the function. Defaults to None.
            instance (optional): The instance. Defaults to None.
        """
        self.f = f
        self._instance = instance

    def __call__(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        instance, args = self.get_instance(args)

        return Sequence(
            partial(self.f, instance, *args, **kwargs)
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
        
        task = SequenceFunc(self.f, self.base_f, instance)
        instance.__dict__[self.f.__name__] = task
        return task


class SelectorFunc(TaskFuncBase):
    """CompositeFunc is used for decorating sequences and selectors
    """

    def __init__(
        self, f, 
        instance=None
    ):
        """Create a composite function

        Args:
            f: The function to execute
            base_f: The base composite function to use (i.e. selector or sequence)
            ctx (Context, optional): The context for the function. Defaults to None.
            instance (optional): The instance. Defaults to None.
        """
        self.f = f
        self._instance = instance

    def __call__(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        instance, args = self.get_instance(args)

        return Selector(
            partial(self.f, instance, *args, **kwargs)
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
        
        task = Selector(self.f, self.base_f, instance)
        instance.__dict__[self.f.__name__] = task
        return task


class StateMachineFunc(TaskFuncBase):
    """CompositeFunc is used for decorating sequences and selectors
    """

    def __init__(
        self, f, init_state: STATE_CALL, 
        instance=None
    ):
        """Create a composite function

        Args:
            f: The function to execute
            base_f: The base composite function to use (i.e. selector or sequence)
            ctx (Context, optional): The context for the function. Defaults to None.
            instance (optional): The instance. Defaults to None.
        """
        self.f = f
        self.init_state = init_state
        self._instance = instance

    def __call__(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        # ctx = _get(instance, self._ctx, ctx)
        instance, args = self.get_instance(args)

        return StateMachine(
            partial(self.f, instance, *args, **kwargs)
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
        super().__init__(instance)
        self.f = f
        self._instance = instance
        self.succeeds_on = succeeds_on
        self.fails_on = fails_on
        self.success_priority = success_priority
        self.preempt = preempt

    def __call__(self, *args, **kwargs) -> Task:
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        instance, args = self.get_instance(args)
        return Parallel(
            partial(
                self.f, instance, *args, **kwargs
            ), self.fails_on, self.succeeds_on,
            self.success_priority, self.preempt
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
        instance.__dict__[
            self.f.__name__
        ] = task
        return task


class CondFunc(TaskFuncBase):
    """TaskF is used for function decorators to 
    """

    def __init__(
        self, f, instance=None
    ):
        """Create a conditional function

        Args:
            f: The conditional function
            instance (optional): _description_. Defaults to None.
        """
        self.f = f
        self._instance = instance

    def __call__(self, *args, **kwargs):
        """Get the task from the function

        Returns:
            The task to exeucte
        """
        # This method will handle "task" and correctly bind to the instance

        instance, args = self.get_instance(args)
        return Condition(
            partial(
                self.f, instance, *args, **kwargs 
            )
        )

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """
        if self.f.__name__ in instance.__dict__:
            return instance.__dict__[self.f.__name__]
        task = CondFunc(self.f, True, instance)
        instance.__dict__[self.f.__name__] = task
        return task
            

class ActionFunc(TaskFuncBase):
    """
    """

    def __init__(
        self, f, instance=None, 
        out: typing.Union[SharedBase, str]=None, 
        to_status: F.TOSTATUS=None
    ):
        """A general decorator for a Task Function

        Args:
            f: The task function
            instance (_type_, optional): The instance for the task. Defaults to None.
            out (typing.Union[SharedBase, str], optional): The processor for the result. Defaults to None.
            to_status (F.TOSTATUS, optional): How to convert to a status. Defaults to None.
        """
        self.f = f
        self._instance = instance
        self._to_status = to_status
        self._out = out

    def tick(self, *args, **kwargs):
        """Get the "task" for the function

        Returns:
            The task
        """

        instance, args = self.get_instance(args)
        return Action(
            partial(self.f, instance, *args, **kwargs), self._to_status, self._out
        )

    def __get__(self, instance, owner):
        """Add the task to the instance if not already there
        """
        if id(self.f) in instance.__dict__:
            return instance.__dict__[id(self.f)]
        
        task = ActionFunc(self.f, True, instance, self._out, self._to_status)
        instance.__dict__[id(self.f)] = task
        return task


def sequence():
    """Decorate a sequence function that yields tasks

    Args:
        ctx (Context, optional): The context. Defaults to None.
        is_method (bool, optional): Whether it is a method. Defaults to False.
    Returns: The task
    """
    def _(f):
        return SequenceFunc(f)
    return _


def statemachine():
    """Decorate a state machine function that yields tasks

    Args:
        ctx (Context, optional): The context for the task. Defaults to None.

    Returns: the task
    """
    def _(f):
        return StateMachineFunc(f, F.statemachinef)
    return _


def selector():
    """Decorate a selector function that yields tasks

    Args:
        ctx (Context, optional): The context for the task. Defaults to None.

    Returns: the task
    """
    def _(f):
        return SelectorFunc(f)
    return _

fallback = selector


def parallel(
    succeeds_on: int=-1, 
    fails_on: int=1, 
    success_priority: bool=True,
    preempt: bool=False, 
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
            success_priority, preempt
        )
    return _


def cond():
    """Decorate a conditional function that returns True or False

    Args:
        is_method (bool, optional): Whether it is a method. Defaults to False.
    """
    def _(f):
        return CondFunc(f)
    return _


def action(out: SharedBase=None, to_status: F.TOSTATUS=None):
    """Decorate a general task function

    Args:
        out (SharedBase, optional): The processor for the result. Defaults to None.
        to_status (F.TOSTATUS, optional): The converter to change the result to a status. Defaults to None.
        is_method (bool, optional): Whether it is a method or not. Defaults to False.
    """
    def _(f):
        t = ActionFunc(
            f, None, F.taskf, out, to_status
        )
        t.__call__ = functools.wraps(f, t.__call__)
        return t
    return _
