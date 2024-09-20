from . import _tasks as tasks
from ._core import Task, TaskStatus
import typing
from functools import wraps
from functools import partial
from typing import Self
from abc import abstractmethod
from . import _functional
from ._core import TaskFunc, SUCCESS, FAILURE



class CondFunc(TaskFunc, tasks.Condition):

    def __call__(self, *args, **kwargs) -> TaskStatus:
        result = self._exec(*args, **kwargs)
        return TaskStatus.from_bool(result)


class ActionFunc(TaskFunc, tasks.Action):

    def __call__(self, *args, **kwargs) -> TaskStatus:
        return self._exec(*args, **kwargs)


class SequenceFunc(TaskFunc, tasks.Sequence):

    def __init__(self, f, is_method: bool=False, instance=None):
        super().__init__(f, is_method, instance)
        self._state = {}
    
    def __call__(self, *args, **kwargs) -> TaskStatus:
        
        result = self._exec(*args, **kwargs)
        return _functional.sequence(result, self._state)
    
    def reset(self):
        
        super().__init__()
        self._state = {}


class SelectorFunc(TaskFunc):

    def __init__(self, f, is_method: bool=False, instance=None):
        super().__init__(f, is_method, instance)
        self._state = {}

    def __call__(self, *args, **kwargs) -> TaskStatus:
        
        result = self._exec(*args, **kwargs)
        return _functional.selector(
            result, self._state
        )
    
    def reset(self):
        
        super().__init__()
        self._state = {}


class UntilFunc(TaskFunc, tasks.Until):

    def __init__(self, f, is_method: bool=False, instance=None, status: TaskStatus=SUCCESS):
        super().__init__(f, is_method, instance)
        self._state = {}
        self._status = status

    def __call__(self, *args, **kwargs) -> TaskStatus:
        
        result = self._exec(*args, **kwargs)
        return _functional.until(
            result, self._state, self._status
        )
    
    def reset(self):
        
        super().__init__()
        self._state = {}


class UnlessFunc(TaskFunc):

    def __init__(self, f, is_method: bool=False, instance=None, status: TaskStatus=FAILURE):
        super().__init__(f, is_method, instance)
        self._state = {}
        self._status = status

    def __call__(self, *args, **kwargs) -> TaskStatus:
        
        result = self._exec(*args, **kwargs)
        return _functional.unless(
            result, self._state, self._status
        )
    
    def reset(self):
        
        super().__init__()
        self._state = {}


class ParallelFunc(TaskFunc):

    def __init__(
        self, iterable: typing.Callable[[typing.Any], bool], is_method: bool=False, 
        *args, fails_on: int=1, 
        succeeds_on=-1, success_priority: bool=True, **kwargs
    ) -> None:

        f = partial(iterable, *args, **kwargs)
        super().__init__(
            f, is_method,
            fails_on=fails_on, succeeds_on=succeeds_on, 
            success_priority=success_priority
        )

    def __call__(self, *args, **kwargs) -> TaskStatus:

        result = self._exec(*args, **kwargs)
        return _functional.parallel(result, self._state)


class NotFunc(TaskFunc):

    def __init__(self, f, is_method: bool=False, instance=None):
        super().__init__(f, is_method, instance)
        self._state = {}

    def __call__(self, *args, **kwargs) -> TaskStatus:
        
        result = self._exec(*args, **kwargs)
        return _functional.not_(
            result, self._state
        )
    
    def reset(self):
        
        super().__init__()
        self._state = {}


def condfunc(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return CondFunc(wrapper)


def condmethod(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    
    return CondFunc(f, True)


def actionfunc(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return ActionFunc(wrapper)


def actionmethod(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return ActionFunc(f, True)


def sequencefunc(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return SequenceFunc(wrapper)


def sequencemethod(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return SequenceFunc(f, True)


def selectorfunc(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return SelectorFunc(wrapper)


def selectormethod(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return SelectorFunc(f, True)



fallbackfunc = selectorfunc



def untilfunc(status: TaskStatus=TaskStatus.SUCCESS):

    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # update this
        if hasattr(f, '__self__') or '__self__' in dir(f):
            return UntilFunc(f, status)
        else:
            return UntilFunc(wrapper, status)
    return _


def unlessfunc(status: TaskStatus=TaskStatus.SUCCESS):

    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return UnlessFunc(wrapper, status)
    return _


def unlessmethod(status: TaskStatus=TaskStatus.SUCCESS):

    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        
        return UnlessFunc(f, status, True)
    return _


def parallelfunc(
    n_succeeds: int=-1, n_fails=1, 
):

    def _(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return ParallelFunc(wrapper, n_succeeds, n_fails)
    return _


def parallelmethod(
    succeeds_on: int=-1, fails_on=1, 
    success_priority: bool=True
):

    def _(f):

        f.factory = partial(
            ParallelFunc, succeeds_on=succeeds_on, fails_on=fails_on, 
            success_priority=success_priority
        )
        return f

    return _


# TODO: Write these functions
# class SequenceFunc(TaskFuncDecorator, tasks.Sequence):

#     def _initiate(self) -> tasks.TaskStatus:
        
#         if self.instance:
#             return self.f(self.instance)
#         return self.f()

#     def _advance(self) -> tasks.TaskStatus:
        
#         if self.instance:
#             return self.f(self.instance)
#         return self.f()


# class _DecMethod(Module):

#     def __init__(self, f: typing.Callable, instance=None):
#         self.f = f
#         self.instance = instance
#         self._stored = None

#     def forward(self, *args, **kwargs) -> typing.Any:
#         if self.instance:
#             return self.f(self.instance, *args, **kwargs)
#         return self.f(*args, **kwargs)

#     def __get__(self, instance, owner):

#         if self._stored is not None and instance is self._stored:
#             return self._stored
#         return _DecMethod(self.f, instance)
