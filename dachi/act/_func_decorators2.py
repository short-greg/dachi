from . import _tasks as tasks
import typing
from functools import wraps


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

class TaskFuncDecorator(object):

    def __init__(self, f: typing.Callable, instance=None):
        super().__init__()
        self.f = f
        self.instance = instance
        self._stored = None
    
    def __get__(self, instance, owner):

        if self._stored is not None and instance is self._stored:
            return self._stored
        self._stored = self.__class__(self.f, instance)
        return self._stored


class CondFunc(TaskFuncDecorator, tasks.Condition):

    def condition(self) -> bool:
        if self.instance:
            return self.f(self.instance)
        return self.f()


class ActionFunc(TaskFuncDecorator, tasks.Action):

    def act(self) -> tasks.TaskStatus:
        if self.instance:
            return self.f(self.instance)
        return self.f()


class SequenceFunc(TaskFuncDecorator, tasks.Sequence):

    def __init__(self, f, instance=None):
        super().__init__(f, instance)
        tasks.Sequence.__init__(self, self.f)


class SelectorFunc(TaskFuncDecorator, tasks.Selector):

    def __init__(self, f, instance=None):
        super().__init__(f, instance)
        tasks.Selector.__init__(self, self.f)


def cond(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    if hasattr(f, '__self__') or '__self__' in dir(f):
        return CondFunc(f)
    else:
        return CondFunc(wrapper)


def action(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    if hasattr(f, '__self__') or '__self__' in dir(f):
        return ActionFunc(f)
    else:
        return ActionFunc(wrapper)
    

def sequence(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    if hasattr(f, '__self__') or '__self__' in dir(f):
        return SequenceFunc(f)
    else:
        return SequenceFunc(wrapper)


def selector(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    if hasattr(f, '__self__') or '__self__' in dir(f):
        return SelectorFunc(f)
    else:
        return SelectorFunc(wrapper)

fallback = selector

# TODO: Add in until and while


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

