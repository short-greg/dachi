# from ._tasks import Task, Action, Parallel, Sequence, Selector, Condition
# import typing
# from ._status import TaskStatus
# import inspect
# from functools import partial


# class CondFunc(Condition):

#     def __init__(self, f: typing.Callable[[typing.Any], bool], *args, **kwargs) -> None:
#         super().__init__()
#         self._f = partial(f, *args, **kwargs)
#         self._args = args
#         self._kwargs = kwargs

#     def condition(self) -> bool:
#         return self._f()


# def condfunc(f):

#     f.factory = CondFunc
#     return f


# class ActionFunc(Action):

#     def __init__(self, action: typing.Callable[[typing.Any], TaskStatus], *args, **kwargs) -> None:
#         super().__init__()
#         self._args = args
#         self._kwargs = kwargs
#         self._iterate_over = inspect.isgeneratorfunction(
#             action
#         )
#         f = partial(action, *args, **kwargs)
#         self._f = f
#         self._iterate = None

#     def act(self) -> TaskStatus:
        
#         if self._iterate_over:
#             if self._iterate is None:
#                 self._iterate = iter(self._f())
#             try:
#                 return next(self._iterate)
#             except StopIteration:
#                 return TaskStatus.SUCCESS
#         else:
#             return self._f(*self._args, **self._kwargs)
    
#     def __reset__(self):
#         super().reset()
#         self._iterate = None


# def actionfunc(f):

#     f.factory = ActionFunc
#     return f


# def taskf(f, *args, **kwargs) -> Task:

#     return f.factory(f, *args, **kwargs)


# class SelectorFunc(Selector):

#     def __init__(self, iterable: typing.Callable[[typing.Any], bool], *args, **kwargs) -> None:
        
#         f = partial(iterable, *args, **kwargs)
#         super().__init__(f)


# def selectorfunc(f):

#     f.factory = SelectorFunc
#     return f


# class SequenceFunc(Sequence):

#     def __init__(self, iterable: typing.Callable[[typing.Any], bool], *args, **kwargs) -> None:
        
#         print('F: ', iterable)
#         print(inspect.isgeneratorfunction(
#             iterable
#         ))
#         f = partial(iterable, *args, **kwargs)
#         print(inspect.isgeneratorfunction(
#             f
#         ))
#         super().__init__(f)


# def sequencefunc(f):

#     f.factory = SequenceFunc

#     return f


# class ParallelFunc(Parallel):

#     def __init__(
#         self, iterable: typing.Callable[[typing.Any], bool], *args, fails_on: int=1, 
#         succeeds_on=-1, success_priority: bool=True, **kwargs
#     ) -> None:

#         f = partial(iterable, *args, **kwargs)
#         super().__init__(
#             f,
#             fails_on=fails_on, succeeds_on=succeeds_on, 
#             success_priority=success_priority
#         )


# def parallelfunc(
#     succeeds_on: int=-1, fails_on=1, 
#     success_priority: bool=True
# ):

#     def _(f):

#         f.factory = partial(
#             ParallelFunc, succeeds_on=succeeds_on, fails_on=fails_on, 
#             success_priority=success_priority
#         )
#         return f

#     return _


# # @actionf
# # def x(x: int, y: int) -> TaskStatus
# #
# # yield self.x.task(1, 2)
# # self.x(1, 2)


