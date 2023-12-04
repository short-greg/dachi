from dachi.behavior import _tasks as behavior
from . import _tasks as behavior
import typing
from abc import abstractproperty, abstractmethod


class BehaviorBuilder(object):

    @abstractproperty
    def parent(self) -> typing.Union['DecoratorBuilder', 'sango', 'CompositeBuilder']:
        pass

    @abstractmethod
    def build(self) -> typing.Type[behavior.Task]:
        pass


class sango(BehaviorBuilder):

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._task = None
        self._parent = None

    @property
    def parent(self) -> 'BehaviorBuilder':

        return self._parent
    
    @parent.setter
    def parent(self, parent) -> 'BehaviorBuilder':

        self._parent = parent

    def parent(self) -> typing.Union['DecoratorBuilder', 'sango', 'CompositeBuilder']:
        return self._parent
    
    def set(self, task: typing.Union[behavior.Task, BehaviorBuilder]):
        assert not isinstance(task, sango)
        self._task = task

    def build(self) -> typing.Type[behavior.Task]:
        
        if isinstance(self._task, behavior.Task):
            task = self._task.clone()
        elif isinstance(self._task, 'BehaviorBuilder'):
            task = self._task.build()

        return behavior.Sango(
            self._name, task
        )

    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        pass


class CompositeBuilder(BehaviorBuilder):

    def __init__(self, parent: 'CompositeBuilder', name: str=''):

        self._tasks = []
        self._parent = parent
        self._name = name

    def add(self, task: behavior.Task):

        self._tasks.append(task)

    @property
    def tasks(self) -> typing.List[behavior.Task]:
        return [*self._tasks]
    
    @property
    def parent(self) -> typing.Union['DecoratorBuilder', 'sango', 'CompositeBuilder']:
        return self._parent

    @parent.setter
    def parent(self, parent) -> 'BehaviorBuilder':

        self._parent = parent
    
    @abstractmethod
    def build_composite(self, tasks: typing.List[behavior.Task]) -> behavior.Composite:
        pass

    def build(self) -> typing.Type[behavior.Composite]:
        
        return self.build_composite([
            task.clone() if isinstance(task, behavior.Task) else task.build
            for task in self._tasks
        ])

    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        if self.parent is None:
            return 
        
        if isinstance(self.parent, typing.Union['DecoratorBuilder', CompositeBuilder]):
            self.parent.add(self)
        elif isinstance(self.parent, sango):
            self.parent.set(self)


class DecoratorBuilder(BehaviorBuilder):

    def __init__(self, wrapped: typing.Union['DecoratorBuilder', CompositeBuilder]):

        self._wrapped = wrapped

    def __enter__(self):
        
        return self
    
    def add(self, task: behavior.Task):

        self._wrapped.add(task)
    
    @property
    def parent(self) -> typing.Union['DecoratorBuilder', 'sango', 'CompositeBuilder']:
        return self._wrapped.parent

    @parent.setter
    def parent(self, parent) -> 'BehaviorBuilder':

        self._wrapped.parent = parent

    @abstractmethod
    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        pass

    def build(self) -> typing.Type[behavior.Decorator]:
        wrapped = self._wrapped.build()
        return self.build_decorator(wrapped)

    def __exit__(self, exc_type, exc_value, traceback):

        if self.parent is None:
            return 
        
        if isinstance(self.parent, typing.Union['DecoratorBuilder', CompositeBuilder]):
            self.parent.add(self)
        elif isinstance(self.parent, sango):
            self.parent.set(self)


class sequence(CompositeBuilder):

    def build(self) -> typing.Type[behavior.Composite]:
        return behavior.Sequence(
            self._tasks, self._name
        )


class select(CompositeBuilder):

    def build(self) -> typing.Type[behavior.Composite]:
        return behavior.Selector(
            self._tasks, self._name
        )


class parallel(CompositeBuilder):

    def __init__(
        self, name: str, parent: typing.Union[behavior.Sango, CompositeBuilder], 
        fails_on: int=1, succeeds_on: int=None, success_priority: bool=False
    ):
        super().__init__(
            name, parent
        )
        self._fails_on = fails_on
        self._succeeds_on = succeeds_on
        self._success_priority = success_priority

    def build(self) -> typing.Type[behavior.Composite]:
        return behavior.Parallel(
            self._tasks, self._name, fails_on=self._fails_on,
            succeeds_on=self._succeeds_on, success_priority=self._success_priority
        )


class not_(DecoratorBuilder):

    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        return behavior.Not(wrapped)


class until_(DecoratorBuilder):

    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        return behavior.Until(wrapped)


class while_(DecoratorBuilder):

    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        return behavior.While(wrapped)
