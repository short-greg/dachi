from . import _tasks as behavior
import typing


class CompositeBuilder(object):

    def __init__(self):

        self._tasks = []

    def add(self, task: behavior.Task):

        self._tasks.append(task)

    @property
    def tasks(self) -> typing.List[behavior.Task]:
        return [*self._tasks]


class sequence(object):

    def __init__(self, name: str, parent: typing.Union[behavior.Sango, CompositeBuilder]):

        self._sequence = CompositeBuilder()
        self._name = name
        self._parent = parent

    def __enter__(self):
        
        return self._sequence
    
    def __exit__(self, exc_type, exc_value, traceback):

        sequence = behavior.Sequence(self._name, [self._sequence.tasks])
        
        if isinstance(self._parent, behavior.Composite):
            self._parent.add(sequence)
        else:
            self._parent.root = sequence


class fallback(object):

    def __init__(self, name: str, parent: typing.Union[behavior.Sango, CompositeBuilder]):
        
        self._fallback = CompositeBuilder()
        self._name = name
        self._parent = parent

    def __enter__(self):

        return self._fallback
    
    def __exit__(self, exc_type, exc_value, traceback):

        fallback = behavior.Fallback(self._name, [self._fallback.tasks])
        
        if isinstance(self._parent, behavior.Composite):
            self._parent.add(fallback)
        else:
            self._parent.root = fallback


class parallel(object):

    def __init__(
        self, name: str, parent: typing.Union[behavior.Sango, CompositeBuilder], 
        fails_on: int=1, succeeds_on: int=None, success_priority: bool=False
    ):
        self._parallel = CompositeBuilder()
        self._name = name
        self._parent = parent
        self._fails_on = fails_on
        self._succeeds_on = succeeds_on
        self._success_priority = success_priority

    def __enter__(self):
        return self._parallel
    
    def __exit__(self, exc_type, exc_value, traceback):

        parallel = behavior.Parallel(
            self._name, [self._parallel.tasks], 
            fails_on=self._fails_on, succeeds_on=self._succeeds_on, 
            success_priority=self._success_priority
        )
        
        if isinstance(self._parent, CompositeBuilder):
            self._parent.add(parallel)
        else:
            self._parent.root = parallel
