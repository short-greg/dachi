from dachi.behavior._tasks import Task
from . import _tasks as behavior
import typing


class sango(object):

    def __init__(self) -> None:
        """Create the behavior tree. This is the root node
        """
        super().__init__()
        self._sango = behavior.Sango()

    def __enter__(self):
        
        return self._sango
    
    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            raise


class composite(object):

    def __init__(self, child: behavior.Task, parent: behavior.Task=None) -> None:
        """Create a composite node that uses a list to store all of the subtasks

        Args:
            name (str): The name of the "Sango"
        """
        super().__init__()
        self._child = child
        self._parent = parent
        
        if parent is None:
            pass
        elif isinstance(parent, behavior.Sango):
            parent.root = child
        elif isinstance(parent, behavior.Serial):
            parent.add(child)
        elif isinstance(parent, behavior.Parallel):
            parent.add(child)

    @property
    def task(self) -> behavior.Task:
        """
        Returns:
            behavior.Task: The composite task
        """
        return self._child

    @property
    def parent(self) -> behavior.Task:
        """
        Returns:
            behavior.Task: The parent task of the composite task
        """
        return self._parent

    def __enter__(self) -> behavior.Serial:
        return self._child
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise


class decorate(object):

    def __init__(
        self, decorate: typing.Type[behavior.Decorator], 
        decorated: typing.Union['decorate', composite], parent: behavior.Task=None
    ) -> None:
        """Decorate the task

        """
        super().__init__()
        assert decorated.parent is None
        self._decorated = decorate(decorated.task)

        self._parent = parent
        
        if parent is None:
            pass
        elif isinstance(parent, behavior.Sango):
            parent.root = self._decorated
        elif isinstance(parent, behavior.Serial):
            parent.add(self._decorated)
        elif isinstance(parent, behavior.Parallel):
            parent.add(self._decorated)

    @property
    def task(self) -> behavior.Task:
        """
        Returns:
            behavior.Task: The decorated task
        """
        return self._decorated

    @property
    def parent(self) -> behavior.Task:
        """
        Returns:
            behavior.Task: The parent of the decorated tak
        """
        return self._parent

    def __enter__(self):
        return self._decorated
    
    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            raise


class sequence(composite):

    def __init__(self, parent: behavior.Task=None) -> None:
        """Create a Sequence task

        Args:
            parent (behavior.Task, optional): The parent of the sequence. Defaults to None.
        """
        super().__init__(behavior.Sequence([]), parent)


class select(composite):

    def __init__(self, parent: behavior.Task=None) -> None:
        """Create a selector task

        Args:
            parent (behavior.Task, optional): The parent of the selector. Defaults to None.
        """
        super().__init__(behavior.Selector([]), parent)


fallback = select


class parallel(composite):

    def __init__(
        self, parent: behavior.Task=None, 
        fails_on: int=1, succeeds_on: int=None, success_priority: bool=False
    ):
        """Execute two tasks in parallel

        Args:
            parent (typing.Union[behavior.Sango, CompositeBuilder], optional): The parent biulder. Defaults to None.
            fails_on (int, optional): The number of tasks that must fail for this to fail. Defaults to 1.
            succeeds_on (int, optional): The number of tasks to succeed for the parallel task to succeed. Defaults to None.
            success_priority (bool, optional): Whether it will return as soon as it succeeds. Defaults to False.
        """
        parallel = behavior.Parallel(
            [], fails_on=fails_on, succeeds_on=succeeds_on, 
            success_priority=success_priority)
        super().__init__(
            parallel, parent
        )


class not_(decorate):

    def __init__(self, decorated: composite, parent: Task = None) -> None:
        """Invert the output of a composite task

        Args:
            decorated (composite): The task to decorate
            parent (Task, optional): The parent task. Defaults to None.
        """
        super().__init__(behavior.Not, decorated, parent)


class while_(decorate):

    def __init__(self, decorated, parent: Task = None) -> None:
        """Loop over the subtask while it 'succeeds'

        Args:
            decorated: The task to loop over
            parent (Task, optional): The parent task. Defaults to None.
        """
        super().__init__(behavior.While, decorated, parent)


class until_(decorate):

    def __init__(self, decorated: composite, parent: Task = None) -> None:
        """Loop over the subtask until it 'succeeds'

        Args:
            decorated: The task to loop over
            parent (Task, optional): The parent task. Defaults to None.
        """
        super().__init__(behavior.Until, decorated, parent)
