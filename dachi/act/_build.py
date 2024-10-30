from dachi.act._tasks import Task
from . import _tasks as behavior
import typing


class build_sango(object):
    """Base context method to build a behavior tree
    """

    def __init__(self) -> None:
        """Create the behavior tree. This is the root node
        """
        super().__init__()
        self._sango = behavior.Root()

    def __enter__(self):
        return self._sango
    
    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            raise


class build_composite(object):
    """Base context method to build a task decorator
    """

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
        elif isinstance(parent, behavior.Root):
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


class build_decorate(object):
    """Base context method to build a task decorator
    """

    def __init__(
        self, decorate: typing.Type[behavior.Decorator], 
        decorated: typing.Union['build_decorate', build_composite], parent: behavior.Task=None,
        **kwargs
    ) -> None:
        """Decorate the task

        """
        super().__init__()
        assert decorated.parent is None
        self._decorated = decorate(decorated.task, **kwargs)

        self._parent = parent
        
        if parent is None:
            pass
        elif isinstance(parent, behavior.Root):
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


class build_sequence(build_composite):
    """Context method to build a sequence task
    """

    def __init__(self, parent: behavior.Task=None) -> None:
        """Create a Sequence task

        Args:
            parent (behavior.Task, optional): The parent of the sequence. Defaults to None.
        """
        super().__init__(behavior.Sequence([]), parent)


class build_select(build_composite):
    """Context method to build a Selector task
    """

    def __init__(self, parent: behavior.Task=None) -> None:
        """Create a selector task

        Args:
            parent (behavior.Task, optional): The parent of the selector. Defaults to None.
        """
        super().__init__(behavior.Selector([]), parent)


build_fallback = build_select


class build_parallel(build_composite):
    """Context manager to build an Parallel task
    """

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


class build_not(build_decorate):
    """Context manager to build an Not decorated task
    """

    def __init__(self, decorated: build_composite, parent: Task = None) -> None:
        """Invert the output of a composite task

        Args:
            decorated (composite): The task to decorate
            parent (Task, optional): The parent task. Defaults to None.
        """
        super().__init__(behavior.Not, decorated, parent)


class build_unless(build_decorate):
    """Context manager to build an Unless decorated task
    """

    def __init__(self, decorated, parent: Task = None, target_status: behavior.TaskStatus=behavior.TaskStatus.FAILURE) -> None:
        """Loop over the subtask while it 'succeeds'

        Args:
            decorated: The task to loop over
            parent (Task, optional): The parent task. Defaults to None.
        """
        super().__init__(behavior.Unless, decorated, parent, target_status=target_status)


class build_until(build_decorate):
    """Context manager to build an Until decorated task
    """

    def __init__(self, decorated: build_composite, parent: Task = None, target_status: behavior.TaskStatus=behavior.TaskStatus.SUCCESS) -> None:
        """Loop over the subtask until it 'succeeds'

        Args:
            decorated: The task to loop over
            parent (Task, optional): The parent task. Defaults to None.
        """
        super().__init__(behavior.Until, decorated, parent, target_status=target_status)
