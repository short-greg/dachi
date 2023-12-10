from dachi.behavior import _tasks as behavior
from . import _tasks as behavior
import typing
from abc import abstractmethod


class BehaviorBuilder(object):

    @abstractmethod
    def build(self) -> typing.Type[behavior.Task]:
        pass


class sango(BehaviorBuilder):

    def __init__(self, name: str) -> None:
        """Create a Sango builder

        Args:
            name (str): The name of the "Sango"
        """
        super().__init__()
        self.name = name
        self._task = None
    
    def set(self, task: typing.Union[behavior.Task, BehaviorBuilder]):
        """Set the root builder for the Sango

        Args:
            task (typing.Union[behavior.Task, BehaviorBuilder]): The root task builder 
        """
        assert not isinstance(task, sango)
        self._task = task

    def build(self) -> behavior.Sango:
        """Build the Sango

        Returns:
            behavior.Sango: The built "Sango"
        """
        if isinstance(self._task, behavior.Task):
            task = self._task.clone()
        elif isinstance(self._task, BehaviorBuilder):
            task = self._task.build()
        else: task = None

        return behavior.Sango(
            self.name, task
        )

    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            raise


class CompositeBuilder(BehaviorBuilder):

    def __init__(self, name: str='', parent: 'CompositeBuilder'=None):
        """Create a composite task

        Args:
            name (str, optional): The name of the task. Defaults to ''.
            parent (CompositeBuilder, optional): The parent for the task. Defaults to None.
        """
        self._tasks = []
        self.parent = parent
        self.name = name

    def add(self, task: behavior.Task):
        """Add a task to teh tasks to build

        Args:
            task (behavior.Task): The task to add
        """
        self._tasks.append(task)

    @property
    def tasks(self) -> typing.List[behavior.Task]:
        """ 
        Returns:
            typing.List[behavior.Task]: the list of tasks making up the composite builder
        """
        return [*self._tasks]
    
    @abstractmethod
    def build_composite(self, tasks: typing.List[behavior.Task]) -> behavior.Composite:
        pass

    def build_tasks(self) -> typing.List[behavior.Task]:

        return [
            task.build() if isinstance(task, BehaviorBuilder)
            else task.clone()
            for task in self._tasks
        ]

    def build(self) -> behavior.Composite:
        """Build the composite task

        Returns:
            behavior.Composite: The built Composite task
        """
        return self.build_composite(self.build_tasks())

    def __enter__(self):
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            raise 

        print(self.name, self.parent)
        if self.parent is None:
            return 

        if isinstance(self.parent,CompositeBuilder) or isinstance(self.parent, DecoratorBuilder):
            self.parent.add(self)
        elif isinstance(self.parent, sango):
            self.parent.set(self)


class DecoratorBuilder(BehaviorBuilder):

    def __init__(self, wrapped: typing.Union['DecoratorBuilder', CompositeBuilder]):
        """

        Args:
            wrapped (typing.Union[&#39;DecoratorBuilder&#39;, CompositeBuilder]): _description_
        """
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

    def build(self) -> behavior.Decorator:
        """Build the decorator task

        Returns:
            behavior.Decorator: The built Decorator task
        """
        wrapped = self._wrapped.build()
        return self.build_decorator(wrapped)

    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            raise 

        if self.parent is None:
            return 

        if isinstance(self.parent,CompositeBuilder) or isinstance(self.parent, DecoratorBuilder):
            self.parent.add(self)
        elif isinstance(self.parent, sango):
            self.parent.set(self)


class sequence(CompositeBuilder):

    def build_composite(self, tasks: typing.List[behavior.Task]) -> behavior.Sequence:  
        """Build the sequential task

        Returns:
            behavior.Sequence: The built Sequence task
        """ 
        return behavior.Sequence(
            tasks, self.name
        )


class select(CompositeBuilder):

    def build_composite(self, tasks: typing.List[behavior.Task]) -> behavior.Selector: 
        """Build the Selector task

        Returns:
            behavior.Selector: The built Selector task
        """   
        return behavior.Selector(
            tasks, self.name
        )

class parallel(CompositeBuilder):

    def __init__(
        self, name: str='', parent: typing.Union[behavior.Sango, CompositeBuilder]=None, 
        fails_on: int=1, succeeds_on: int=None, success_priority: bool=False
    ):
        """Execute two tasks in parallel

        Args:
            name (str): The name of the parallel node
            parent (typing.Union[behavior.Sango, CompositeBuilder], optional): The parent biulder. Defaults to None.
            fails_on (int, optional): The number of tasks that must fail for this to fail. Defaults to 1.
            succeeds_on (int, optional): The number of tasks to succeed for the parallel task to succeed. Defaults to None.
            success_priority (bool, optional): Whether it will return as soon as it succeeds. Defaults to False.
        """
        super().__init__(
            name, parent
        )
        self.fails_on = fails_on
        self.succeeds_on = succeeds_on
        self.success_priority = success_priority

    def build_composite(self, tasks: typing.List[behavior.Task]) -> behavior.Parallel: 
        """Build the parallel task

        Returns:
            behavior.Parallel: The built Parallel task
        """  
        return behavior.Parallel(
            tasks, self.name, fails_on=self.fails_on,
            succeeds_on=self.succeeds_on, success_priority=self.success_priority
        )


class not_(DecoratorBuilder):
    """Negate the task that is decorated. So if the decorated task
    outputs Success, it will be a Failure and vice verss
    """

    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        return behavior.Not(wrapped)


class until_(DecoratorBuilder):
    """Loop on the task that is decorated until it succeeds. 
    """

    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        return behavior.Until(wrapped)


class while_(DecoratorBuilder):
    """Loop on the task that is while it succeeds. 
    """
    def build_decorator(self, wrapped: behavior.Task) -> behavior.Decorator:
        return behavior.While(wrapped)
