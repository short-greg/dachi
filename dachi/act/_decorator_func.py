# TODO: Complete
from ._core import TaskStatus
import functools
from ._core import FuncTask


class SequenceFTask(FuncTask):
    """A task that executes a function and returns a status
    """

    def __post_init__(self):
        """Initialize the SequenceFTask"""
        super().__post_init__()
        self._gen = None

    async def func_tick(self):
        """Execute the function and return the status
        """
        if self._gen is None:
            self._gen = await self.f(*self.args, **self.kwargs)
        try:
            res = await anext(self._gen)
        except StopAsyncIteration:
            return TaskStatus.SUCCESS
        if res.failure:
            return TaskStatus.FAILURE
        return res
        

class SelectorFTask(FuncTask):
    """A task that executes a function and returns a status
    """

    def __post_init__(self):
        """Initialize the SelectorFTask"""
        super().__post_init__()
        self._gen = None

    async def func_tick(self):
        """Execute the function and return the status
        """
        if self._gen is None:
            self._gen = await self.f(*self.args, **self.kwargs)
        try:
            res = await anext(self._gen)
        except StopAsyncIteration:
            return TaskStatus.FAILURE
        
        if res.success:
            return TaskStatus.SUCCESS
        return res
    
    def reset(self):
        """Reset the task state"""
        super().reset()
        self._gen = None


class ActionFTask(FuncTask):
    """A task that executes a function and returns a status
    """

    def __post_init__(self):
        """Initialize the ActionFTask"""
        super().__post_init__()
        self._gen = None

    async def func_tick(self):
        """Execute the function and return the status
        """
        if self._gen is None:
            self._gen = await self.f(*self.args, **self.kwargs)
        
        try:
            res = await anext(self._gen)
        except StopAsyncIteration:
            raise RuntimeError(
                "The function did not return a status. "
                "Ensure the function yields a TaskStatus or raises StopIteration."
            )
        return res

    def reset(self):
        """Reset the task state"""
        super().reset()
        self._gen = None


class CondFTask(FuncTask):
    """A task that executes a function and 
    returns a boolean condition
    """
    async def func_tick(self):
        """Execute the function and return the status
        """

        result = await self.f(
            *self.args, **self.kwargs
        )
        return TaskStatus.from_bool(result)


def condtask(f):
    """Decorator for a conditional function that 
    returns True or False

    Args:
        f (Callable): The function to execute

    Returns:
        CondFTask: The task that executes the function
    """
    async def wrapper(*args, **kwargs):
        """Wrapper to create a CondFTask"""
        return await f(*args, **kwargs)
    
    @functools.wraps(f, wrapper)
    def create(*args, **kwargs):
        task = CondFTask(name=f.__name__) 
        
        task.f = f
        task.args = args
        task.kwargs = kwargs
        return task

    wrapper.is_task = True
    wrapper.create = create
    return wrapper


def actiontask(f):
    """Decorator for a function that returns a TaskStatus

    Args:
        f (Callable): The function to execute

    Returns:
        ActionFTask: The task that executes the function
    """
    async def wrapper(*args, **kwargs):
        """Wrapper to create an ActionFTask"""
        return await f(*args, **kwargs)
    
    @functools.wraps(f, wrapper)
    def create(*args, **kwargs):
        task = ActionFTask(name=f.__name__) 
        
        task.f = f
        task.args = args
        task.kwargs = kwargs
        return task

    wrapper.is_task = True
    wrapper.create = create
    return wrapper


def sequencetask(f):
    """Decorator for a function that returns a sequence of tasks

    Args:
        f (Callable): The function to execute

    Returns:
        SequenceFTask: The task that executes the function
    """
    async def wrapper(*args, **kwargs):
        """Wrapper to create a SequenceFTask"""
        return await f(*args, **kwargs)
    
    @functools.wraps(f, wrapper)
    def create(*args, **kwargs):
        task = SequenceFTask(name=f.__name__) 
        
        task.f = f
        task.args = args
        task.kwargs = kwargs
        return task

    wrapper.is_task = True
    wrapper.create = create
    return wrapper


def selectortask(f):
    """Decorator for a function that returns a selector of tasks

    Args:
        f (Callable): The function to execute

    Returns:
        SelectorFTask: The task that executes the function
    """
    async def wrapper(*args, **kwargs):
        """Wrapper to create a SelectorFTask"""
        return await f(*args, **kwargs)
    
    @functools.wraps(f, wrapper)
    def create(*args, **kwargs):
        task = SelectorFTask(name=f.__name__) 
        
        task.f = f
        task.args = args
        task.kwargs = kwargs
        return task

    wrapper.is_task = True
    wrapper.create = create
    return wrapper

fallbacktask = selectortask
