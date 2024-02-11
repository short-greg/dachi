from abc import abstractmethod
import typing
from ..base import Storable
from ._status import SangoStatus
from dataclasses import dataclass
from ..storage import (
    Struct, Q, R, DDict, 
    PromptConv, Completion,
)
from ..graph import Tako
from ..comm import (
    Request, UIQuery, LLMQuery, UI,
    ProcessResponse, Processed,
    NullProcessResponse
)

from functools import partial

import threading


@dataclass
class TaskMessage:

    name: str
    data: typing.Any


class Task(Storable):
    """The base class for a task in the behavior tree
    """

    SUCCESS = SangoStatus.SUCCESS
    FAILURE = SangoStatus.FAILURE
    RUNNING = SangoStatus.RUNNING

    def __init__(self) -> None:
        """Create the task

        Args:
            name (str): The name of the task
        """
        super().__init__()
        self._status = SangoStatus.READY

    @abstractmethod    
    def tick(self) -> SangoStatus:
        raise NotImplementedError

    def __call__(self) -> SangoStatus:
        """

        Args:
            terminal (Terminal): _description_

        Returns:
            SangoStatus: _description_
        """
        return self.tick()

    def reset(self):
        """Reset the terminal

        """
        self._status = SangoStatus.READY
    
    @property
    def status(self) -> SangoStatus:
        return self._status

    @property
    def id(self):
        return self._id


class Sango(Task):

    def __init__(self, root: 'Task'=None):
        """Create a tree to store the tasks

        Args:
            root (Task, optional): The root task. Defaults to None.
            name (str): The name of the tree
        """
        super().__init__()
        self._root = root

    def tick(self) -> SangoStatus:
        """Update the task

        Returns:
            SangoStatus: The status after tick
        """
        if self._root is None:
            return SangoStatus.SUCCESS

        return self._root.tick()

    @property
    def root(self) -> Task:
        """
        Returns:
            Task: The root task
        """
        return self._root
    
    @root.setter
    def root(self, root: 'Task'):
        """
        Args:
            root (Task): The root task
        """
        self._root = root

    def reset(self):
        """Reset the status of the task

        """
        super().reset()
        if self._root is not None:
            self._root.reset()


class Serial(Task):
    """Task composed of subtasks
    """
    def __init__(
        self, tasks: typing.Iterable[Task], stop_on: bool=SangoStatus.FAILURE
    ):
        super().__init__()
        self._tasks = tasks if tasks is not None else []
        self._cur = None
        self._iter = None
        self._stop_on = stop_on
        self._ticked = set()

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return self._tasks

    @abstractmethod
    def subtick(self) -> SangoStatus:
        """Tick each subtask. Implement when implementing a new Composite task"""
        
        if self._status.is_done:
            return self._status
    
        # Not started yet?
        if self._iter is None:
            self._iter = iter(self._tasks)

            try:
                self._cur = next(self._iter)
                self._ticked.add(self._cur)
            except StopIteration:
                self._iter = None
                self._cur = None
                return SangoStatus.SUCCESS

        status = self._cur.tick()

        # Still in progress?
        if status == SangoStatus.RUNNING:
            return SangoStatus.RUNNING
    
        # Reached a stopping condition?
        if (status == SangoStatus.FAILURE == self._stop_on):
            self._cur = None
            self._iter = None
            return status

        # Made it to the end?
        try:
            self._cur = next(self._iter)
            self._ticked.append(self._cur)
        except StopIteration:
            self._cur = None
            self._iter = None
            return SangoStatus.SUCCESS

        return SangoStatus.RUNNING

    def tick(self) -> SangoStatus:
        self._status = self.subtick()
        
        return self._status
    
    def reset(self):
        super().reset()
        self._cur = None
        self._iter = None
        for task in self._ticked:
            task.reset()
        self._ticked = set()


class Sequence(Serial):

    def __init__(self, tasks: typing.Iterable[Task]):
        super().__init__(tasks, False)


class Selector(Serial):

    def __init__(self, tasks: typing.Iterable[Task]):
        super().__init__(tasks, False)


class Parallel(Task):
    """A composite task for running multiple tasks in parallel
    """

    def __init__(self, tasks: typing.Iterable[Task]=None, fails_on: int=None, succeeds_on: int=None, success_priority: bool=True):
        """

        Args:
            tasks (typing.Iterable[Task]): 
            runner (, optional): . Defaults to None.
            fails_on (int, optional): . Defaults to None.
            succeeds_on (int, optional): . Defaults to None.
            success_priority (bool, optional): . Defaults to True.
        """
        super().__init__()
        self._tasks = tasks if tasks is not None else []
        self.set_condition(fails_on, succeeds_on)
        self._success_priority = success_priority
        self._ticked = set()

    # def add(self, task: Task):
    #     """

    #     Args:
    #         task (Task): 
    #     """
    #     self._tasks.append(task)

    # def add_tasks(self, tasks: typing.Iterable[Task]) -> 'Serial':
    #     self._tasks.extend(tasks)

    def set_condition(self, fails_on: int, succeeds_on: int):
        """Set the number of falures or successes it takes to end

        Args:
            fails_on (int): The number of failures it takes to fail
            succeeds_on (int): The number of successes it takes to succeed

        Raises:
            ValueError: If teh number of successes or failures is invalid
        """
        self._fails_on = fails_on if fails_on is not None else len(self._tasks)
        self._succeeds_on = succeeds_on if succeeds_on is not None else (len(self._tasks) + 1 - self._fails_on)

    def validate(self):
        
        if (self._fails_on + self._succeeds_on - 1) > len(self._tasks):
            raise ValueError('')
        if self._fails_on <= 0 or self._succeeds_on <= 0:
            raise ValueError('')
    
    # def _default_runner(self, tasks: typing.List[Task]) -> typing.List[SangoStatus]:
    #     """Run the paralel

    #     Args:
    #         tasks (typing.List[Task]): The tasks to run

    #     Returns:
    #         typing.List[SangoStatus]:
    #     """
    #     statuses = []
    #     for task in tasks:
    #         statuses.append(task.tick())

    #     return statuses

    def _accumulate(self, statuses: typing.List[SangoStatus]) -> SangoStatus:
        
        successes = 0
        failures = 0
        waiting = 0
        dones = 0
        for status in statuses:
            failures += status.failure
            successes += status.success

            dones += status.is_done
            waiting += status.waiting

        has_failed = failures >= self._fails_on
        has_succeeded = successes >= self._succeeds_on
        if self._success_priority:
            if has_succeeded:
                return SangoStatus.SUCCESS
            if has_failed:
                return SangoStatus.FAILURE

        if has_failed:
            return SangoStatus.FAILURE
        if has_succeeded:
            return SangoStatus.SUCCESS
        if waiting == (len(statuses) - dones):
            return SangoStatus.WAITING
        # failures + successes - 1
        return SangoStatus.RUNNING

    def subtick(self) -> SangoStatus:

        statuses = []
        for task in self.tasks:
            statuses.append(task.tick())
            self._ticked.add(task)

        return self._accumulate(statuses)

    @property
    def fails_on(self) -> int:
        return self._fails_on

    @fails_on.setter
    def fails_on(self, fails_on) -> int:
        self._fails_on = fails_on
        return self._fails_on

    @property
    def succeeds_on(self) -> int:
        return self._succeeds_on        
    
    @succeeds_on.setter
    def succeeds_on(self, succeeds_on) -> int:
        self._succeeds_on = succeeds_on
        return self._succeeds_on
    
    def tick(self) -> SangoStatus:
        
        self._status = self.subtick()
        return self._status

    def reset(self):

        super().reset()
        for task in self._ticked:
            task.reset()
        self._ticked = set()


class Action(Task):

    @abstractmethod
    def act(self) -> SangoStatus:
        raise NotImplementedError

    def tick(self) -> SangoStatus:

        if self._status.is_done:
            return self._status
        self._status = self.act()
        return self._status


class Condition(Task):
    """Check whether a condition is satisfied before
    running a task.
    """

    @abstractmethod
    def condition(self) -> bool:
        pass

    def tick(self) -> SangoStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self._status = SangoStatus.SUCCESS if self.condition() else SangoStatus.FAILURE
        return self._status


class Decorator(Task):

    # name should retrieve the name of the decorated
    def __init__(self, task: Task) -> None:
        super().__init__()
        self._task = task

    @abstractmethod
    def decorate(self, status: SangoStatus) -> bool:
        pass

    @property
    def task(self) -> Task:
        return self._task

    def tick(self) -> SangoStatus:
        """Decorate the tick for the decorated task

        Args:
            terminal (Terminal): The terminal for the task

        Returns:
            SangoStatus: The decorated status
        """

        if self._status.is_done:
            return self._status

        self._status = self.decorate(
            self.task.tick()
        )
        return self._status

    def reset(self):

        super().reset()
        self._task.reset()


class Until(Decorator):
    """Loop until a condition is met
    """

    def decorate(self, status: SangoStatus) -> SangoStatus:
        """Continue running unless the result is a success

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status.success:
            return SangoStatus.SUCCESS
        if status.failure:
            self._task.reset()
            return SangoStatus.RUNNING
        return status


class While(Decorator):
    """Loop while a condition is met
    """

    def decorate(self, status: SangoStatus) -> SangoStatus:
        """Continue running unless the result is a failure

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status.failure:
            return SangoStatus.FAILURE
        if status.success:
            self._task.reset()
            return SangoStatus.RUNNING
        return status


class Not(Decorator):
    """Invert the result
    """

    def decorate(self, status: SangoStatus) -> SangoStatus:
        """Return Success if status is a Failure or Failure if it is a SUCCESS

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status.failure:
            return SangoStatus.SUCCESS
        if status.success:
            return SangoStatus.FAILURE
        return status


class Check(Condition):

    def __init__(self, data: Q, f: Q):
        super().__init__()
        self.data = data
        self.f = f

    def condition(self) -> bool:
        return self.f(self.data())


class CheckReady(Check):

    def __init__(self, r: R):
        super().__init__(r, lambda r: r() is not None)


class CheckTrue(Check):

    def __init__(self, r: R):
        super().__init__(r, lambda r: r is True)


class CheckFalse(Check):

    def __init__(self, r: R):
        super().__init__(r, lambda r: r is False or r is None)


class Reset(Action):

    def __init__(self, data: Q[Struct], on_condition: typing.Callable[[], None]=None):

        super().__init__()
        self._data = data
        self._cond = on_condition

    def act(self) -> SangoStatus:
        data = self._data()

        if self._cond is None or self._cond():
            data.reset()
            return SangoStatus.SUCCESS
        return SangoStatus.FAILURE


class TakoTask(Action):

    def __init__(self, tako: Tako):

        super().__init__()
        self._tako = tako
        self._executed = False

    def exec(self):
        self._tako()

    def act(self) -> SangoStatus:
        
        if self._status == self.READY:
            thread = threading.Thread(target=self.exec, args=[])
            thread.start()

        if self._executed:
            return self.SUCCESS
        
        return self.RUNNING


class Converse(Action):

    def __init__(
        self, prompt_conv: PromptConv, llm: LLMQuery, user_interface: UI,
        response_processor: ProcessResponse=None, **prompt_components: Struct
    ):
        # Use to send a message from the LLM
        super().__init__()
        self._conv = prompt_conv
        self._llm_query = llm
        self._ui_query = UIQuery(user_interface)
        self._ui = user_interface
        self._request = Request()
        self._processed: Processed = None
        self._response_processor = response_processor or NullProcessResponse()
        self._components = prompt_components

    def converse_turn(self):

        self._request = Request()
        conv = self._conv.with_components(**self._components)
        response = self._llm_query(conv, asynchronous=False)
        self._processed  = self._response_processor.process(response)

        if self._processed.to_interrupt:
            return
        self._ui.post_message('assistant', self._processed.text)

        self._conv.add_turn(
            'assistant', self._processed.text
        )
        self._request.contents = self._conv
        self._ui_query.post(self._request)

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            # use threading here
            thread = threading.Thread(
                target=self.converse_turn, args=[]
            )
            thread.start()

        if self._processed is not None and (
            self._request.responded is True or self._processed.to_interrupt
        ):
            
            if not self._processed.to_interrupt and self._request.success is True:
                self._conv.add_turn(
                    'user', self._request.response
                )
            if self._processed.succeeded:
                print('SUCCEEDED!')
                return SangoStatus.SUCCESS
            else:
                print('FAILED!')
                return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING

    def reset(self):
        super().reset()
        self._interrupt = False
        self._request = Request()


class PromptCompleter(Action):
    # Use to send a message from the LLM

    def __init__(
        self, completion: Completion, llm: LLMQuery, user_interface: UI,
        post_processor: typing.Callable=None, **prompt_components: Struct
    ):
        """

        Args:
            completion (Completion): 
            llm (Query): 
            user_interface (UI): 
        """
        super().__init__()
        self._completion = completion
        self._llm_query = llm
        self._ui = user_interface
        self._request = Request()
        self._post_processor = post_processor
        self._components = prompt_components

    def respond(self):

        self._request = Request()
        completion = self._completion.with_components(**self._components)
        response = self._llm_query(completion, asynchronous=False)
        self._ui.post_message('assistant', response)
        self._completion.response = response
        if self._post_processor is not None:
            self._post_processor()
        self._request.respond(response)

    def act(self) -> SangoStatus:
        
        if self._status == SangoStatus.READY:
            # use threading here
            thread = threading.Thread(target=self.respond, args=[])
            thread.start()

        if self._request.responded is True:
            # self._request.status # wouldn't this be easier? 
            if self._request.success is True:
                return SangoStatus.SUCCESS
            else:
                return SangoStatus.FAILURE
        
        return SangoStatus.RUNNING

    def reset(self):
        super().reset()
        self._request = Request()


class FAction(Action):

    def __init__(self, action: typing.Callable[[typing.Any], SangoStatus], *args, **kwargs) -> None:
        super().__init__()
        self._action = action
        self._args = args
        self._kwargs = kwargs

    def act(self) -> SangoStatus:
        return self._action(*self._args, **self._kwargs)


class FCond(Condition):

    def __init__(self, cond: typing.Callable[[typing.Any], bool], *args, **kwargs) -> None:
        super().__init__()
        self._cond = cond
        self._args = args
        self._kwargs = kwargs

    def condition(self) -> bool:
        return self._cond(*self._args, **self._kwargs)


class FSerial(Serial):

    def __init__(self, iterable: typing.Callable[[typing.Any], bool], *args, stop_on: bool=SangoStatus.FAILURE, **kwargs) -> None:
        super().__init__(
            partial(iterable, *args, **kwargs), stop_on=stop_on
        )


class FParallel(Parallel):

    def __init__(self, iterable: typing.Callable[[typing.Any], bool], *args, stop_on: bool=SangoStatus.FAILURE, **kwargs) -> None:
        super().__init__(
            partial(iterable, *args, **kwargs), stop_on=stop_on
        )


# class Selector(Serial):

#     def add(self, task: Task) -> 'Sequence':
#         """Add task to the selector

#         Args:
#             task (Task): 

#         Returns:
#             Sequence: The sequence added to
#         """
#         self._tasks.append(task)
#         return self
    
#     def subtick(self) -> SangoStatus:
        
#         if self._status.is_done:
#             return self._status
    
#         task = self._tasks[self._idx]
#         status = task.tick()

#         if status == SangoStatus.SUCCESS:
#             return SangoStatus.SUCCESS
        
#         if status == SangoStatus.FAILURE:
#             self._idx += 1
#             status = SangoStatus.RUNNING
#         if self._idx >= len(self._tasks):
#             status = SangoStatus.FAILURE

#         return status

#     def reset(self):
#         super().reset()
#         self._idx = 0
        
