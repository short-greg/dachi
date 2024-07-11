# 1st party
from abc import abstractmethod
import typing
from dataclasses import dataclass
import threading

# 3rd party
from .._core import Storable
from ._status import TaskStatus


@dataclass
class TaskMessage:

    name: str
    data: typing.Any


class Task(Storable):
    """The base class for a task in the behavior tree
    """

    SUCCESS = TaskStatus.SUCCESS
    FAILURE = TaskStatus.FAILURE
    RUNNING = TaskStatus.RUNNING

    def __init__(self) -> None:
        """Create the task

        Args:
            name (str): The name of the task
        """
        super().__init__()
        self._status = TaskStatus.READY

    @abstractmethod    
    def tick(self) -> TaskStatus:
        raise NotImplementedError

    def __call__(self) -> TaskStatus:
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
        self._status = TaskStatus.READY
    
    @property
    def status(self) -> TaskStatus:
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

    def tick(self) -> TaskStatus:
        """Update the task

        Returns:
            SangoStatus: The status after tick
        """
        if self._root is None:
            return TaskStatus.SUCCESS

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
        self, tasks: typing.Iterable[Task], 
        stop_on: TaskStatus=TaskStatus.FAILURE,
        complete_with: TaskStatus=TaskStatus.SUCCESS
    ):
        super().__init__()
        self._tasks = tasks if tasks is not None else []
        self._cur = None
        self._iter = None
        self._stop_on = stop_on
        self._ticked = set()
        self._complete_with = complete_with

    @property
    def n(self):
        """The number of subtasks"""
        return len(self._tasks)
    
    @property
    def tasks(self):
        """The subtasks"""
        return self._tasks

    def _advance(self):

        if self._iter is None:
            
            if isinstance(self._tasks, typing.Iterable):
                tasks = self._tasks
            else:
                tasks = self._tasks()
            self._iter = iter(tasks)

        try:
            self._cur = next(self._iter)
            if not isinstance(self._cur, TaskStatus):
                self._ticked.add(self._cur)
            return False
        except StopIteration:
            self._cur = None
            return True
    
    def _initiate(self):

        if self._iter is None:

            if isinstance(self._tasks, typing.Iterable):
                tasks = self._tasks
            else:
                tasks = self._tasks()
            self._iter = iter(tasks)
            return self._advance()
        return False

    def subtick(self) -> TaskStatus:
        """Tick each subtask. Implement when implementing a new Composite task"""
        
        if self._status.is_done:
            return self._status
    
        # Not started yet?
        if self._initiate():
            return TaskStatus.SUCCESS
    
        if isinstance(self._cur, TaskStatus):
            status = self._cur
        else: 
            status = self._cur.tick()

        # Still in progress?
        if status == TaskStatus.RUNNING:
            return TaskStatus.RUNNING
    
        # Reached a stopping condition?
        if (status == self._stop_on):
            self._cur = None
            return status
        
        if self._advance():
            return self._complete_with

        return TaskStatus.RUNNING

    def tick(self) -> TaskStatus:
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
        super().__init__(tasks, TaskStatus.FAILURE)


class Selector(Serial):

    def __init__(self, tasks: typing.Iterable[Task]):
        super().__init__(tasks, TaskStatus.SUCCESS, TaskStatus.FAILURE)


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

    @property
    def tasks(self) -> typing.Iterable[Task]:
        return self._tasks

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

    def _accumulate(self, statuses: typing.List[TaskStatus]) -> TaskStatus:
        
        successes = 0
        failures = 0
        # waiting = 0
        dones = 0
        for status in statuses:
            failures += status.failure
            successes += status.success

            dones += status.is_done
            # waiting += status.waiting

        has_failed = failures >= self._fails_on
        has_succeeded = successes >= self._succeeds_on
        if self._success_priority:
            if has_succeeded:
                return TaskStatus.SUCCESS
            if has_failed:
                return TaskStatus.FAILURE

        if has_failed:
            return TaskStatus.FAILURE
        if has_succeeded:
            return TaskStatus.SUCCESS
        # if waiting == (len(statuses) - dones):
        #     return TaskStatus.WAITING
        # failures + successes - 1
        return TaskStatus.RUNNING

    def subtick(self) -> TaskStatus:

        statuses = []
        if isinstance(self.tasks, typing.Iterable):
            tasks = self.tasks
        else: tasks = self.tasks()
        for task in tasks:
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
    
    def tick(self) -> TaskStatus:
        
        self._status = self.subtick()
        return self._status

    def reset(self):

        super().reset()
        for task in self._ticked:
            task.reset()
        self._ticked = set()


class Action(Task):

    @abstractmethod
    def act(self) -> TaskStatus:
        raise NotImplementedError

    def tick(self) -> TaskStatus:

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

    def tick(self) -> TaskStatus:
        """Check the condition

        Returns:
            SangoStatus: Whether the condition failed or succeeded
        """
        self._status = TaskStatus.SUCCESS if self.condition() else TaskStatus.FAILURE
        return self._status


class Decorator(Task):

    # name should retrieve the name of the decorated
    def __init__(self, task: Task) -> None:
        super().__init__()
        self._task = task

    @abstractmethod
    def decorate(self, status: TaskStatus) -> bool:
        pass

    @property
    def task(self) -> Task:
        return self._task

    def tick(self) -> TaskStatus:
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

    def decorate(self, status: TaskStatus) -> TaskStatus:
        """Continue running unless the result is a success

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status.success:
            return TaskStatus.SUCCESS
        if status.failure:
            self._task.reset()
            return TaskStatus.RUNNING
        return status


class Unless(Decorator):
    """Loop while a condition is met
    """

    def decorate(self, status: TaskStatus) -> TaskStatus:
        """Continue running unless the result is a failure

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status.failure:
            return TaskStatus.FAILURE
        if status.success:
            self._task.reset()
            return TaskStatus.RUNNING
        return status


class Not(Decorator):
    """Invert the result
    """

    def decorate(self, status: TaskStatus) -> TaskStatus:
        """Return Success if status is a Failure or Failure if it is a SUCCESS

        Args:
            status (SangoStatus): The status of the decorated task

        Returns:
            SangoStatus: The decorated status
        """
        if status.failure:
            return TaskStatus.SUCCESS
        if status.success:
            return TaskStatus.FAILURE
        return status


# class Check(Condition):

#     def __init__(self, data: Retrieve, f: Retrieve):
#         super().__init__()
#         self.data = data
#         self.f = f

#     def condition(self) -> bool:
#         return self.f(self.data())


# class CheckReady(Check):

#     def __init__(self, r: SRetrieve):
#         super().__init__(r, lambda r: r() is not None)


# class CheckTrue(Check):

#     def __init__(self, r: SRetrieve):
#         super().__init__(r, lambda r: r is True)


# class CheckFalse(Check):

#     def __init__(self, r: SRetrieve):
#         super().__init__(r, lambda r: r is False or r is None)


# class Reset(Action):

#     def __init__(self, data: Retrieve[Struct], on_condition: typing.Callable[[], None]=None):

#         super().__init__()
#         self._data = data
#         self._cond = on_condition

#     def act(self) -> TaskStatus:
#         data = self._data()

#         if self._cond is None or self._cond():
#             data.reset()
#             return TaskStatus.SUCCESS
#         return TaskStatus.FAILURE


# class TakoTask(Action):

#     def __init__(self, tako: TakoBase):

#         super().__init__()
#         self._tako = tako
#         self._executed = False

#     def exec(self):
#         self._tako()

#     def act(self) -> TaskStatus:
        
#         if self._status == self.READY:
#             thread = threading.Thread(target=self.exec, args=[])
#             thread.start()

#         if self._executed:
#             return self.SUCCESS
        
#         return self.RUNNING


# class Converse(Action):

#     def __init__(
#         self, prompt_conv: PromptConv, llm: OpenAIRequest, user_interface: UI,
#         **prompt_components: Struct
#     ):
#         # Use to send a message from the LLM
#         super().__init__()
#         self._conv = prompt_conv
#         self._llm_query = llm
#         self._ui_query = UIQuery(user_interface)
#         self._ui = user_interface
#         self._request = Request()
#         self._components = prompt_components

#     def converse_turn(self):

#         self._request = Request()
#         conv = self._conv.with_components(**self._components)
#         response = self._llm_query(conv, asynchronous=False)
#         self._processed  = self._response_processor.process(response)

#         if self._processed.to_interrupt:
#             return
#         self._ui.post_message('assistant', self._processed.text)

#         self._conv.add_turn(
#             'assistant', self._processed.text
#         )
#         self._request.contents = self._conv
#         self._ui_query.post(self._request)

#     def act(self) -> TaskStatus:
        
#         if self._status == TaskStatus.READY:
#             # use threading here
#             thread = threading.Thread(
#                 target=self.converse_turn, args=[]
#             )
#             thread.start()

#         if self._processed is not None and (
#             self._request.responded is True or self._processed.to_interrupt
#         ):
            
#             if not self._processed.to_interrupt and self._request.success is True:
#                 self._conv.add_turn(
#                     'user', self._request.response
#                 )
#             if self._processed.succeeded:
#                 print('SUCCEEDED!')
#                 return TaskStatus.SUCCESS
#             else:
#                 print('FAILED!')
#                 return TaskStatus.FAILURE
        
#         return TaskStatus.RUNNING

#     def reset(self):
#         super().reset()
#         self._interrupt = False
#         self._request = Request()


# class PromptCompleter(Action):
#     # Use to send a message from the LLM

#     def __init__(
#         self, completion: Completion, llm: OpenAIRequest, user_interface: UI,
#         post_processor: typing.Callable=None, **prompt_components: Struct
#     ):
#         """

#         Args:
#             completion (Completion): 
#             llm (Query): 
#             user_interface (UI): 
#         """
#         super().__init__()
#         self._completion = completion
#         self._llm_query = llm
#         self._ui = user_interface
#         self._request = Request()
#         self._post_processor = post_processor
#         self._components = prompt_components

#     def respond(self):

#         self._request = Request()
#         completion = self._completion.with_components(**self._components)
#         response = self._llm_query(completion, asynchronous=False)
#         self._ui.post_message('assistant', response)
#         self._completion.response = response
#         if self._post_processor is not None:
#             self._post_processor()
#         self._request.respond(response)

#     def act(self) -> TaskStatus:
        
#         if self._status == TaskStatus.READY:
#             # use threading here
#             thread = threading.Thread(target=self.respond, args=[])
#             thread.start()

#         if self._request.responded is True:
#             # self._request.status # wouldn't this be easier? 
#             if self._request.success is True:
#                 return TaskStatus.SUCCESS
#             else:
#                 return TaskStatus.FAILURE
        
#         return TaskStatus.RUNNING

#     def reset(self):
#         super().reset()
#         self._request = Request()
