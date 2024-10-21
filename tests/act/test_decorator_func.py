from dachi.act import _core, _functional as F
from dachi.act import TaskStatus
from dachi.data import _data as utils
import typing
from ..core.test_ai import DummyAIModel
from dachi import _core as core
from dachi.data import Shared
import time
from dachi.act import _decorator_func as DF, _functional as F, _core as core


class TaskAgent:

    def __init__(self):
        
        self.data = Shared()

    @DF.taskfunc()
    def increment(self, x):
        x = x + 1
        if x < 0:
            return core.FAILURE
        if x < 5:
            return core.RUNNING
        return core.SUCCESS

    def convert(self, x):
        if x < 0:
            return core.FAILURE
        if x < 5:
            return core.RUNNING
        return core.SUCCESS

    @DF.taskfunc(out='data', to_status='convert')
    def increment2(self, x):
        x = x + 1
        return x


class TestTaskFunc:

    def test_task_func_returns_running(self):

        agent = TaskAgent()
        assert agent.increment.task(2)() == core.RUNNING

    def test_task_func_returns_success(self):

        agent = TaskAgent()
        assert agent.increment.task(4)() == core.SUCCESS

    def test_task_func_returns_failure(self):

        agent = TaskAgent()
        assert agent.increment.task(-2)() == core.FAILURE


    def test_task_func2_returns_running(self):
        agent = TaskAgent()
        assert agent.increment2.task(2)() == core.RUNNING

    def test_task_func2_returns_success(self):

        agent = TaskAgent()
        assert agent.increment2.task(4)() == core.SUCCESS

    def test_task_func2_returns_failure(self):

        agent = TaskAgent()
        agent.increment2.task(-2)()
        assert agent.data.get() == -1
