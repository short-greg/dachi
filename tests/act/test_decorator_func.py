from dachi.act import _core, _functional as F
from dachi.act import TaskStatus
from dachi.data import _data as utils
import typing
from ..ai.test_ai import DummyAIModel
from dachi import _core as core
from dachi.data import Shared, Context
import time
from dachi.act import _decorator_func as DF, _functional as F, _core as core


class TaskAgent:

    def __init__(self):
        
        self.data = Shared()
        self.ctx = Context()

    @DF.taskfunc()
    def increment(self, x):
        x = x + 1
        if x < 0:
            return core.FAILURE
        if x < 5:
            return core.RUNNING
        return core.SUCCESS


    @DF.taskfunc()
    def decrement(self, x):
        x = x - 1
        if x < 0:
            return core.SUCCESS
        if x > 5:
            return core.FAILURE
        return core.RUNNING

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
    
    @DF.condfunc()
    def is_nonnegative(self, x):
        return x >= 0

    @DF.sequencefunc('ctx')
    def increment_seq(self, x) -> typing.Iterator[core.Task]:

        yield self.is_nonnegative.task(x)
        yield self.increment.task(x)

    @DF.selectorfunc('ctx')
    def increment_sel(self, x) -> typing.Iterator[core.Task]:

        yield self.increment.task(x)
        yield self.decrement.task(x)

    @DF.parallelfunc()
    def increment_parallel(self, x, y) -> typing.Iterator[core.Task]:

        yield self.increment.task(x)
        yield self.increment.task(y)


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


class TestCondFunc:

    def test_task_func_returns_success(self):

        agent = TaskAgent()
        assert agent.is_nonnegative.task(2)() == core.SUCCESS

    def test_task_func_returns_success_with_0(self):

        agent = TaskAgent()
        assert agent.is_nonnegative.task(0)() == core.SUCCESS

    def test_task_func_returns_failure_with_neg_1(self):

        agent = TaskAgent()
        assert agent.is_nonnegative.task(-1)() == core.FAILURE


class TestSeqFunc:

    def test_task_func_returns_running2(self):

        agent = TaskAgent()
        t = agent.increment_seq.task(4)
        status = t()
        assert status == TaskStatus.RUNNING

    def test_task_func_returns_success_after_2(self):

        agent = TaskAgent()
        t = agent.increment_seq.task(4)
        status = t()
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_running_after_2(self):

        agent = TaskAgent()
        t = agent.increment_seq.task(2)
        status = t()
        status = t()
        assert status == TaskStatus.RUNNING

    def test_task_func_returns_running_after_1(self):

        agent = TaskAgent()
        t = agent.increment_seq.task(-1)
        status = t()
        assert status == TaskStatus.FAILURE


class TestSelectorFunc:

    def test_task_func_returns_success_after1(self):

        agent = TaskAgent()
        t = agent.increment_sel.task(4)
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_success_after2(self):

        agent = TaskAgent()
        t = agent.increment_sel.task(-2)

        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_running_after2(self):

        agent = TaskAgent()
        t = agent.increment_sel.task(1)

        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.RUNNING


class TestParallelFunc:

    def test_task_func_returns_running(self):

        agent = TaskAgent()
        t = agent.increment_parallel.task(5, 2)
        status = t()
        assert status == TaskStatus.RUNNING

    def test_task_func_returns_running(self):

        agent = TaskAgent()
        t = agent.increment_parallel.task(5, 5)
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_failure(self):

        agent = TaskAgent()
        t = agent.increment_parallel.task(4, -2)
        status = t()
        assert status == TaskStatus.FAILURE
