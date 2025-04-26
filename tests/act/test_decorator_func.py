from dachi.act import TaskStatus
import typing
from dachi import proc as core
from dachi.store import Shared, Context
from dachi.act import _decorator_func as DF, _functional as F, _core as core


class TaskAgent:

    def __init__(self):
        
        self.data = Shared()
        self.ctx = Context()

    @DF.taskmethod()
    def increment(self, x, reset: bool=False):
        x = x + 1
        if x < 0:
            return core.FAILURE
        if x < 5:
            return core.RUNNING
        return core.SUCCESS

    @DF.taskmethod()
    def decrement(self, x, reset: bool=False):
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

    @DF.taskmethod(out='data', to_status='convert')
    def increment2(self, x, reset: bool=False):
        x = x + 1
        return x
    
    @DF.condfunc()
    def is_nonnegative(self, x, reset: bool=False):
        return x >= 0

    @DF.sequencemethod()
    def increment_seq(self, ctx: Context, x, reset: bool=False) -> typing.Iterator[core.Task]:

        yield self.is_nonnegative(x)
        yield self.increment(x)

    @DF.selectormethod()
    def increment_sel(self, ctx: Context, x, reset: bool=False) -> typing.Iterator[core.Task]:

        yield self.increment(x)
        yield self.decrement(x)

    @DF.parallelmethod()
    def increment_parallel(self, x, y, reset: bool=False) -> typing.Iterator[core.Task]:

        yield self.increment(x)
        yield self.increment(y)


class TestTaskFunc:

    def test_task_func_returns_running(self):

        agent = TaskAgent()
        assert agent.increment(2)() == core.RUNNING

    def test_task_func_returns_success(self):

        agent = TaskAgent()
        assert agent.increment(4)() == core.SUCCESS

    def test_task_func_returns_failure(self):

        agent = TaskAgent()
        assert agent.increment(-2)() == core.FAILURE

    def test_task_func2_returns_running(self):
        agent = TaskAgent()
        assert agent.increment2(2)() == core.RUNNING

    def test_task_func2_returns_success(self):

        agent = TaskAgent()
        assert agent.increment2(4)() == core.SUCCESS

    def test_task_func2_returns_failure(self):

        agent = TaskAgent()
        agent.increment2(-2)()
        assert agent.data.get() == -1

    def test_taskfunc_increment_success(self):
        agent = TaskAgent()
        assert agent.increment(5)() == core.SUCCESS

    def test_taskfunc_increment_running(self):
        agent = TaskAgent()
        assert agent.increment(3)() == core.RUNNING

    def test_taskfunc_decrement_success(self):
        agent = TaskAgent()
        assert agent.decrement(-1)() == core.SUCCESS

    def test_taskfunc_decrement_running(self):
        agent = TaskAgent()
        assert agent.decrement(6)() == core.RUNNING

    def test_taskfunc_decrement_failure(self):
        agent = TaskAgent()
        assert agent.decrement(10)() == core.FAILURE

    def test_taskfunc_increment2_success(self):
        agent = TaskAgent()
        assert agent.increment2(4)() == core.SUCCESS

    def test_taskfunc_increment2_running(self):
        agent = TaskAgent()
        assert agent.increment2(2)() == core.RUNNING

    def test_taskfunc_increment2_failure(self):
        agent = TaskAgent()
        agent.increment2(-2)()
        assert agent.data.get() == -1


class TestCondFunc:

    def test_task_func_returns_success(self):

        agent = TaskAgent()
        assert agent.is_nonnegative(2)() == core.SUCCESS

    def test_task_func_returns_success_with_0(self):

        agent = TaskAgent()
        assert agent.is_nonnegative(0)() == core.SUCCESS

    def test_task_func_returns_failure_with_neg_1(self):

        agent = TaskAgent()
        assert agent.is_nonnegative(-1)() == core.FAILURE

    def test_condfunc_success(self):
        agent = TaskAgent()
        assert agent.is_nonnegative(3)() == core.SUCCESS

    def test_condfunc_success_with_zero(self):
        agent = TaskAgent()
        assert agent.is_nonnegative(0)() == core.SUCCESS

    def test_condfunc_failure(self):
        agent = TaskAgent()
        assert agent.is_nonnegative(-1)() == core.FAILURE


class TestSeqFunc:

    def test_task_func_returns_running2(self):

        agent = TaskAgent()
        t = agent.increment_seq(Context(), 4)
        status = t()
        assert status == TaskStatus.RUNNING

    def test_task_func_returns_success_after_2(self):

        agent = TaskAgent()
        t = agent.increment_seq(Context(), 4)
        status = t()
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_running_after_2(self):

        agent = TaskAgent()
        t = agent.increment_seq(Context(), 2)
        status = t()
        status = t()
        assert status == TaskStatus.RUNNING

    def test_task_func_returns_running_after_1(self):

        agent = TaskAgent()
        t = agent.increment_seq(Context(), -1)
        status = t()
        assert status == TaskStatus.FAILURE

    def test_sequencefunc_all_tasks_success(self):
        agent = TaskAgent()
        t = agent.increment_seq(Context(), 5)
        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_sequencefunc_task_failure(self):
        agent = TaskAgent()
        t = agent.increment_seq(Context(), -1)
        status = t()
        assert status == TaskStatus.FAILURE

    def test_sequencefunc_partial_success(self):
        agent = TaskAgent()
        t = agent.increment_seq(Context(), 2)
        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.RUNNING


class TestSelectorFunc:

    def test_task_func_returns_success_after1(self):

        agent = TaskAgent()
        t = agent.increment_sel(Context(), 4)
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_success_after2(self):

        agent = TaskAgent()
        t = agent.increment_sel(Context(), -2)

        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_running_after2(self):

        agent = TaskAgent()
        t = agent.increment_sel(Context(), 1)

        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.RUNNING

    def test_selectorfunc_first_task_success(self):
        agent = TaskAgent()
        t = agent.increment_sel(Context(), 5)
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_selectorfunc_second_task_success(self):
        agent = TaskAgent()
        t = agent.increment_sel(Context(), -2)
        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_selectorfunc_all_tasks_running(self):
        agent = TaskAgent()
        t = agent.increment_sel(Context(), 2)
        status = t()
        assert status == TaskStatus.RUNNING
        status = t()
        assert status == TaskStatus.RUNNING

    def test_selectorfunc_first_task_failure(self):
        agent = TaskAgent()
        t = agent.increment_sel(Context(), -5)
        status = t()
        status = t()
        assert status == TaskStatus.SUCCESS


class TestParallelFunc:

    def test_parallelfunc_all_tasks_success(self):
        agent = TaskAgent()
        t = agent.increment_parallel(5, 5)
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_parallelfunc_one_task_running(self):
        agent = TaskAgent()
        t = agent.increment_parallel(5, 2)
        status = t()
        assert status == TaskStatus.RUNNING

    def test_parallelfunc_one_task_failure(self):
        agent = TaskAgent()
        t = agent.increment_parallel(4, -2)
        status = t()
        assert status == TaskStatus.FAILURE

    def test_parallelfunc_all_tasks_running(self):
        agent = TaskAgent()
        t = agent.increment_parallel(2, 2)
        status = t()
        assert status == TaskStatus.RUNNING

    # def test_parallelfunc_mixed_status_with_failure_priority(self):
    #     agent = TaskAgent()
    #     t = DF.parallelfunc(fails_on=1, success_priority=False)(agent.increment_parallel)(5, -2)
    #     status = t()
    #     assert status == TaskStatus.FAILURE

    # def test_parallelfunc_custom_succeeds_on(self):
    #     agent = TaskAgent()
    #     t = DF.parallelfunc(succeeds_on=2)(agent.increment_parallel)(5, 5)
    #     status = t()
    #     assert status == TaskStatus.SUCCESS

    # def test_parallelfunc_custom_fails_on(self):
    #     agent = TaskAgent()
    #     t = DF.parallelfunc(fails_on=2)(agent.increment_parallel)(Context(), -2, -2)
    #     status = t()
    #     assert status == TaskStatus.FAILURE

    def test_task_func_returns_running(self):

        agent = TaskAgent()
        t = agent.increment_parallel(5, 2)
        status = t()
        assert status == TaskStatus.RUNNING

    def test_task_func_returns_running(self):

        agent = TaskAgent()
        t = agent.increment_parallel(5, 5)
        status = t()
        assert status == TaskStatus.SUCCESS

    def test_task_func_returns_failure(self):

        agent = TaskAgent()
        t = agent.increment_parallel(4, -2)
        status = t()
        assert status == TaskStatus.FAILURE
