from dachi.act._status import TaskStatus
from dachi.act import actionfunc
from dachi.act._func_decorators import (
    actionfunc, taskf, condfunc,
    selectorfunc, sequencefunc,
    parallelfunc
)

# parallelf

@actionfunc
def x(val1):

    if val1 == 1:
        yield TaskStatus.SUCCESS

    yield TaskStatus.RUNNING
    yield TaskStatus.SUCCESS


class TestTaskFuncs:

    @actionfunc
    def simple_action(self, x: int):
        yield TaskStatus.SUCCESS

    @actionfunc
    def complex_action(self, x: int):
        yield TaskStatus.RUNNING
        if x > 0:
            yield TaskStatus.SUCCESS
        else:
            yield TaskStatus.FAILURE

    @condfunc
    def simple_cond(self, x: int):
        return False

    @condfunc
    def simple_cond(self, x: int):
        return False

    @sequencefunc
    def simple_sequence(self, x: int):
        
        yield taskf(self.simple_action, x)
        yield taskf(self.simple_action, x)
        yield taskf(self.simple_cond, x)

    @selectorfunc
    def simple_selector(self, x: int):
        
        yield taskf(self.simple_action, x)
        yield taskf(self.simple_action, x)
        yield taskf(self.simple_cond, x)

    @parallelfunc()
    def simple_parallel(self, x: int):
        
        yield taskf(self.simple_action, x)
        yield taskf(self.simple_action, x)
        yield taskf(self.simple_action, x)

    def test_action_func(self):
        action = taskf(self.simple_action, 1)

        assert action.tick() == TaskStatus.SUCCESS

    def test_cond_func(self):
        action = taskf(self.complex_action, 1)

        assert action.tick() == TaskStatus.RUNNING

    def test_complex_action(self):
        action = taskf(self.complex_action, 1)

        assert action.tick() == TaskStatus.RUNNING
        assert action.tick() == TaskStatus.SUCCESS

    def test_complex_action2(self):
        action = taskf(self.complex_action, -1)

        assert action.tick() == TaskStatus.RUNNING
        assert action.tick() == TaskStatus.FAILURE

    def test_sequence_returns_failure(self):
        sequence = taskf(self.simple_sequence, 1)
        assert sequence.tick() == TaskStatus.RUNNING
        assert sequence.tick() == TaskStatus.RUNNING
        assert sequence.tick() == TaskStatus.FAILURE

    def test_selector_returns_success(self):
        selector = taskf(self.simple_selector, 1)
        assert selector.tick() == TaskStatus.SUCCESS

    def test_parallel_returns_success(self):
        parallel = taskf(self.simple_parallel, 1)
        assert parallel.tick() == TaskStatus.SUCCESS