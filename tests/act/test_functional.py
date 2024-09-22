from dachi.act import _core, _functional as F
import typing


def sample_action(state: typing.Dict, x: int) -> _core.TaskStatus:

    val = _core.get_or_set(state, 'val', 0)
    state['val'] = x + val

    if state['val'] < 3:
        return _core.TaskStatus.RUNNING
    return _core.TaskStatus.SUCCESS


def sample_cond(x: int) -> bool:

    return x > 3


class TestAction:

    def test_action_returns_in_progress(self):
        state = {}

        status = sample_action(state, 2)
        assert status.in_progress

    def test_action_returns_success(self):
        state = {}

        status = sample_action(state, 2)
        status2 = sample_action(state, 3)
        assert status2.success
    

class TestCond:

    def test_cond_returns_failure(self):

        status = F.tick(F.cond(sample_cond, 2))
        assert status.failure

    def test_cond_returns_success(self):

        status = F.tick(F.cond(sample_cond, 4))
        assert status.success


class TestSequence:

    def test_sequence_executes_and_returns_running(self):
        state = _core.ContextStorage()

        status = F.tick(F.sequence([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S))

        print(status)
        assert status.running

    def test_sequence_executes_and_returns_success(self):
        state = _core.ContextStorage()

        status = F.sequence([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        status = F.sequence([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()
        assert status.success

    def test_returns_success_if_no_tasks(self):
        state = _core.ContextStorage()

        status = F.sequence([
        ], state.S)()

        assert status.success


    def test_sequence_executes_and_returns_failure(self):
        state = _core.ContextStorage()

        status = F.sequence([
            F.cond(sample_cond, 0),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        assert status.failure


class TestSelector:

    def test_selector_executes_and_returns_running(self):
        state = _core.ContextStorage()

        status = F.selector([
            F.cond(sample_cond, 2),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        assert status.running

    def test_selector_executes_and_returns_success(self):
        state = _core.ContextStorage()

        status = F.selector([
            F.cond(sample_cond, 2),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        status = F.selector([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()
        assert status.success

    def test_returns_failure_if_no_tasks(self):
        state = _core.ContextStorage()

        status = F.selector([
        ], state.S)()

        assert status.failure

    def test_selector_executes_and_returns_success(self):
        state = _core.ContextStorage()

        status = F.selector([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        assert status.success


class TestParallel:

    def test_parallel_returns_failure_if_one_fails(self):
        state = _core.ContextStorage()

        status = F.parallel([
            F.cond(sample_cond, 2),
            F.action(sample_action, state.A, 4)
        ])()

        assert status.failure

    def test_parallel_returns_success_if_both_succeed(self):
        state = _core.ContextStorage()

        status = F.parallel([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ])()

        assert status.success

    def test_parallel_returns_running_if_one_running(self):
        state = _core.ContextStorage()

        status = F.parallel([
            F.cond(sample_cond, 4),
            F.action(sample_action, state.A, 2)
        ])()

        assert status.running

    def test_parallel_returns_running_with_nested_sequence(self):
        state = _core.ContextStorage()

        status = F.parallel([
            F.cond(sample_cond, 4),
            F.sequence(
                [F.action(sample_action, state.A, 2)], state.S
            )
        ])()

        assert status.running


class TestUnless:

    def test_unless_returns_failure_if_failed(self):
        state = _core.ContextStorage()

        status = F.unless(
            F.sequence([
                F.cond(sample_cond, 2),
                F.action(sample_action, state.A, 4)
            ], state.S)
        , state)()

        assert status.failure

    def test_unless_returns_running_if_succeeded(self):
        state = _core.ContextStorage()

        status = F.unless(
            F.sequence([
                F.cond(sample_cond, 4),
                F.action(sample_action, state.A, 4)
            ], state.S)
        )()

        status = F.unless(
            F.sequence([
                F.cond(sample_cond, 4),
                F.action(sample_action, state.S, 4)
            ], state.A)
        )()

        assert status.running


class TestUntil:

    def test_until_returns_running_if_failed(self):
        state = _core.ContextStorage()

        status = F.until(
            F.sequence([
                F.cond(sample_cond, 2),
                F.action(sample_action, 4)
            ], state.S)
        )()

        assert status.running

    def test_unless_returns_success_if_succeeded(self):
        state = _core.ContextStorage()

        status = F.until(
            F.sequence([
                F.cond(sample_cond, 4),
                F.action(sample_action, state.A, 4)
            ],state.S)
        )()

        status = F.until(
            F.sequence([
                F.cond(sample_cond, 4),
                F.action(sample_action, state.A, 4)
            ], state.S)
        )()

        assert status.success


class TestNot:

    def test_not_returns_success_if_failed(self):

        status = F.not_(
            F.cond(sample_cond, 2),
        )()

        assert status.success


    def test_not_returns_failed_if_success(self):

        status = F.tick(F.not_(
            F.cond(sample_cond, 4),
        ))

        assert status.failure
