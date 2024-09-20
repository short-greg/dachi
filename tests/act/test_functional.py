from dachi.act import _core, _functional
import typing


def sample_action(state: typing.Dict, x: int) -> _core.TaskStatus:

    val = _core.get_or_set(state, 'val', 0)
    state['val'] = x + val

    if state['val'] < 3:
        return _core.TaskStatus.RUNNING
    return _core.TaskStatus.SUCCESS


def sample_cond(state: typing.Dict, x: int) -> bool:

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
        state = {}

        status = _functional.cond(sample_cond, state, 2)
        assert status.failure

    def test_cond_returns_success(self):
        state = {}

        status = _functional.cond(sample_cond, state, 4)
        assert status.success


class TestSequence:

    def test_sequence_executes_and_returns_running(self):
        state = {}

        status = _functional.sequence([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 4)
        ], state)

        print(status)
        assert status.running

    def test_sequence_executes_and_returns_success(self):
        state = {}

        status = _functional.sequence([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 4)
        ], state)

        status = _functional.sequence([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 4)
        ], state)
        assert status.success

    def test_returns_success_if_no_tasks(self):
        state = {}

        status = _functional.sequence([
        ], state)

        assert status.success


    def test_sequence_executes_and_returns_failure(self):
        state = {}

        status = _functional.sequence([
            _functional.nest_cond(sample_cond, 0),
            _functional.nest_action(sample_action, 4)
        ], state)

        assert status.failure
    # def test_cond_returns_success(self):
    #     state = {}

    #     status = _functional.cond(sample_cond, state, 4)
    #     assert status.success


class TestSelector:

    def test_selector_executes_and_returns_running(self):
        state = {}

        status = _functional.selector([
            _functional.nest_cond(sample_cond, 2),
            _functional.nest_action(sample_action, 4)
        ], state)

        assert status.running

    def test_selector_executes_and_returns_success(self):
        state = {}

        status = _functional.selector([
            _functional.nest_cond(sample_cond, 2),
            _functional.nest_action(sample_action, 4)
        ], state)

        status = _functional.selector([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 4)
        ], state)
        assert status.success

    def test_returns_failure_if_no_tasks(self):
        state = {}

        status = _functional.selector([
        ], state)

        assert status.failure

    def test_selector_executes_and_returns_success(self):
        state = {}

        status = _functional.selector([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 4)
        ], state)

        assert status.success


class TestParallel:

    def test_parallel_returns_failure_if_one_fails(self):
        state = {}

        status = _functional.parallel([
            _functional.nest_cond(sample_cond, 2),
            _functional.nest_action(sample_action, 4)
        ], state)

        assert status.failure

    def test_parallel_returns_success_if_both_succeed(self):
        state = {}

        status = _functional.parallel([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 4)
        ], state)

        assert status.success


    def test_parallel_returns_running_if_one_running(self):
        state = {}

        status = _functional.parallel([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_action(sample_action, 2)
        ], state)

        assert status.running

    def test_parallel_returns_running_with_nested_sequence(self):
        state = {}

        status = _functional.parallel([
            _functional.nest_cond(sample_cond, 4),
            _functional.nest_sequence(
                [_functional.nest_action(sample_action, 2)]
            )
        ], state)

        assert status.running


class TestUnless:

    def test_unless_returns_failure_if_failed(self):
        state = {}

        status = _functional.unless(
            _functional.nest_sequence([
                _functional.nest_cond(sample_cond, 2),
                _functional.nest_action(sample_action, 4)
            ])
        , state)

        assert status.failure

    def test_unless_returns_running_if_succeeded(self):
        state = {}

        status = _functional.unless(
            _functional.nest_sequence([
                _functional.nest_cond(sample_cond, 4),
                _functional.nest_action(sample_action, 4)
            ])
        , state)

        status = _functional.unless(
            _functional.nest_sequence([
                _functional.nest_cond(sample_cond, 4),
                _functional.nest_action(sample_action, 4)
            ])
        , state)

        assert status.running


class TestUntil:

    def test_until_returns_running_if_failed(self):
        state = {}

        status = _functional.until(
            _functional.nest_sequence([
                _functional.nest_cond(sample_cond, 2),
                _functional.nest_action(sample_action, 4)
            ])
        , state)

        assert status.running

    def test_unless_returns_success_if_succeeded(self):
        state = {}

        status = _functional.until(
            _functional.nest_sequence([
                _functional.nest_cond(sample_cond, 4),
                _functional.nest_action(sample_action, 4)
            ])
        , state)

        status = _functional.until(
            _functional.nest_sequence([
                _functional.nest_cond(sample_cond, 4),
                _functional.nest_action(sample_action, 4)
            ])
        , state)

        assert status.success


class TestNot:

    def test_not_returns_success_if_failed(self):
        state = {}

        status = _functional.not_(
            _functional.nest_cond(sample_cond, 2),
        state)

        assert status.success


    def test_not_returns_failed_if_success(self):
        state = {}

        status = _functional.not_(
            _functional.nest_cond(sample_cond, 4),
        state)

        assert status.failure
