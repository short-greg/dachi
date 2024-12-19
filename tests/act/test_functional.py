from dachi.act import _core, _functional as F
from dachi.act import TaskStatus
from dachi.data import _data as utils
import typing
from ..ai.test_ai import DummyAIModel
from dachi import _core as core
import time


def sample_action(state: typing.Dict, x: int) -> _core.TaskStatus:

    val = utils.get_or_set(state, 'val', 0)
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

        status = F.tick(F.condf(sample_cond, 2))
        assert status.failure

    def test_cond_returns_success(self):

        status = F.tick(F.condf(sample_cond, 4))
        assert status.success


class TestSequence:

    def test_sequence_executes_and_returns_running(self):
        state = utils.ContextStorage()

        status = F.tick(F.sequence([
            F.condf(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S))

        print(status)
        assert status.running

    def test_sequence_executes_and_returns_success(self):
        state = utils.ContextStorage()

        status = F.sequence([
            F.condf(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        status = F.sequence([
            F.condf(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()
        assert status.success

    def test_returns_success_if_no_tasks(self):
        state = utils.ContextStorage()

        status = F.sequence([
        ], state.S)()

        assert status.success


    def test_sequence_executes_and_returns_failure(self):
        state = utils.ContextStorage()

        status = F.sequence([
            F.condf(sample_cond, 0),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        assert status.failure


class TestSelector:

    def test_selector_executes_and_returns_running(self):
        state = utils.ContextStorage()

        status = F.selector([
            F.condf(sample_cond, 2),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        assert status.running

    def test_selector_executes_and_returns_success(self):
        state = utils.ContextStorage()

        status = F.selector([
            F.condf(sample_cond, 2),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        status = F.selector([
            F.condf(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()
        assert status.success

    def test_returns_failure_if_no_tasks(self):
        state = utils.ContextStorage()

        status = F.selector([
        ], state.S)()

        assert status.failure

    def test_selector_executes_and_returns_success(self):
        state = utils.ContextStorage()

        status = F.selector([
            F.condf(sample_cond, 4),
            F.action(sample_action, state.A, 4)
        ], state.S)()

        assert status.success


class TestParallel:

    def test_parallel_returns_failure_if_one_fails(self):
        state = utils.ContextStorage()

        status = F.parallel([
            F.condf(sample_cond, 2),
            F.action(sample_action, state.A, 4)
        ])()

        assert status.failure

    # def test_parallel_returns_success_if_both_succeed(self):
    #     state = utils.ContextStorage()

    #     status = F.parallel([
    #         F.condf(sample_cond, 4),
    #         F.action(sample_action, state.A, 4)
    #     ])()

    #     assert status.success

    # def test_parallel_returns_running_if_one_running(self):
    #     state = utils.ContextStorage()

    #     status = F.parallel([
    #         F.condf(sample_cond, 4),
    #         F.action(sample_action, state.A, 2)
    #     ])()

    #     assert status.running

    # def test_parallel_returns_running_with_nested_sequence(self):
    #     state = utils.ContextStorage()

    #     status = F.parallel([
    #         F.condf(sample_cond, 4),
    #         F.sequence(
    #             [F.action(sample_action, state.A, 2)], state.S
    #         )
    #     ])()

    #     assert status.running


class TestUnless:

    def test_unless_returns_failure_if_failed(self):
        state = utils.ContextStorage()

        status = F.unless(
            F.sequence([
                F.condf(sample_cond, 2),
                F.action(sample_action, state.A, 4)
            ], state.S)
        , state)()

        assert status.failure

    def test_unless_returns_running_if_succeeded(self):
        state = utils.ContextStorage()

        status = F.unless(
            F.sequence([
                F.condf(sample_cond, 4),
                F.action(sample_action, state.A, 4)
            ], state.S)
        )()

        status = F.unless(
            F.sequence([
                F.condf(sample_cond, 4),
                F.action(sample_action, state.S, 4)
            ], state.A)
        )()

        assert status.running


class TestUntil:

    def test_until_returns_running_if_failed(self):
        state = utils.ContextStorage()

        status = F.until(
            F.sequence([
                F.condf(sample_cond, 2),
                F.action(sample_action, 4)
            ], state.S)
        )()

        assert status.running

    def test_unless_returns_success_if_succeeded(self):
        state = utils.ContextStorage()

        status = F.until(
            F.sequence([
                F.condf(sample_cond, 4),
                F.action(sample_action, state.A, 4)
            ],state.S)
        )()

        status = F.until(
            F.sequence([
                F.condf(sample_cond, 4),
                F.action(sample_action, state.A, 4)
            ], state.S)
        )()

        assert status.success


class TestNot:

    def test_not_returns_success_if_failed(self):

        status = F.not_(
            F.condf(sample_cond, 2),
        )()

        assert status.success


    def test_not_returns_failed_if_success(self):

        status = F.tick(F.not_(
            F.condf(sample_cond, 4),
        ))

        assert status.failure


class TestBuffer:

    def test_buffer_returns_correct_Status(self):

        buffer = utils.Buffer()
        model = DummyAIModel()
        message = core.Message(
            source='user', data={'text': 'text'}
        )
        ctx = utils.Context()
        stream = F.stream_model(
            buffer, model, message, ctx, interval=1./400.
        )
        res = stream()
        time.sleep(0.1)
        res = stream()

        assert res == TaskStatus.SUCCESS

    def test_buffer_has_correct_value(self):

        buffer = utils.Buffer()
        model = DummyAIModel()
        message = core.Message(
            source='user', data={'text': 'text'}
        )
        ctx = utils.Context()
        stream = F.stream_model(
            buffer, model, message, ctx, interval=1./400.
        )
        stream()
        time.sleep(0.1)
        stream()
        res = ''.join((r.val for r in buffer.get()))

        assert res == 'Great!'


class TestSharedTask:

    def test_shared_task_returns_correct_status(self):

        shared = utils.Shared()
        model = DummyAIModel()
        message = core.Message(
            source='user', data={'text': 'text'}
        )
        ctx = utils.Context()
        stream = F.exec_model(
            shared, model, message, ctx, interval=1./400.
        )
        res = stream()
        time.sleep(0.1)
        res = stream()

        assert res == TaskStatus.SUCCESS

    def test_shared(self):

        shared = utils.Shared()
        model = DummyAIModel()
        message = core.Message(
            source='user', data={'text': 'text'}
        )
        ctx = utils.Context()
        stream = F.exec_model(
            shared, model, message, ctx, interval=1./400.
        )
        res = stream()
        time.sleep(0.1)
        res = stream()

        assert shared.data.val == 'Great!'
