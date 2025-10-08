"""Unit tests for StateChart base components.

Tests cover ChartStatus and ChartBase classes following the framework
testing conventions.
"""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dachi', 'act', '_chart'))
import _base

ChartStatus = _base.ChartStatus
ChartBase = _base.ChartBase
InvalidTransition = _base.InvalidTransition


class TestChartStatus:

    def test_waiting_value_is_waiting(self):
        assert ChartStatus.WAITING.value == "waiting"

    def test_running_value_is_running(self):
        assert ChartStatus.RUNNING.value == "running"

    def test_preempting_value_is_preempting(self):
        assert ChartStatus.PREEMPTING.value == "preempting"

    def test_success_value_is_success(self):
        assert ChartStatus.SUCCESS.value == "success"

    def test_canceled_value_is_canceled(self):
        assert ChartStatus.CANCELED.value == "canceled"

    def test_failure_value_is_failure(self):
        assert ChartStatus.FAILURE.value == "failure"

    def test_is_waiting_returns_true_when_waiting(self):
        assert ChartStatus.WAITING.is_waiting() is True

    def test_is_waiting_returns_false_when_not_waiting(self):
        assert ChartStatus.RUNNING.is_waiting() is False

    def test_is_running_returns_true_when_running(self):
        assert ChartStatus.RUNNING.is_running() is True

    def test_is_running_returns_false_when_not_running(self):
        assert ChartStatus.SUCCESS.is_running() is False

    def test_is_preempting_returns_true_when_preempting(self):
        assert ChartStatus.PREEMPTING.is_preempting() is True

    def test_is_preempting_returns_false_when_not_preempting(self):
        assert ChartStatus.RUNNING.is_preempting() is False

    def test_is_success_returns_true_when_success(self):
        assert ChartStatus.SUCCESS.is_success() is True

    def test_is_success_returns_false_when_not_success(self):
        assert ChartStatus.RUNNING.is_success() is False

    def test_is_canceled_returns_true_when_canceled(self):
        assert ChartStatus.CANCELED.is_canceled() is True

    def test_is_canceled_returns_false_when_not_canceled(self):
        assert ChartStatus.RUNNING.is_canceled() is False

    def test_is_failure_returns_true_when_failure(self):
        assert ChartStatus.FAILURE.is_failure() is True

    def test_is_failure_returns_false_when_not_failure(self):
        assert ChartStatus.RUNNING.is_failure() is False

    def test_is_completed_returns_true_for_success(self):
        assert ChartStatus.SUCCESS.is_completed() is True

    def test_is_completed_returns_true_for_failure(self):
        assert ChartStatus.FAILURE.is_completed() is True

    def test_is_completed_returns_true_for_canceled(self):
        assert ChartStatus.CANCELED.is_completed() is True

    def test_is_completed_returns_false_for_waiting(self):
        assert ChartStatus.WAITING.is_completed() is False

    def test_is_completed_returns_false_for_running(self):
        assert ChartStatus.RUNNING.is_completed() is False

    def test_is_completed_returns_false_for_preempting(self):
        assert ChartStatus.PREEMPTING.is_completed() is False


class TestChartBase:

    def test_post_init_sets_name_to_class_name_when_none(self):
        base = ChartBase()
        assert base.name == "ChartBase"

    def test_post_init_preserves_custom_name(self):
        base = ChartBase(name="CustomName")
        assert base.name == "CustomName"

    def test_post_init_initializes_status_to_waiting(self):
        base = ChartBase()
        assert base.get_status() == ChartStatus.WAITING

    def test_post_init_initializes_empty_finish_callbacks(self):
        base = ChartBase()
        assert base._finish_callbacks == {}

    def test_get_status_returns_current_status(self):
        base = ChartBase()
        assert base.get_status() == ChartStatus.WAITING

    def test_can_reset_returns_true_for_success(self):
        base = ChartBase()
        base._status.set(ChartStatus.SUCCESS)
        assert base.can_reset() is True

    def test_can_reset_returns_true_for_failure(self):
        base = ChartBase()
        base._status.set(ChartStatus.FAILURE)
        assert base.can_reset() is True

    def test_can_reset_returns_true_for_canceled(self):
        base = ChartBase()
        base._status.set(ChartStatus.CANCELED)
        assert base.can_reset() is True

    def test_can_reset_returns_false_for_running(self):
        base = ChartBase()
        base._status.set(ChartStatus.RUNNING)
        assert base.can_reset() is False

    def test_can_reset_returns_false_for_waiting(self):
        base = ChartBase()
        assert base.can_reset() is False

    def test_register_finish_callback_adds_callback_without_args(self):
        base = ChartBase()
        callback = lambda: None
        base.register_finish_callback(callback)
        assert callback in base._finish_callbacks

    def test_register_finish_callback_adds_callback_with_args(self):
        base = ChartBase()
        callback = lambda x: None
        base.register_finish_callback(callback, "arg1")
        assert base._finish_callbacks[callback] == (("arg1",), {})

    def test_register_finish_callback_adds_callback_with_kwargs(self):
        base = ChartBase()
        callback = lambda x=None: None
        base.register_finish_callback(callback, key="value")
        assert base._finish_callbacks[callback] == ((), {"key": "value"})

    def test_register_finish_callback_adds_callback_with_args_and_kwargs(self):
        base = ChartBase()
        callback = lambda x, y=None: None
        base.register_finish_callback(callback, "arg1", key="value")
        assert base._finish_callbacks[callback] == (("arg1",), {"key": "value"})

    def test_register_finish_callback_replaces_on_duplicate(self):
        base = ChartBase()
        callback = lambda: None
        base.register_finish_callback(callback, "first")
        base.register_finish_callback(callback, "second")
        assert base._finish_callbacks[callback] == (("second",), {})
        assert len(base._finish_callbacks) == 1

    def test_unregister_finish_callback_removes_callback(self):
        base = ChartBase()
        callback = lambda: None
        base.register_finish_callback(callback)
        base.unregister_finish_callback(callback)
        assert callback not in base._finish_callbacks

    def test_unregister_finish_callback_does_nothing_when_not_registered(self):
        base = ChartBase()
        callback = lambda: None
        base.unregister_finish_callback(callback)
        assert callback not in base._finish_callbacks


class TestChartBaseFinish:

    @pytest.mark.asyncio
    async def test_finish_does_not_change_status(self):
        base = ChartBase()
        await base.finish()
        assert base.get_status() == ChartStatus.WAITING

    @pytest.mark.asyncio
    async def test_finish_calls_registered_callback(self):
        base = ChartBase()
        called = False

        def callback():
            nonlocal called
            called = True

        base.register_finish_callback(callback)
        await base.finish()
        assert called is True

    @pytest.mark.asyncio
    async def test_finish_calls_callback_with_args(self):
        base = ChartBase()
        result = []

        def callback(value):
            result.append(value)

        base.register_finish_callback(callback, "test_value")
        await base.finish()
        assert result == ["test_value"]

    @pytest.mark.asyncio
    async def test_finish_calls_callback_with_kwargs(self):
        base = ChartBase()
        result = {}

        def callback(key=None):
            result["key"] = key

        base.register_finish_callback(callback, key="value")
        await base.finish()
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_finish_calls_callback_with_args_and_kwargs(self):
        base = ChartBase()
        result = {}

        def callback(arg, key=None):
            result["arg"] = arg
            result["key"] = key

        base.register_finish_callback(callback, "arg_value", key="kwarg_value")
        await base.finish()
        assert result == {"arg": "arg_value", "key": "kwarg_value"}

    @pytest.mark.asyncio
    async def test_finish_calls_multiple_callbacks(self):
        base = ChartBase()
        results = []

        def callback1():
            results.append(1)

        def callback2():
            results.append(2)

        base.register_finish_callback(callback1)
        base.register_finish_callback(callback2)
        await base.finish()
        assert 1 in results
        assert 2 in results
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_finish_with_no_callbacks_completes(self):
        base = ChartBase()
        await base.finish()
        assert base.get_status() == ChartStatus.WAITING
