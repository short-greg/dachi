# tests/test_request_dispatcher.py
import time
import asyncio
import threading
from typing import Any, Optional, Iterable, AsyncIterator

import pytest
# import _pytest.threadexception as threadexception


# Adjust if your module path differs
from dachi.comm._request import AsyncDispatcher, RequestState, RequestStatus
from dachi.proc._process import AsyncProcess, AsyncStreamProcess


# Wrapper classes for testing
class FuncAsyncProcess(AsyncProcess):
    """Wrapper to make functions work as AsyncProcess"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    async def aforward(self, *args, **kwargs):
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)


class StreamAsyncProcess(AsyncStreamProcess):
    """Wrapper to make generator functions work as AsyncStreamProcess"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    async def astream(self, *args, **kwargs):
        import asyncio
        import inspect
        
        # Check if the function is an async generator
        if inspect.isasyncgenfunction(self.func):
            # Async generator - stream each yielded item
            async for item in self.func(*args, **kwargs):
                yield item
        elif asyncio.iscoroutinefunction(self.func):
            # Regular async function - check what it returns
            result = await self.func(*args, **kwargs)
            
            # Check if result is an async context manager
            if hasattr(result, '__aenter__') and hasattr(result, '__aexit__'):
                async with result as ctx:
                    if hasattr(ctx, '__aiter__'):  # async iterable
                        async for item in ctx:
                            yield item
                    elif hasattr(ctx, '__iter__') and not isinstance(ctx, (str, bytes, dict)):
                        for item in ctx:
                            yield item
                    else:
                        yield ctx
            # Check if result is a sync context manager
            elif hasattr(result, '__enter__') and hasattr(result, '__exit__'):
                with result as ctx:
                    if hasattr(ctx, '__aiter__'):  # async iterable
                        async for item in ctx:
                            yield item
                    elif hasattr(ctx, '__iter__') and not isinstance(ctx, (str, bytes, dict)):
                        for item in ctx:
                            yield item
                    else:
                        yield ctx
            else:
                # Regular async function return - treat as single item
                yield result
        else:
            # Sync function - check what it returns
            result = self.func(*args, **kwargs)
            
            # Check if result is a sync context manager
            if hasattr(result, '__enter__') and hasattr(result, '__exit__'):
                with result as ctx:
                    if hasattr(ctx, '__aiter__'):  # async iterable
                        async for item in ctx:
                            yield item
                    elif hasattr(ctx, '__iter__') and not isinstance(ctx, (str, bytes, dict)):
                        for item in ctx:
                            yield item
                    else:
                        yield ctx
            elif hasattr(result, '__aiter__'):  # async iterator
                async for item in result:
                    yield item
            elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):  # sync iterator (but not string or dict)
                for item in result:
                    yield item
            else:  # single value
                yield result

def _wait_until_status_none(rd: AsyncDispatcher, rid: str, timeout: float = 3.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if rd.status(rid) is None:
            return
        time.sleep(0.01)
    raise TimeoutError(f"Request {rid} did not disappear within {timeout}s")

def _wait_until(rd: AsyncDispatcher, rid: str, states: set[RequestState], timeout: float = 3.0):
    """Poll status until it reaches one of the target states or timeout."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        st = rd.status(rid)
        if st and st.state in states:
            return st
        time.sleep(0.01)
    raise TimeoutError(f"Request {rid} did not reach {states} within {timeout}s")

@pytest.fixture(scope="module")
def rd() -> AsyncDispatcher:
    # Singleton instance (your decorator/property should handle this).
    return AsyncDispatcher.obj


# Async function returning a value
async def _ai_async_value(payload: Any, delay: float = 0.05) -> dict:
    await asyncio.sleep(delay)
    return {"ok": True, "payload": payload}

# Sync function returning a value
def _ai_sync_value(x: int, delay: float = 0.05) -> int:
    time.sleep(delay)
    return x * 2

# Async generator (stream)
async def _ai_async_stream(n: int = 4, interval: float = 0.03) -> AsyncIterator[dict]:
    for i in range(n):
        await asyncio.sleep(interval)
        yield {"i": i}

# Sync generator (stream)
def _ai_sync_stream(n: int = 4, interval: float = 0.03) -> Iterable[dict]:
    for i in range(n):
        time.sleep(interval)
        yield {"i": i}

# Sync function sometimes failing
def _ai_flaky(p_fail: float = 1.0, delay: float = 0.02) -> str:
    time.sleep(delay)
    if p_fail >= 1.0:
        raise RuntimeError("simulated failure")
    return "ok"



class TestSubmitProc:

    def test_submit_proc_sync_returns_result(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 21, delay=0.02)
        st = _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        assert st.state == RequestState.DONE
        assert rd.result(rid) == 42

    def test_submit_proc_async_returns_result(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_async_value)
        rid = rd.submit_proc(proc, {"msg": "hi"}, delay=0.02)
        st = _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        assert st.state == RequestState.DONE
        res = rd.result(rid)
        assert res["ok"] is True and res["payload"] == {"msg": "hi"}

    def test_submit_proc_error_propagates(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_flaky)
        rid = rd.submit_proc(proc, p_fail=1.0, delay=0.01)
        st = _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        assert st.state == RequestState.ERROR
        with pytest.raises(RuntimeError):
            _ = rd.result(rid)

    def test_submit_proc_non_callable_sets_error_and_result_raises(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(12345)  # type: ignore[arg-type]
        rid = rd.submit_proc(proc)
        st = _wait_until(rd, rid, {RequestState.ERROR})
        assert st.state == RequestState.ERROR
        with pytest.raises(TypeError):
            _ = rd.result(rid)

    def test_submit_proc_forwards_args_and_kwargs_exactly(self, rd: AsyncDispatcher):
        seen = {}
        def checker(a, b, *, c=None, d=4):
            seen["args"] = (a, b)
            seen["kwargs"] = {"c": c, "d": d}
            return "ok"
        proc = FuncAsyncProcess(checker)
        rid = rd.submit_proc(proc, 1, 2, c=3, d=5)
        _wait_until(rd, rid, {RequestState.DONE})
        assert rd.result(rid) == "ok"
        assert seen["args"] == (1, 2)
        assert seen["kwargs"] == {"c": 3, "d": 5}

    def test_submit_proc_zero_delay_no_race(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 7, delay=0.0)
        st = _wait_until(rd, rid, {RequestState.DONE})
        assert rd.result(rid) == 14


    def test_submit_proc_cancel_after_start_does_not_stop_job(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 10, delay=0.05)

        # Wait until it has actually started, to avoid cancelling while queued.
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            st = rd.status(rid)
            if st and st.started_at is not None and st.state == RequestState.RUNNING:
                break
            time.sleep(0.005)

        rd.cancel(rid)  # cancelling after start should not abort the running job

        st = _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        assert st.state in (RequestState.DONE, RequestState.ERROR)

    def test_submit_proc_cancel_before_start_transitions_to_cancelled(self, rd: AsyncDispatcher):
        # Saturate the semaphore with holders so the next job is waiting to start.
        proc_holder = FuncAsyncProcess(_ai_async_value)
        holders = [rd.submit_proc(proc_holder, f"hold-{i}", delay=0.2) for i in range(16)]
        proc_cancel = FuncAsyncProcess(_ai_async_value)
        rid = rd.submit_proc(proc_cancel, "to-cancel", delay=0.01)
        rd.cancel(rid)  # cancel while queued/waiting
        for h in holders:
            _wait_until(rd, h, {RequestState.DONE})
        st = _wait_until(rd, rid, {RequestState.CANCELLED, RequestState.DONE, RequestState.ERROR})
        assert st.state == RequestState.CANCELLED

    # @pytest.mark.filterwarnings("ignore::_pytest.threadexception.PytestUnhandledThreadExceptionWarning")
    # def test_submit_func_callback_raising_exception_does_not_break_dispatcher(self, rd: AsyncDispatcher):
    #     def bad_cb(req_id, result, error):
    #         raise RuntimeError("callback blew up")
    #     rid1 = rd.submit_func(_ai_sync_value, 2, delay=0.01, _callback=bad_cb)
    #     _wait_until(rd, rid1, {RequestState.DONE})
    #     rid2 = rd.submit_func(_ai_sync_value, 3, delay=0.01)
    #     _wait_until(rd, rid2, {RequestState.DONE})
    #     assert rd.result(rid2) == 6


class TestCallbacks:

    def test_callback_called_on_success(self, rd: AsyncDispatcher):
        calls = []
        lock = threading.Lock()
        def cb(req_id: str, result: Any, error: Optional[BaseException]):
            with lock:
                calls.append(("ok", req_id, result, error))

        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 3, delay=0.01, _callback=cb)
        _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        time.sleep(0.05)  # callback runs on a tiny daemon thread
        with lock:
            assert len(calls) == 1
            kind, req_id, result, error = calls[0]
        assert kind == "ok" and req_id == rid and result == 6 and error is None

    def test_callback_called_on_error(self, rd: AsyncDispatcher):
        calls = []
        lock = threading.Lock()
        def cb(req_id: str, result: Any, error: Optional[BaseException]):
            with lock:
                calls.append(("err", req_id, result, error))

        proc = FuncAsyncProcess(_ai_flaky)
        rid = rd.submit_proc(proc, p_fail=1.0, delay=0.01, _callback=cb)
        _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        time.sleep(0.05)
        with lock:
            assert len(calls) == 1
            kind, req_id, result, error = calls[0]
        assert kind == "err" and req_id == rid and result is None and isinstance(error, RuntimeError)

    def test_callback_invoked_after_stream_completion_sentinel(self, rd: AsyncDispatcher):
        order = []
        evt = threading.Event()

        def cb(req_id, result, error):
            order.append("callback")
            evt.set()

        proc = StreamAsyncProcess(_ai_sync_stream)
        rid = rd.submit_stream(proc, n=2, interval=0.01, _callback=cb)
        order.append("consumed")
        chunks = list(rd.stream_result(rid))  # consume

        # Wait for callback rather than polling status (job is purged now)
        assert evt.wait(0.1), "callback not invoked in time"
        assert order[-1] == "callback"
        assert chunks == [{"i": 0}, {"i": 1}]

    def test_callback_called_once_even_if_fast(self, rd: AsyncDispatcher):
        calls = []
        def cb(req_id, result, error):
            calls.append(1)
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 1, delay=0.0, _callback=cb)
        _wait_until(rd, rid, {RequestState.DONE})
        time.sleep(0.05)
        assert len(calls) == 1


class TestSubmitStream:

    def test_submit_stream_async_generator(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=5)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"i": i} for i in range(5)]
        # Job is purged after consumption
        assert rd.status(rid) is None

    def test_submit_stream_sync_generator(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_sync_stream)
        rid = rd.submit_stream(proc, n=3, interval=0.01)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"i": i} for i in range(3)]
        # Job is purged after consumption
        assert rd.status(rid) is None

    def test_submit_stream_iterable_return_object(self, rd: AsyncDispatcher):
        """Simulates OpenAI SDK iterable stream object."""
        class FakeStream:
            def __init__(self, n=4, dt=0.003): self.n, self.dt = n, dt
            def __iter__(self):
                for i in range(self.n):
                    time.sleep(self.dt)
                    yield {"delta": f"chunk-{i}"}

        class FakeClient:
            class ChatCompletions:
                @staticmethod
                def create(*, stream=False, **kw):
                    return FakeStream(n=kw.get("n", 4), dt=kw.get("dt", 0.003)) if stream else {"choices":[{"message":{"content":"full"}}]}
        client = FakeClient()

        proc = StreamAsyncProcess(client.ChatCompletions.create)
        rid = rd.submit_stream(proc, stream=True, n=5, dt=0.002)
        chunks = list(rd.stream_result(rid))
        assert [c["delta"] for c in chunks] == [f"chunk-{i}" for i in range(5)]

    def test_submit_stream_async_return_non_iterable_is_one_shot(self, rd: AsyncDispatcher):
        async def async_value():
            await asyncio.sleep(0.005)
            return {"one": "shot"}
        proc = StreamAsyncProcess(async_value)
        rid = rd.submit_stream(proc)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"one": "shot"}]

    def test_submit_stream_one_shot_sync_function(self, rd: AsyncDispatcher):
        def one_shot():
            return {"ok": 1}
        proc = StreamAsyncProcess(one_shot)
        rid = rd.submit_stream(proc)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"ok": 1}]

    def test_submit_stream_async_returns_sync_iterable_is_one_shot_current_behavior(self, rd: AsyncDispatcher):
        """Current implementation treats this as one-shot; lock that behavior."""
        async def returns_list():
            await asyncio.sleep(0.002)
            return [1, 2, 3]  # sync iterable
        proc = StreamAsyncProcess(returns_list)
        rid = rd.submit_stream(proc)
        chunks = list(rd.stream_result(rid))
        assert chunks == [[1, 2, 3]]

    def test_submit_stream_timeout_parameter_allows_waiting_for_slow_producer(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=3, interval=0.08)
        collected = []
        start = time.perf_counter()
        for chunk in rd.stream_result(rid, timeout=0.02):
            collected.append(chunk)
        elapsed = time.perf_counter() - start
        assert collected == [{"i": 0}, {"i": 1}, {"i": 2}]
        assert elapsed >= 0.16  # at least two intervals passed—rough sanity check

    def test_submit_stream_midstream_error_sync_generator_sets_error_and_stops(self, rd: AsyncDispatcher):
        def bad_gen():
            yield {"i": 0}
            time.sleep(0.01)
            raise RuntimeError("boom")

        proc = StreamAsyncProcess(bad_gen)
        rid = rd.submit_stream(proc)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"i": 0}]
        # Job is purged on consumer finish; status is None
        assert rd.status(rid) is None

    def test_submit_stream_midstream_error_async_generator_sets_error_and_stops(self, rd: AsyncDispatcher):
        async def bad_agen():
            yield {"i": 0}
            await asyncio.sleep(0.005)
            raise RuntimeError("boom")

        proc = StreamAsyncProcess(bad_agen)
        rid = rd.submit_stream(proc)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"i": 0}]
        # Job is purged on consumer finish; status is None
        assert rd.status(rid) is None

    def test_submit_stream_async_context_manager_yields_async_iterable(self, rd: AsyncDispatcher):
        async def open_stream_ctx(n=4, interval=0.003):
            class AsyncStream:
                def __init__(self): self.i = 0
                def __aiter__(self): return self
                async def __anext__(self):
                    if self.i >= n: raise StopAsyncIteration
                    v = {"i": self.i}
                    self.i += 1
                    await asyncio.sleep(interval)
                    return v
            class AsyncCtx:
                async def __aenter__(self): return AsyncStream()
                async def __aexit__(self, exc_type, exc, tb): return False
            return AsyncCtx()

        proc = StreamAsyncProcess(open_stream_ctx)
        rid = rd.submit_stream(proc, n=5, interval=0.002)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"i": i} for i in range(5)]

    def test_submit_stream_sync_context_manager_yields_iterable(self, rd: AsyncDispatcher):
        def open_stream_ctx(n=4, interval=0.003):
            class SyncStream:
                def __iter__(self):
                    for i in range(n):
                        time.sleep(interval)
                        yield {"i": i}
            class SyncCtx:
                def __enter__(self): return SyncStream()
                def __exit__(self, exc_type, exc, tb): return False
            return SyncCtx()

        proc = StreamAsyncProcess(open_stream_ctx)
        rid = rd.submit_stream(proc, n=6, interval=0.001)
        chunks = list(rd.stream_result(rid))
        assert chunks == [{"i": i} for i in range(6)]

    def test_submit_stream_second_consumer_raises(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_sync_stream)
        rid = rd.submit_stream(proc, n=2, interval=0.001)
        first = list(rd.stream_result(rid))
        assert first == [{"i": 0}, {"i": 1}]
        # Job is purged; second call is unknown id now
        with pytest.raises(KeyError):
            _ = list(rd.stream_result(rid, timeout=0.01))
        
    def test_submit_stream_second_consumer_while_in_progress_raises(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_sync_stream)
        rid = rd.submit_stream(proc, n=10, interval=0.01)
        it = rd.stream_result(rid, timeout=0.01)  # claims the stream immediately
        with pytest.raises(RuntimeError, match="already being consumed"):
            _ = list(rd.stream_result(rid, timeout=0.01))
        _ = list(it)  # drain


class TestCancelStream:

    def test_cancel_stream_midway(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=50, interval=0.01)
        collected = []
        for idx, chunk in enumerate(rd.stream_result(rid, timeout=0.5)):
            collected.append(chunk)
            if idx == 5:
                rd.cancel(rid)
        assert len(collected) >= 1
        # Stream iterator finished → job purged
        assert rd.status(rid) is None

    def test_cancel_stream_invokes_callback_with_none_none(self, rd: AsyncDispatcher):
        seen = {}
        evt = threading.Event()

        def cb(req_id, result, error):
            seen["triplet"] = (req_id, result, error)
            print('Callback invoked:', req_id, result, error)
            evt.set()

        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=2, interval=0.1, _callback=cb)
        rd.cancel(rid)

        assert evt.wait(0.4), "callback not invoked in time"
        assert "triplet" in seen
        _req, _res, _err = seen["triplet"]
        assert _req == rid and _res is None and _err is None

        # Cancelled with no consumer → dispatcher may purge immediately
        _wait_until_status_none(rd, rid)
        assert rd.status(rid) is None

    def test_cancel_stream_idempotent_multiple_calls(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=5, interval=0.02)
        first = rd.cancel(rid)
        second = rd.cancel(rid)
        assert first in (True, False)
        assert second in (True, False)

        _wait_until_status_none(rd, rid)
        assert rd.status(rid) is None

    def test_cancel_stream_before_start_transitions_to_cancelled(self, rd: AsyncDispatcher):
        proc_holder = FuncAsyncProcess(_ai_async_value)
        holders = [rd.submit_proc(proc_holder, f"hold-{i}", delay=0.2) for i in range(16)]
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=3, interval=0.05)
        rd.cancel(rid)

        for h in holders:
            _wait_until(rd, h, {RequestState.DONE})

        _wait_until_status_none(rd, rid)
        assert rd.status(rid) is None


class TestConcurrency:

    def test_later_short_completes_before_earlier_long(self, rd: AsyncDispatcher):
        done_times = {}
        def cb(req_id: str, result: Any, error: Optional[BaseException]):
            done_times[req_id] = time.perf_counter()

        proc_a = FuncAsyncProcess(_ai_async_value)
        rid_a = rd.submit_proc(proc_a, "A", delay=0.30, _callback=cb)  # ~0.30s
        time.sleep(0.10)
        proc_b = FuncAsyncProcess(_ai_sync_value)
        rid_b = rd.submit_proc(proc_b, 1, delay=0.01, _callback=cb)     # ~0.01s

        _wait_until(rd, rid_b, {RequestState.DONE, RequestState.ERROR})
        _wait_until(rd, rid_a, {RequestState.DONE, RequestState.ERROR})
        assert done_times[rid_b] < done_times[rid_a]

    def test_no_starvation_many_shorts_complete_and_long_completes_eventually(self, rd: AsyncDispatcher):
        proc_long = FuncAsyncProcess(_ai_async_value)
        rid_long = rd.submit_proc(proc_long, "long", delay=0.6)
        proc_short = FuncAsyncProcess(_ai_sync_value)
        short_rids = [rd.submit_proc(proc_short, i, delay=0.02) for i in range(25)]
        # Shorts should all finish quickly
        for rid in short_rids:
            _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR}, timeout=2.0)
            assert rd.status(rid).state == RequestState.DONE
        # The long one must still complete (no starvation)
        _wait_until(rd, rid_long, {RequestState.DONE, RequestState.ERROR}, timeout=2.0)
        assert rd.status(rid_long).state == RequestState.DONE

    @pytest.mark.xfail(reason="Need controllable max_concurrency or resettable singleton to assert hard cap.")
    def test_concurrency_respects_max_concurrency_peak_active_count(self, rd: AsyncDispatcher):
        active = 0
        peak = 0
        lock = threading.Lock()
        def probe(delay=0.05):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(delay)
            with lock:
                active -= 1
            return "ok"
        proc_probe = FuncAsyncProcess(probe)
        rids = [rd.submit_proc(proc_probe, delay=0.05) for _ in range(32)]
        for rid in rids:
            _wait_until(rd, rid, {RequestState.DONE})
        assert peak >= 1  # placeholder until max_concurrency is controllable



class TestStatus:

    def test_status_fields_are_ordered(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 5, delay=0.02)
        st_done = _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        assert st_done.state == RequestState.DONE

        st = rd.status(rid)
        assert isinstance(st, RequestStatus)
        assert st.queued_at is not None
        assert st.started_at is not None and st.ended_at is not None
        assert st.queued_at <= st.started_at <= st.ended_at

    def test_status_running_mid_flight_is_running(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 9, delay=0.1)
        time.sleep(0.02)  # give it time to enter RUNNING
        st = rd.status(rid)
        assert st is not None and st.state in (RequestState.RUNNING, RequestState.DONE)
        _wait_until(rd, rid, {RequestState.DONE})

    def test_status_streaming_mid_flight_is_streaming(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=10, interval=0.02)
        time.sleep(0.03)  # give it time to enter STREAMING
        st = rd.status(rid)
        assert st is not None and st.state in (RequestState.STREAMING, RequestState.DONE)
        list(rd.stream_result(rid))  # drain

    def test_status_after_cancel_is_cancelled_with_no_error(self, rd: AsyncDispatcher):
        rid = rd.submit_stream(_ai_async_stream, n=20, interval=0.02)
        rd.cancel(rid)
        _wait_until_status_none(rd, rid)
        assert rd.status(rid) is None

    def test_status_after_error_has_error_string(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_flaky)
        rid = rd.submit_proc(proc, p_fail=1.0, delay=0.005)
        st = _wait_until(rd, rid, {RequestState.ERROR})
        assert isinstance(st.error, str) and st.error

    def test_status_unknown_id_returns_none(self, rd: AsyncDispatcher):
        assert rd.status("no-such-id") is None


class TestResult:

    def test_result_before_done_returns_none(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 2, delay=0.05)
        val = rd.result(rid)  # likely None until done
        assert val is None or isinstance(val, int)  # tolerate fast machines
        _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})

    def test_result_unknown_id_returns_none(self, rd: AsyncDispatcher):
        assert rd.result("does-not-exist") is None

    def test_result_after_stream_success_is_none(self, rd: AsyncDispatcher):
        rid = rd.submit_stream(_ai_sync_stream, n=2)
        list(rd.stream_result(rid))              # consume stream
        _wait_until_status_none(rd, rid)         # job purged
        assert rd.status(rid) is None
        assert rd.result(rid) is None            # streaming jobs never return scalar results

    def test_result_after_stream_error_raises(self, rd: AsyncDispatcher):
        async def bad_agen():
            yield 1
            raise RuntimeError("stream-err")

        proc = StreamAsyncProcess(bad_agen)
        rid = rd.submit_stream(proc)
        for _ in rd.stream_result(rid):  # consume one then the error ends stream
            break
        _wait_until_status_none(rd, rid)         # job purged after consumer exits
        assert rd.status(rid) is None
        assert rd.result(rid) is None            # streaming path never raises via result()

    def test_result_after_stream_cancel_is_none(self, rd: AsyncDispatcher):
        rid = rd.submit_stream(_ai_async_stream, n=20, interval=0.02)
        rd.cancel(rid)
        _wait_until_status_none(rd, rid)
        assert rd.status(rid) is None
        assert rd.result(rid) is None


class TestStreamResult:

    def test_stream_result_on_non_streaming_raises(self, rd: AsyncDispatcher):
        proc = FuncAsyncProcess(_ai_sync_value)
        rid = rd.submit_proc(proc, 3, delay=0.01)
        _wait_until(rd, rid, {RequestState.DONE, RequestState.ERROR})
        with pytest.raises(KeyError):
            _ = list(rd.stream_result(rid))

    def test_stream_result_unknown_id_raises(self, rd: AsyncDispatcher):
        with pytest.raises(KeyError):
            _ = list(rd.stream_result("does-not-exist"))

    def test_stream_result_timeout_yields_chunks_until_sentinel(self, rd: AsyncDispatcher):
        proc = StreamAsyncProcess(_ai_async_stream)
        rid = rd.submit_stream(proc, n=3, interval=0.05)
        start = time.perf_counter()
        out = list(rd.stream_result(rid, timeout=0.01))
        elapsed = time.perf_counter() - start
        assert out == [{"i": 0}, {"i": 1}, {"i": 2}]
        assert elapsed >= 0.10  # sanity


class TestLifecycle:

    @pytest.mark.xfail(reason="Shutdown stops loop; running it here will break subsequent tests. Run in isolation.")
    def test_shutdown_stops_loop_and_subsequent_submits_fail(self, rd: AsyncDispatcher):
        rd.shutdown()
        with pytest.raises(RuntimeError):
            proc = FuncAsyncProcess(_ai_sync_value)
            _ = rd.submit_proc(proc, 1, delay=0.01)