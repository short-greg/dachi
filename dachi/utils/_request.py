import asyncio
import threading
import time
import uuid
import inspect
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional, Iterator
from concurrent.futures import Future
from queue import Queue, Empty
from ._utils import singleton
import threading
import traceback
from queue import Queue, Empty
from typing import Iterator, Optional

import inspect
from typing import Callable, Any, Optional
# You already have this:
# from your_project import singleton


class RequestState(Enum):
    QUEUED = auto()
    RUNNING = auto()
    STREAMING = auto()
    DONE = auto()
    ERROR = auto()
    CANCELLED = auto()

@dataclass
class RequestStatus:
    req_id: str
    state: RequestState
    queued_at: float
    started_at: Optional[float]
    ended_at: Optional[float]
    error: Optional[str]

@dataclass
class _Job:
    req_id: str
    future: Future
    state: RequestState
    queued_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    result: Any = None
    error: Optional[BaseException] = None
    stream_q: Optional[Queue] = None
    callback: Optional[Callable[[str, Any, Optional[BaseException]], None]] = None
    cancelled: bool = False

    # consumer/consumption state
    stream_reader_created: bool = False
    stream_consumed: bool = False
    result_consumed: bool = False

    # callback handoff (no threads)
    deferred_cb: Optional[tuple[Any, Optional[BaseException]]] = None
    callback_fired: bool = False


_SENTINEL = object()

MAX_CONCURRENCY = 8 

@singleton
class RequestDispatcher:
    """
    Sync-friendly dispatcher with true concurrency (bounded by a semaphore).
    Submit:
      - submit_func(f, *args, _callback=None, **kwargs) -> req_id
      - submit_stream(f, *args, _callback=None, **kwargs) -> req_id
      - submit_rest(method, url, *, as_json=False, **requests_kwargs) -> req_id
    Consume:
      - status(req_id) -> RequestStatus | None
      - result(req_id) -> Any | None (raises stored error on ERROR)
      - stream_result(req_id, timeout=None) -> iterator over chunks
    """

    def __init__(self, max_concurrency: int = 8) -> None:
        self._loop = asyncio.new_event_loop()
        self._sema = asyncio.Semaphore(max_concurrency)
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def submit_func(
        self,
        f: Callable[..., Any],
        /,
        *args,
        _callback: Optional[Callable[[str, Any, Optional[BaseException]], None]] = None,
        **kwargs,
    ) -> str:
        """Schedule a function (sync or async). Returns request id."""
        req_id, job = self._create_job(_callback, stream=False)

        async def _job():
            async with self._sema:
                if job.cancelled:  # in case cancelled before starting
                    return self._complete_cancel(job)
                self._mark_running(job, streaming=False)
                try:
                    res = await self._run_callable(f, *args, **kwargs)
                    self._complete_ok(job, res)
                except Exception as exc:
                    self._complete_err(job, exc)

        self._schedule(_job())
        return req_id

    def submit_stream(
        self,
        f: Callable[..., Any],
        /,
        *args,
        _callback: Optional[Callable[[str, Any, Optional[BaseException]], None]] = None,
        **kwargs,
    ) -> str:
        """
        Submit a streaming job.

        Producer shapes supported:
        - async generator function
        - async function returning: async context manager (yielding async-iter / sync-iter / scalar),
                                    async iterable, sync iterable (ONE-SHOT per current behavior), or scalar
        - sync generator function
        - sync function returning: sync context manager (yielding iterable / scalar),
                                    sync iterable, or scalar

        Exit paths (exactly one):
        - success: sentinel → _complete_ok(job, None)
        - error:   sentinel → _complete_err(job, exc)
        - cancel:  sentinel → _complete_cancel(job)
        """
        req_id, job = self._create_job(_callback, stream=True)

        # ---- small shape predicates ----
        def _is_async_gen_fn(fn): return inspect.isasyncgenfunction(fn)
        def _is_async_fn(fn):     return inspect.iscoroutinefunction(fn)
        def _is_gen_fn(fn):       return inspect.isgeneratorfunction(fn)

        def _is_async_cm(x):   return hasattr(x, "__aenter__") and hasattr(x, "__aexit__")
        def _is_async_iter(x): return hasattr(x, "__aiter__")
        def _is_sync_cm(x):    return hasattr(x, "__enter__") and hasattr(x, "__exit__")
        def _is_sync_iter(x):  return hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict))

        # ---- emission & cancel checks ----
        class _CancelledSignal(Exception):
            pass

        def _emit_one(x) -> None:
            if job.cancelled:
                job.stream_q.put(_SENTINEL)
                raise _CancelledSignal()
            job.stream_q.put(x)

        async def _consume_async_iter(ait) -> None:
            async for chunk in ait:
                if job.cancelled:
                    job.stream_q.put(_SENTINEL)
                    raise _CancelledSignal()
                job.stream_q.put(chunk)

        def _consume_sync_iter(it) -> None:
            for chunk in it:
                if job.cancelled:
                    job.stream_q.put(_SENTINEL)
                    raise _CancelledSignal()
                job.stream_q.put(chunk)

        # ---- unified finish helpers ----
        def _finish_ok():
            job.stream_q.put(_SENTINEL)
            return self._complete_ok(job, None)

        def _finish_err(exc: BaseException):
            job.stream_q.put(_SENTINEL)
            return self._complete_err(job, exc)

        def _finish_cancel():
            job.stream_q.put(_SENTINEL)
            return self._complete_cancel(job)

        async def _job():
            async with self._sema:
                # cancelled while queued
                if job.cancelled:
                    return _finish_cancel()

                self._mark_running(job, streaming=True)

                try:
                    # (A) async generator function
                    if _is_async_gen_fn(f):
                        agen = f(*args, **kwargs)
                        await _consume_async_iter(agen)
                        return _finish_ok()

                    # (B) async function → normalize result
                    if _is_async_fn(f):
                        res = await f(*args, **kwargs)

                        # async context manager
                        if _is_async_cm(res):
                            async with res as aobj:
                                if _is_async_iter(aobj):
                                    await _consume_async_iter(aobj)
                                elif _is_sync_iter(aobj):
                                    # iterate sync iterable in a thread
                                    await self._loop.run_in_executor(None, _consume_sync_iter, aobj)
                                else:
                                    _emit_one(aobj)
                            return _finish_ok()

                        # async iterable
                        if _is_async_iter(res):
                            await _consume_async_iter(res)
                            return _finish_ok()

                        if _is_sync_iter(res):
                            _emit_one(res)   # emit the whole iterable as a single chunk
                            return _finish_ok()

                        # scalar one-shot
                        _emit_one(res)
                        return _finish_ok()

                    # (C) sync generator function
                    if _is_gen_fn(f):
                        def _work_gen():
                            _consume_sync_iter(f(*args, **kwargs))
                        await self._loop.run_in_executor(None, _work_gen)
                        return _finish_ok()

                    # (D) sync function → normalize value
                    def _call_sync():
                        return f(*args, **kwargs)
                    val = await self._loop.run_in_executor(None, _call_sync)

                    # sync context manager
                    if _is_sync_cm(val):
                        def _work_cm():
                            with val as obj:
                                if _is_sync_iter(obj):
                                    _consume_sync_iter(obj)
                                else:
                                    _emit_one(obj)
                        await self._loop.run_in_executor(None, _work_cm)
                        return _finish_ok()

                    # sync iterable
                    if _is_sync_iter(val):
                        await self._loop.run_in_executor(None, _consume_sync_iter, val)
                        return _finish_ok()

                    # scalar one-shot
                    _emit_one(val)
                    return _finish_ok()

                except _CancelledSignal:
                    return _finish_cancel()
                except Exception as exc:
                    return _finish_err(exc)

        self._schedule(_job())
        return req_id

    def submit_rest(
        self,
        method: str,
        url: str,
        /,
        *,
        as_json: bool = False,
        **requests_kwargs,
    ) -> str:
        """
        Run a blocking HTTP request via `requests` without blocking others.
        """
        import requests
        req_id, job = self._create_job(stream=False)

        async def _job():
            async with self._sema:
                if job.cancelled:
                    return self._complete_cancel(job)
                self._mark_running(job, streaming=False)
                try:
                    def _call():
                        r = requests.request(method, url, **requests_kwargs)
                        if as_json:
                            return {"status_code": r.status_code, "json": r.json()}
                        return {
                            "status_code": r.status_code,
                            "headers": dict(r.headers),
                            "text": r.text,
                        }
                    res = await self._loop.run_in_executor(None, _call)
                    self._complete_ok(job, res)
                except Exception as exc:
                    self._complete_err(job, exc)

        self._schedule(_job())
        return req_id

    def status(self, req_id: str) -> RequestStatus | None:
        with self._lock:
            job = self._jobs.get(req_id)
            if not job:
                return None
            return RequestStatus(
                req_id=req_id,
                state=job.state,
                queued_at=job.queued_at,
                started_at=job.started_at,
                ended_at=job.ended_at,
                error=str(job.error) if job.error else None,
            )

    def result(self, req_id: str) -> Any | None:
        with self._lock:
            job = self._jobs.get(req_id)
            if not job:
                return None
            if job.stream_q is not None:
                return None  # streaming jobs have no scalar result

            if job.state == RequestState.DONE:
                if job.result_consumed:
                    return None
                out = job.result
                job.result = None
                job.result_consumed = True
                self._jobs.pop(req_id, None)   # purge now
                return out

            if job.state == RequestState.ERROR:
                if job.result_consumed:
                    return None
                job.result_consumed = True
                err = job.error
                self._jobs.pop(req_id, None)   # purge now
                raise err  # type: ignore[misc]

            return None


    def stream_result(self, req_id: str, timeout: float | None = None):
        with self._lock:
            job = self._jobs.get(req_id)
            if not job or job.stream_q is None:
                raise KeyError(f"Unknown or non-streaming req_id={req_id}")

            if job.stream_reader_created:
                if job.stream_consumed:
                    raise RuntimeError(f"Stream already consumed for req_id={req_id}")
                raise RuntimeError(f"Stream is already being consumed for req_id={req_id}")

            job.stream_reader_created = True
            q = job.stream_q
            job_ref = job
            rid = job.req_id

        def _iter():
            from queue import Empty
            try:
                while True:
                    try:
                        item = q.get(timeout=timeout)
                    except Empty:
                        with self._lock:
                            finished = job_ref.state in (
                                RequestState.DONE,
                                RequestState.ERROR,
                                RequestState.CANCELLED,
                            )
                            if finished and q.empty():
                                job_ref.stream_consumed = True
                                return
                        continue

                    if item is _SENTINEL:
                        with self._lock:
                            job_ref.stream_consumed = True
                        return

                    yield item
            finally:
                to_fire = None
                with self._lock:
                    if job_ref.stream_consumed and job_ref.callback and job_ref.deferred_cb and not job_ref.callback_fired:
                        res, err = job_ref.deferred_cb
                        job_ref.callback_fired = True
                        to_fire = (job_ref.callback, rid, res, err)
                    # purge (idempotent if already popped)
                    self._jobs.pop(rid, None)

                if to_fire is not None:
                    cb, rid2, res2, err2 = to_fire
                    import threading, traceback
                    def _run():
                        try:
                            cb(rid2, res2, err2)
                        except Exception:
                            try: traceback.print_exc()
                            except Exception: pass
                    threading.Thread(target=_run, daemon=True).start()

        return _iter()


    def _invoke_callback(self, req_id: str, result, error: BaseException | None) -> None:
        cb = None
        with self._lock:
            job = self._jobs.get(req_id)
            if job:
                cb = job.callback  # your field name may differ; see §2 below
        if not cb:
            return

        def _runner():
            try:
                cb(req_id, result, error)
            except Exception:
                # Never propagate; keep dispatcher resilient
                try:
                    traceback.print_exc()  # or log if you have a logger
                except Exception:
                    pass
        
        t = threading.Thread(target=_runner, daemon=True)
        t.start()

    def _purge_finished(self, ttl: float | None = None) -> int:
        """Remove DONE/ERROR/CANCELLED jobs."""
        now = time.time()
        removed = 0
        with self._lock:
            for rid, j in list(self._jobs.items()):
                if j.ended_at is None:
                    continue
                if ttl is None or (now - j.ended_at) >= ttl:
                    del self._jobs[rid]
                    removed += 1
        return removed

    def cancel(self, req_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(req_id)
            if not job:
                return False
            if job.state in (RequestState.DONE, RequestState.ERROR, RequestState.CANCELLED):
                return False

            job.cancelled = True
            is_stream = job.stream_q is not None
            cb, rid = job.callback, job.req_id

            # queued and no consumer → complete immediately
            if is_stream and job.started_at is None and not job.stream_reader_created:
                job.ended_at = time.time()
                job.state = RequestState.CANCELLED
                job.stream_q.put(_SENTINEL)
                to_fire = (cb, rid) if cb else None
                self._jobs.pop(rid, None)
            else:
                # already started or consumer exists
                if is_stream:
                    job.stream_q.put(_SENTINEL)
                # if a consumer exists and hasn't finished, let it fire via deferred_cb
                to_fire = None
                if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                    job.deferred_cb = (None, None)

        if to_fire is not None:
            cb2, rid2 = to_fire
            def _run():
                try: cb2(rid2, None, None)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        return True

    def shutdown(self):
        """Stop the loop (useful in tests)."""
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _schedule(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _create_job(
        self,
        callback: Optional[Callable[[str, Any, Optional[BaseException]], None]] = None,
        stream: bool = False,
    ) -> tuple[str, _Job]:
        req_id = str(uuid.uuid4())
        job = _Job(
            req_id=req_id,
            future=Future(),
            state=RequestState.QUEUED,
            queued_at=time.time(),
            stream_q=Queue() if stream else None,
            callback=callback,
        )
        with self._lock:
            self._jobs[req_id] = job
        return req_id, job

    def _mark_running(self, job: _Job, *, streaming: bool):
        job.started_at = time.time()
        job.state = RequestState.STREAMING if streaming else RequestState.RUNNING

    def _complete_ok(self, job, result):
        # sentinel should have been enqueued by submit_stream before calling here
        with self._lock:
            job.ended_at = time.time()
            job.state = RequestState.DONE
            job.result = result
            is_stream = job.stream_q is not None
            cb, rid = job.callback, job.req_id

            if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                job.deferred_cb = (result, None)
                return  # consumer will fire it

        # non-stream → fire now (result() will purge)
        if not is_stream:
            if cb:
                import threading, traceback
                def _run():
                    try: cb(rid, result, None)
                    except Exception:
                        try: traceback.print_exc()
                        except Exception: pass
                threading.Thread(target=_run, daemon=True).start()
            return

        # streaming + no consumer → fire now and purge
        if cb:
            import threading, traceback
            def _run():
                try: cb(rid, result, None)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        with self._lock:
            self._jobs.pop(rid, None)


    def _complete_err(self, job, exc: BaseException):
        with self._lock:
            job.ended_at = time.time()
            job.state = RequestState.ERROR
            job.error = exc
            is_stream = job.stream_q is not None
            cb, rid = job.callback, job.req_id

            if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                job.deferred_cb = (None, exc)
                return

        if not is_stream:
            if cb:
                import threading, traceback
                def _run():
                    try: cb(rid, None, exc)
                    except Exception:
                        try: traceback.print_exc()
                        except Exception: pass
                threading.Thread(target=_run, daemon=True).start()
            return

        if cb:
            import threading, traceback
            def _run():
                try: cb(rid, None, exc)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        with self._lock:
            self._jobs.pop(rid, None)


    def _complete_cancel(self, job):
        with self._lock:
            job.ended_at = time.time()
            job.state = RequestState.CANCELLED
            is_stream = job.stream_q is not None
            cb, rid = job.callback, job.req_id

            if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                job.deferred_cb = (None, None)
                return

        if not is_stream:
            if cb:
                import threading, traceback
                def _run():
                    try: cb(rid, None, None)
                    except Exception:
                        try: traceback.print_exc()
                        except Exception: pass
                threading.Thread(target=_run, daemon=True).start()
            return

        if cb:
            import threading, traceback
            def _run():
                try: cb(rid, None, None)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        with self._lock:
            self._jobs.pop(rid, None)

    async def _run_callable(self, f: Callable[..., Any], *args, **kwargs) -> Any:
        """Await async callables; run sync ones in a default thread pool."""
        if inspect.iscoroutinefunction(f):
            return await f(*args, **kwargs)
        return await self._loop.run_in_executor(None, lambda: f(*args, **kwargs))
