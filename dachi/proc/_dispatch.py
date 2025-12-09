"""Async request dispatcher for concurrent AI processing with centralized job management.

Provides AsyncDispatcher for managing concurrent requests to AI providers with features like:
- Concurrency limiting to prevent API rate limit issues
- Request state tracking for integration with behavior trees and state machines  
- Streaming support with proper resource management
- Callback-based completion notification
- Thread-safe job management across sync/async boundaries

Typical usage in behavior trees:
    dispatcher = AsyncDispatcher(max_concurrency=5)
    
    # Task dispatches request and returns immediately
    req_id = dispatcher.submit_proc(llm_adapter, message)
    
    # Later, task checks status without blocking
    status = dispatcher.status(req_id)
    if status.state == RequestState.DONE:
        result = dispatcher.result(req_id)
        return TaskStatus.SUCCESS
    elif status.state == RequestState.ERROR:
        return TaskStatus.FAILURE  
    else:
        return TaskStatus.RUNNING  # Still processing
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
import traceback
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue, Empty
from typing import Any, Iterator, Optional, Callable
import threading, traceback

from ..utils._utils import singleton
from ._process import AsyncProcess, AsyncStreamProcess


class RequestState(Enum):
    """State transitions for async requests in the dispatcher.
    
    State flow:
    QUEUED -> RUNNING/STREAMING -> DONE/ERROR/CANCELLED
    
    - QUEUED: Request submitted but not yet started (waiting for concurrency slot)
    - RUNNING: Non-streaming request currently executing
    - STREAMING: Streaming request currently producing data
    - DONE: Request completed successfully
    - ERROR: Request failed with exception
    - CANCELLED: Request was cancelled before or during execution
    """
    QUEUED = auto()
    RUNNING = auto()
    STREAMING = auto()
    DONE = auto()
    ERROR = auto()
    CANCELLED = auto()

@dataclass
class DispatchStatus:
    """Public status information for tracking async request progress.
    
    Used by behavior trees and state machines to monitor request completion
    without blocking execution. Provides timing information for performance analysis.
    
    Attributes:
        disp_id: Unique request identifier
        state: Current request state (QUEUED, RUNNING, etc.)
        queued_at: Timestamp when request was submitted
        started_at: Timestamp when execution began (None if not started)
        ended_at: Timestamp when request completed (None if still running)
        error: Error message if request failed (None if successful)
    """
    disp_id: str
    state: RequestState
    queued_at: float
    started_at: Optional[float]
    ended_at: Optional[float]
    error: Optional[str]

@dataclass
class _Job:
    """Internal job tracking for async request lifecycle management.
    
    Manages the complete lifecycle of an async request including execution state,
    result storage, streaming coordination, and callback management. Handles
    complex scenarios like streaming with multiple consumers and thread-safe
    state transitions.
    
    Key responsibilities:
    - Track request execution state and timing
    - Coordinate streaming data flow between producers and consumers
    - Manage callback execution to prevent race conditions
    - Handle cancellation across different execution phases
    """
    disp_id: str
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

MAX_CONCURRENCY = 8  # Default maximum concurrent requests to prevent API overload 

@singleton
class AsyncDispatcher:
    """
    Centralized async request dispatcher for AI processing with concurrency control.
    
    Core component for managing concurrent AI requests across the system. Enables
    behavior trees and state machines to dispatch long-running AI operations without
    blocking, then poll for completion or consume results asynchronously.
    
    Key Features:
    - **Concurrency Limiting**: Prevents overwhelming APIs with too many parallel requests
    - **Non-blocking Integration**: Tasks can dispatch requests and check status later
    - **Streaming Support**: Handles streaming AI responses with proper resource management
    - **Thread Safety**: Safe coordination between sync behavior trees and async AI calls
    - **Callback Support**: Optional completion notifications for event-driven patterns
    
    Submission Methods:
        submit_proc(proc, *args, _callback=None, **kwargs) -> disp_id
            Submit non-streaming async process for execution
            
        submit_stream(proc, *args, _callback=None, **kwargs) -> disp_id  
            Submit streaming async process for execution
    
    Consumption Methods:
        status(disp_id) -> DispatchStatus | None
            Get current request status without blocking
            
        result(disp_id) -> Any | None  
            Get result for completed non-streaming request (raises on error)
            
        stream_result(req_id, timeout=None) -> Iterator
            Get iterator disp_id consuming streaming request output
    
    Example Integration with Behavior Trees:
        class LLMQueryTask(Task):
            def __init__(self, dispatcher, llm, query):
                self.dispatcher = dispatcher
                self.llm = llm
                self.query = query
                self.disp_id = None

            def tick(self) -> TaskStatus:
                if self.disp_id is None:
                    # First tick: dispatch request
                    self.disp_id = self.dispatcher.submit_proc(
                        self.llm, self.query
                    )
                    return TaskStatus.RUNNING
                    
                # Subsequent ticks: check status
                status = self.dispatcher.status(self.disp_id)
                if status.state == RequestState.DONE:
                    self.result = self.dispatcher.result(self.disp_id)
                    return TaskStatus.SUCCESS
                elif status.state == RequestState.ERROR:
                    return TaskStatus.FAILURE
                else:
                    return TaskStatus.RUNNING  # Still processing
    """

    def __init__(self, max_concurrency: int = 8) -> None:
        self._loop = asyncio.new_event_loop()
        self._sema = asyncio.Semaphore(max_concurrency)
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def submit_proc(
        self,
        proc: AsyncProcess,
        /,
        *args,
        _callback: Optional[Callable[[str, Any, Optional[BaseException]], None]] = None,
        **kwargs,
    ) -> str:
        """Submit non-streaming async process for concurrent execution.
        
        Dispatches an AI processing task (like LLM inference) to run concurrently
        with other operations. The calling task can continue execution and poll
        for completion later using the returned request ID.
        
        Args:
            proc: AsyncProcess instance (like OpenAIChat) to execute
            *args: Positional arguments to pass to proc.aforward()
            _callback: Optional completion callback (disp_id, result, error) -> None
            **kwargs: Keyword arguments to pass to proc.aforward()
            
        Returns:
            Unique dispatch ID for tracking this submission

        Example:
            dispatcher = AsyncDispatcher()
            llm = OpenAIChat()
            
            # Submit request
            disp_id = dispatcher.submit_proc(
                llm, 
                Msg(role="user", text="Hello"),
                temperature=0.7
            )
            
            # Continue other work...
            
            # Later check result
            if dispatcher.status(disp_id).state == RequestState.DONE:
                response = dispatcher.result(disp_id)
        """
        disp_id, job = self._create_job(_callback, stream=False)

        async def _job():
            async with self._sema:
                if job.cancelled:  # in case cancelled before starting
                    return self._complete_cancel(job)
                self._mark_running(job, streaming=False)
                try:
                    res = await proc.aforward(*args, **kwargs)
                    self._complete_ok(job, res)
                except Exception as exc:
                    self._complete_err(job, exc)

        self._schedule(_job())
        return disp_id

    def submit_stream(
        self,
        proc: AsyncStreamProcess,
        /,
        *args,
        _callback: Optional[Callable[[str, Any, Optional[BaseException]], None]] = None,
        _sleep: float | None = None,
        **kwargs,
    ) -> str:
        """
        Submit streaming async process for concurrent execution.
        
        Dispatches a streaming AI task (like streaming LLM inference) to run 
        concurrently. The calling task can continue and later consume the stream
        using stream_result(). Handles proper cleanup and resource management.
        
        Args:
            proc: AsyncStreamProcess instance (like streaming OpenAIChat) to execute
            *args: Positional arguments to pass to proc.astream()
            _callback: Optional completion callback (disp_id, result, error) -> None
            **kwargs: Keyword arguments to pass to proc.astream()
            
        Returns:
            Unique dispatch ID for consuming the stream

        Example:
            dispatcher = AsyncDispatcher()
            llm = OpenAIChat()  # Streaming-capable
            
            # Submit streaming request
            disp_id = dispatcher.submit_stream(
                llm,
                Msg(role="user", text="Write a story"),
                temperature=0.8
            )
            
            # Consume stream when ready
            for chunk in dispatcher.stream_result(disp_id):
                print(chunk.delta.text, end="")
                
        Note:
            Streaming exit paths (exactly one occurs):
            - Success: All chunks consumed → callback with (disp_id, None, None)
            - Error: Exception during streaming → callback with (disp_id, None, exception)
            - Cancel: Request cancelled → callback with (disp_id, None, None)
        """
        disp_id, job = self._create_job(_callback, stream=True)

        # ---- emission & cancel checks ----
        class _CancelledSignal(Exception):
            pass

        async def _consume_async_iter(ait) -> None:
            async for chunk in ait:
                if job.cancelled:
                    job.stream_q.put(_SENTINEL)
                    raise _CancelledSignal()
                job.stream_q.put(chunk)
                if _sleep:
                    await asyncio.sleep(_sleep)

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
                    # Use AsyncStreamProcess.astream method
                    async_iter = proc.astream(*args, **kwargs)
                    await _consume_async_iter(async_iter)
                    return _finish_ok()

                except _CancelledSignal:
                    return _finish_cancel()
                except Exception as exc:
                    return _finish_err(exc)

        self._schedule(_job())
        return disp_id

    def status(self, disp_id: str) -> DispatchStatus | None:
        """Get current status of submitted request without blocking.
        
        Essential for behavior trees and state machines to monitor request
        progress without blocking task execution. Returns None if request
        ID is not found (may have been cleaned up).
        
        Args:
            disp_id: Request ID returned from submit_proc() or submit_stream()
            
        Returns:
            RequestStatus with current state and timing info, or None if not found
            
        Example:
            status = dispatcher.status(disp_id)
            if status is None:
                # Request not found (cleaned up or invalid ID)
                return TaskStatus.FAILURE
            elif status.state == RequestState.DONE:
                # Request completed successfully
                result = dispatcher.result(disp_id)
                return TaskStatus.SUCCESS
            elif status.state == RequestState.ERROR:
                # Request failed
                return TaskStatus.FAILURE
            else:
                # Request still running (QUEUED/RUNNING/STREAMING)
                return TaskStatus.RUNNING
        """
        with self._lock:
            job = self._jobs.get(disp_id)
            if not job:
                return None
            return DispatchStatus(
                disp_id=disp_id,
                state=job.state,
                queued_at=job.queued_at,
                started_at=job.started_at,
                ended_at=job.ended_at,
                error=str(job.error) if job.error else None,
            )

    def result(self, disp_id: str) -> Any | None:
        """Get result from completed non-streaming request.
        
        Retrieves the final result from a request submitted via submit_proc().
        Result can only be consumed once - subsequent calls return None.
        Job is automatically cleaned up after result consumption.
        
        Args:
            disp_id: Dispatch ID from submit_proc()

        Returns:
            Request result if completed successfully, None otherwise
            
        Raises:
            Exception: The original exception if request failed
            
        Note:
            - Returns None for streaming requests (use stream_result() instead)
            - Returns None for non-completed requests (check status() first)
            - Returns None if result was already consumed
            - Job is cleaned up after successful result retrieval
            
        Example:
            # Check completion before retrieving result
            status = dispatcher.status(disp_id)
            if status and status.state == RequestState.DONE:
                try:
                    result = dispatcher.result(disp_id)
                    process_result(result)
                except Exception as e:
                    # Request failed
                    handle_error(e)
        """
        with self._lock:
            job = self._jobs.get(disp_id)
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
                self._jobs.pop(disp_id, None)   # purge now
                return out

            if job.state == RequestState.ERROR:
                if job.result_consumed:
                    return None
                job.result_consumed = True
                err = job.error
                self._jobs.pop(disp_id, None)   # purge now
                raise err  # type: ignore[misc]

            return None

    def stream_result(self, disp_id: str, timeout: float | None = None):
        """Get iterator for consuming streaming request results.
        
        Creates iterator for consuming chunks from a streaming request submitted
        via submit_stream(). Iterator can only be created once per request.
        Automatically handles cleanup and callback execution when stream is consumed.
        
        Args:
            disp_id: Dispatch ID from submit_stream()
            timeout: Timeout in seconds for waiting for chunks (None = no timeout)
            
        Returns:
            Iterator yielding chunks from the streaming request
            
        Raises:
            KeyError: If disp_id is unknown or not a streaming request
            RuntimeError: If stream iterator already created or consumed
            
        Example:
            # Submit streaming request
            disp_id = dispatcher.submit_stream(streaming_llm, message)
            
            # Consume stream (can only do this once)
            accumulated_text = ""
            for chunk in dispatcher.stream_result(disp_id):
                accumulated_text += chunk.delta.text or ""
                print(chunk.delta.text, end="", flush=True)
            
            # Stream is now consumed and cleaned up
            
        Note:
            - Stream can only be consumed once per request
            - Job is automatically cleaned up when stream is fully consumed
            - Callbacks are executed after stream consumption completes
            - Timeout only applies to individual chunk retrieval, not total time
        """
        with self._lock:
            job = self._jobs.get(disp_id)
            if not job or job.stream_q is None:
                raise KeyError(f"Unknown or non-streaming disp_id={disp_id}")

            if job.stream_reader_created:
                if job.stream_consumed:
                    raise RuntimeError(f"Stream already consumed for disp_id={disp_id}")
                raise RuntimeError(f"Stream is already being consumed for disp_id={disp_id}")

            job.stream_reader_created = True
            q = job.stream_q
            job_ref = job
            rid = job.disp_id

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


    def _invoke_callback(self,  disp_id: str, result, error: BaseException | None) -> None:
        cb = None
        with self._lock:
            job = self._jobs.get(disp_id)
            if job:
                cb = job.callback  # your field name may differ; see §2 below
        if not cb:
            return

        def _runner():
            try:
                cb(disp_id, result, error)
            except Exception:
                # Never propagate; keep dispatcher resilient
                try:
                    traceback.print_exc()  # or log if you have a logger
                except Exception:
                    pass
        
        t = threading.Thread(target=_runner, daemon=True)
        t.start()

    def _purge_finished(self, ttl: float | None = None) -> int:
        """Clean up completed jobs to prevent memory leaks.
        
        Removes finished jobs from internal tracking to free memory.
        Can be called periodically for long-running applications.
        
        Args:
            ttl: Time-to-live in seconds (None = remove all finished jobs)
            
        Returns:
            Number of jobs that were purged
        """
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

    def cancel(self, disp_id: str) -> bool:
        """Cancel a pending or running request.
        
        Attempts to cancel a request at any stage of execution. Success depends
        on current request state and whether cancellation can be performed safely.
        
        Args:
            disp_id: Dispatch ID to cancel

        Returns:
            True if cancellation was successful, False otherwise
            
        Cancellation Behavior:
            - QUEUED requests: Cancelled immediately before execution starts
            - RUNNING/STREAMING requests: Marked for cancellation, may complete naturally
            - DONE/ERROR/CANCELLED requests: Cannot be cancelled (returns False)
            
        Example:
            disp_id = dispatcher.submit_proc(llm, message)

            # Later decide to cancel
            if dispatcher.cancel(disp_id):
                print("Request cancelled successfully")
            else:
                print("Could not cancel request (may have completed)")
        """
        with self._lock:
            job = self._jobs.get(disp_id)
            if not job:
                return False
            if job.state in (RequestState.DONE, RequestState.ERROR, RequestState.CANCELLED):
                return False

            job.cancelled = True
            is_stream = job.stream_q is not None
            cb, rid = job.callback, job.disp_id

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
        """Shutdown the dispatcher event loop.
        
        Stops the internal asyncio event loop used for request processing.
        Should be called when the dispatcher is no longer needed, particularly
        in testing scenarios or during application shutdown.
        
        Note:
            This is primarily for cleanup in tests and shutdown procedures.
            In normal operation, the singleton dispatcher runs for the application lifetime.
        """
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
        disp_id = str(uuid.uuid4())
        job = _Job(
            disp_id=disp_id,
            future=Future(),
            state=RequestState.QUEUED,
            queued_at=time.time(),
            stream_q=Queue() if stream else None,
            callback=callback,
        )
        with self._lock:
            self._jobs[disp_id] = job
        return disp_id, job

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
            cb, disp_id = job.callback, job.disp_id

            if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                job.deferred_cb = (result, None)
                return  # consumer will fire it

        # non-stream → fire now (result() will purge)
        if not is_stream:
            if cb:
                def _run():
                    try: cb(disp_id, result, None)
                    except Exception:
                        try: traceback.print_exc()
                        except Exception: pass
                threading.Thread(target=_run, daemon=True).start()
            return

        # streaming + no consumer → fire now and purge
        if cb:
            def _run():
                try: cb(disp_id, result, None)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        with self._lock:
            self._jobs.pop(disp_id, None)

    def _complete_err(self, job, exc: BaseException):
        with self._lock:
            job.ended_at = time.time()
            job.state = RequestState.ERROR
            job.error = exc
            is_stream = job.stream_q is not None
            cb, disp_id = job.callback, job.disp_id

            if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                job.deferred_cb = (None, exc)
                return

        if not is_stream:
            if cb:
                def _run():
                    try: cb(disp_id, None, exc)
                    except Exception:
                        try: traceback.print_exc()
                        except Exception: pass
                threading.Thread(target=_run, daemon=True).start()
            return

        if cb:
            def _run():
                try: cb(disp_id, None, exc)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        with self._lock:
            self._jobs.pop(disp_id, None)


    def _complete_cancel(self, job):
        with self._lock:
            job.ended_at = time.time()
            job.state = RequestState.CANCELLED
            is_stream = job.stream_q is not None
            cb, disp_id = job.callback, job.disp_id

            if is_stream and job.stream_reader_created and not job.stream_consumed and cb:
                job.deferred_cb = (None, None)
                return

        if not is_stream:
            if cb:
                def _run():
                    try: cb(disp_id, None, None)
                    except Exception:
                        try: traceback.print_exc()
                        except Exception: pass
                threading.Thread(target=_run, daemon=True).start()
            return

        if cb: 
            def _run():
                try: cb(disp_id, None, None)
                except Exception:
                    try: traceback.print_exc()
                    except Exception: pass
            threading.Thread(target=_run, daemon=True).start()

        with self._lock:
            self._jobs.pop(disp_id, None)
