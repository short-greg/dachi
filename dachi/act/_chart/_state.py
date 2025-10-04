from __future__ import annotations

from typing import Any, Dict, List, Union, Optional, Callable, Tuple
from abc import ABC
from abc import abstractmethod
from enum import Enum, auto
import asyncio

from dachi.core import Attr, Ctx
from ._event import Post
from dachi.utils._utils import resolve_fields, resolve_from_signature
from ._base import ChartBase, ChartStatus
import typing as t


JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

from ._base import ChartBase, ChartStatus


class RunResult(Enum):
    COMPLETED = auto()   # State finished normally
    PREEMPTED = auto()   # State was cancelled/preempted


# class StateStatus(Enum):
#     """State lifecycle status."""
#     WAITING = "waiting"
#     RUNNING = "running" 
#     COMPLETED = "completed"
#     PREEMPTED = "preempted"
#     FAILED = "failed"
#     CANCELED = "canceled"


class BaseState(ChartBase, ABC):
    """Base class for all state types."""
    name: Optional[str] = None
        
    def __post_init__(self):
        super().__post_init__()
        # Auto-generate name from class if not provided
        if self.name is None:
            self.name = self.__class__.__name__

        self._status = Attr[ChartStatus](data=ChartStatus.WAITING)
        self._termination_requested = Attr[bool](data=False)
        self._run_completed = Attr[bool](data=False)
        self._finish_callbacks: List[Tuple[Callable, tuple, dict]] = []

    # async def finish(self) -> None:
    #     """Mark region as finished and invoke finish callbacks"""
    #     self._status.set(ChartStatus.COMPLETED)
    #     for callback, args, kwargs in self._finish_callbacks:
    #         if asyncio.iscoroutinefunction(callback):
    #             await callback(*args, **kwargs)
    #         else:
    #             callback(*args, **kwargs)

    def enter(self, post: Post, ctx: Ctx) -> None:
        """Called when entering the state."""
        self._status.set(ChartStatus.RUNNING)
        self._termination_requested.set(False)
        self._run_completed.set(False)

    @abstractmethod
    async def execute(self, post: "Post", **inputs: Any) -> t.Iterator[t.Dict | None] | Optional[t.Dict]:
        """Execute the state's main logic. Return optional output."""
        pass

    @abstractmethod
    async def run(self, post: "Post", ctx: Ctx) -> None:
        pass

    def exit(self) -> None:
        """Called when exiting the state. Sets final status."""
        if self._run_completed.get():
            # Case 1: run() already completed successfully
            self._status.set(ChartStatus.COMPLETED)
        else:
            # Case 2: run() hasn't completed, request termination
            self._status.set(ChartStatus.PREEMPTED)
            self._termination_requested.set(True)

    def is_final(self) -> bool:
        """Return True if this is a final state."""
        return False
        
    def request_termination(self) -> None:
        """Request termination for preemptible states."""
        self._termination_requested.set(True)
        
    def get_status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()
    
        
class LeafState(BaseState, ABC):
    """Leaf state that does not contain nested regions."""
        
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Process optional inputs, outputs, and emit declarations
        sc_params = {"inputs": {}, "outputs": {}, "emit": {}}
        
        if hasattr(cls, 'inputs'):
            sc_params["inputs"] = cls._process_ports(cls.inputs)
        
        if hasattr(cls, 'outputs'):
            sc_params["outputs"] = cls._process_ports(cls.outputs)
            
        if hasattr(cls, 'emit'):
            sc_params["emit"] = cls._process_ports(cls.emit)
        
        cls.sc_params = sc_params
    
    @classmethod
    def _process_ports(cls, port_class):
        """Extract port information from inputs, outputs, or emit class"""
        if port_class is None:
            return {}
        
        # Get annotations from the class
        annotations = getattr(port_class, '__annotations__', {})
        
        port_info = {}
        for name, type_hint in annotations.items():
            # Skip private attributes
            if name.startswith('_'):
                continue
                
            info = {"type": type_hint}
            
            # Check if there's a default value
            if hasattr(port_class, name):
                default_value = getattr(port_class, name)
                info["default"] = default_value
            
            port_info[name] = info
        
        return port_info

    def build_inputs(self, ctx) -> dict:
        """Build inputs from context data using class definition or function signature"""
        if hasattr(self.__class__, 'inputs'):
            # Use inputs class if defined
            return resolve_fields(ctx, self.__class__.inputs)
        else:
            # Use function signature inspection, excluding 'post' parameter
            return resolve_from_signature(ctx, self.execute, exclude_params={'post'})


class State(LeafState):
    """Single-shot state that runs execute() to completion."""
    
    # States inherit the abstract execute method from BaseState

    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Execute state with inputs built from context."""
        try:
            # Build inputs from context using framework pattern
            inputs = self.build_inputs(ctx)
            result = await self.execute(post, **inputs)
            if result is not None:
                # Update context with result
                ctx.update(result)
            
            # Mark that we completed successfully  
            self._run_completed.set(True)
            self._status.set(ChartStatus.COMPLETED)
            self._finish_callbacks.clear()
        except asyncio.CancelledError:
            # Don't set _run_completed on exception
            self._status.set(ChartStatus.CANCELED)

        except Exception:
            # Don't set _run_completed on exception
            self._status.set(ChartStatus.FAILED)
            # TODO: Decide whether to re-raise or log
        finally:
            await post.finish()


class StreamState(LeafState, ABC):
    """Streaming state with preemption at yield points."""
    
    @abstractmethod
    async def execute(self, post: "Post", **inputs: Any) -> t.Iterator[Optional[Any]]:
        """Execute astream with preemption checks at each yield."""
        pass
    
    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Execute streaming state with context updates."""
        try:
            # Build inputs from context using framework pattern
            inputs = self.build_inputs(ctx)
            
            async for result in self.execute(post, **inputs):
                # Check for termination request at each checkpoint
                if result is not None:
                    # Update context with streamed output
                    ctx.update(result)
            
                if self._termination_requested.get():
                    break
                    
            # If we got here without termination, we completed successfully
            self._run_completed.set(True)
            if self._termination_requested.get():
                self._status.set(ChartStatus.PREEMPTED)
            else:
                self._status.set(ChartStatus.COMPLETED)

        # I only want to catch the exception if the async task was cancelled what exception is that
        # ??? 
        except asyncio.CancelledError:
            # Don't set _run_completed on exception
            self._status.set(ChartStatus.CANCELED)
        except Exception:
            # Don't set _run_completed on exception
            self._status.set(ChartStatus.FAILED)
            # TODO: Decide whether to re-raise or log
        finally:
            await post.finish()


class FinalState(State):
    """Final state that marks region completion."""
    
    async def execute(self, post: "Post", **inputs: Any) -> Optional[Any]:
        """FinalState has no work to do."""
        return None
    
    def is_final(self) -> bool:
        """FinalState is always final."""
        return True

