from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Union, Optional
import inspect
from dachi.core import Attr, Ctx
from abc import abstractmethod
from enum import Enum

from dachi.core import BaseModule
from ._event import Post
from dachi.utils._utils import resolve_fields, resolve_from_signature

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class StateStatus(Enum):
    """State lifecycle status."""
    WAITING = "waiting"
    RUNNING = "running" 
    COMPLETED = "completed"
    PREEMPTED = "preempted"


class BaseState(BaseModule):
    """Base class for all state types."""
        
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
        
    def __post_init__(self):
        super().__post_init__()
        self._status = Attr[StateStatus](data=StateStatus.WAITING)
        self._termination_requested = Attr[bool](data=False)
        self._run_completed = Attr[bool](data=False)

    def enter(self) -> None:
        """Called when entering the state."""
        self._status.set(StateStatus.RUNNING)
        self._termination_requested.set(False)
        self._run_completed.set(False)

    @abstractmethod
    async def execute(self, post: "Post", **inputs: Any) -> Optional[Any]:
        """Execute the state's main logic. Return optional output."""
        pass

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
            
        except Exception:
            # Don't set _run_completed on exception
            raise
        finally:
            await post.finish()

    def exit(self) -> None:
        """Called when exiting the state. Sets final status."""
        if self._run_completed.get():
            # Case 1: run() already completed successfully
            self._status.set(StateStatus.COMPLETED)
        else:
            # Case 2: run() hasn't completed, request termination
            self._status.set(StateStatus.PREEMPTED)
            self._termination_requested.set(True)

    def is_final(self) -> bool:
        """Return True if this is a final state."""
        return False
        
    def request_termination(self) -> None:
        """Request termination for preemptible states."""
        self._termination_requested.set(True)
        
    def get_status(self) -> StateStatus:
        """Get current state status."""
        return self._status.get()
    
    def build_inputs(self, ctx) -> dict:
        """Build inputs from context data using class definition or function signature"""
        if hasattr(self.__class__, 'inputs'):
            # Use inputs class if defined
            return resolve_fields(ctx, self.__class__.inputs)
        else:
            # Use function signature inspection, excluding 'post' parameter
            return resolve_from_signature(ctx, self.execute, exclude_params={'post'})


class State(BaseState):
    """Single-shot state that runs execute() to completion."""
    
    # States inherit the abstract execute method from BaseState


class StreamState(BaseState):
    """Streaming state with preemption at yield points."""
    
    @abstractmethod
    async def astream(self, post: "Post", **inputs: Any) -> AsyncIterator[Optional[Any]]:
        """Implement streaming state logic. Yield optional outputs at checkpoints."""
        yield  # Must have at least one yield for preemption
    
    async def execute(self, post: "Post", **inputs: Any) -> Optional[Any]:
        """Execute astream with preemption checks at each yield."""
        last_result = None
        async for result in self.astream(post, **inputs):
            # Check for termination request at each checkpoint
            if self._termination_requested.get():
                break
            last_result = result
        
        # Return the last yielded result
        return last_result
    
    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Execute streaming state with context updates."""
        try:
            # Build inputs from context using framework pattern
            inputs = self.build_inputs(ctx)
            
            async for result in self.astream(post, **inputs):
                # Check for termination request at each checkpoint
                if self._termination_requested.get():
                    break
                    
                if result is not None:
                    # Update context with streamed output
                    ctx.update(result)
            
            # If we got here without termination, we completed successfully
            if not self._termination_requested.get():
                self._run_completed.set(True)
                
        except Exception:
            # Don't set _run_completed on exception
            raise
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

