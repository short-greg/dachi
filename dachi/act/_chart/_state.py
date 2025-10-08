from __future__ import annotations

from typing import Any, Dict, List, Union, Optional, Callable, Tuple
from abc import ABC
import typing as t
from abc import abstractmethod
from enum import Enum, auto
import asyncio

from dachi.core import Attr, Ctx
from ._event import Post
from dachi.utils._utils import resolve_fields, resolve_from_signature
from ._base import ChartBase, ChartStatus, InvalidTransition

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class RunResult(Enum):
    COMPLETED = auto()   # State finished normally
    PREEMPTED = auto()   # State was cancelled/preempted


class BaseState(ChartBase, ABC):
    """Base class for all state types."""
    name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = self.__class__.__name__

        self._status = Attr[ChartStatus](data=ChartStatus.WAITING)
        self._termination_requested = Attr[bool](data=False)
        self._run_completed = Attr[bool](data=False)
        self._executing = Attr[bool](data=False)
        self._entered = Attr[bool](data=False)
        self._exiting = Attr[bool](data=False)

    def can_enter(self) -> bool:
        """Check if state can be entered."""
        return not self._entered.get()

    def can_run(self) -> bool:
        """Check if state can run."""
        return (self._entered.get() and
                not self._executing.get() and
                not self._run_completed.get())

    def can_exit(self) -> bool:
        """Check if state can exit."""
        return (self._entered.get() and not self._exiting.get())

    def enter(self, post: Post, ctx: Ctx) -> None:
        """Called when entering the state.

        Raises:
            InvalidStateTransition: If state is not in WAITING status.
        """
        if not self.can_enter():
            raise InvalidTransition(
                f"Cannot enter state '{self.name}' from status {self._status.get()}. "
                f"Must be in WAITING status."
            )

        self._status.set(ChartStatus.RUNNING)
        self._termination_requested.set(False)
        self._run_completed.set(False)
        self._executing.set(False)
        self._entered.set(True)
        self._exiting.set(False)

    @abstractmethod
    async def execute(self, post: "Post", **inputs: Any) -> t.Iterator[t.Dict | None] | Optional[t.Dict]:
        """Execute the state's main logic. Return optional output."""
        pass

    @abstractmethod
    async def run(self, post: "Post", ctx: Ctx) -> None:
        pass

    def is_final(self) -> bool:
        """Return True if this is a final state."""
        return False

    def completed(self) -> bool:
        """Return True if state is in a completed status."""
        return self._status.get().is_completed()

    def request_termination(self) -> None:
        """Request termination for preemptible states."""
        self._termination_requested.set(True)

    def get_status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()

    def reset(self) -> None:
        """Reset state to WAITING.

        Raises:
            InvalidStateTransition: If state is not in a final status.
        """
        if not self.can_reset():
            raise InvalidTransition(
                f"Cannot reset state '{self.name}' from status {self._status.get()}. "
                f"Must be in a completed status (SUCCESS, FAILURE, or CANCELED)."
            )

        self._status.set(ChartStatus.WAITING)
        self._termination_requested.set(False)
        self._run_completed.set(False)
        self._executing.set(False)
        self._entered.set(False)
        self._exiting.set(False)
    
        
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

    async def exit(self, post: Post, ctx: Ctx) -> None:
        """Called when exiting the state. Sets final status.

        Raises:
            InvalidStateTransition: If state cannot be exited.
        """
        if not self.can_exit():
            raise InvalidTransition(
                f"Cannot exit state '{self.name}' from status {self._status.get()}. "
                f"Must be entered and RUNNING, and not already exiting."
            )

        self._exiting.set(True)

        if self._run_completed.get():
            if self._status.get().is_running():
                self._status.set(ChartStatus.SUCCESS)
            await self.finish()
        else:
            self._status.set(ChartStatus.PREEMPTING)
            self._termination_requested.set(True)


class State(LeafState):
    """Single-shot state that runs execute() to completion."""

    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Execute state with inputs built from context.

        Raises:
            InvalidStateTransition: If state cannot run.
        """
        if not self.can_run():
            raise InvalidTransition(
                f"Cannot run state '{self.name}' from status {self._status.get()}. "
                f"State must be RUNNING, not executing, and not completed."
            )

        self._executing.set(True)
        try:
            inputs = self.build_inputs(ctx)
            result = await self.execute(post, **inputs)
            if result is not None:
                ctx.update(result)
            if self._exiting.get():
                self._status.set(ChartStatus.SUCCESS)

        except asyncio.CancelledError:
            self._status.set(ChartStatus.CANCELED)

        except Exception:
            self._run_completed.set(True)
            self._executing.set(False)
            self._status.set(ChartStatus.FAILURE)
            raise

        finally:
            self._run_completed.set(True)
            self._executing.set(False)
            if self._termination_requested.get() or self._status.get() is ChartStatus.CANCELED:
                await self.finish()


class StreamState(LeafState, ABC):
    """Streaming state with preemption at yield points."""

    @abstractmethod
    async def execute(self, post: "Post", **inputs: Any) -> t.Iterator[Optional[Any]]:
        """Execute with preemption checks at each yield."""
        pass

    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Execute streaming state with context updates.

        Raises:
            InvalidStateTransition: If state cannot run.
        """
        if not self.can_run():
            raise InvalidTransition(
                f"Cannot run state '{self.name}' from status {self._status.get()}. "
                f"State must be RUNNING, not executing, and not completed."
            )

        self._executing.set(True)
        try:
            inputs = self.build_inputs(ctx)

            async for result in self.execute(post, **inputs):
                if result is not None:
                    ctx.update(result)

                if self._termination_requested.get():
                    break

            # Only mark completed if we finished naturally (not terminated)
            if self._termination_requested.get():
                self._status.set(ChartStatus.CANCELED)

        except asyncio.CancelledError:
            self._status.set(ChartStatus.CANCELED)

        except Exception:
            self._status.set(ChartStatus.FAILURE)
            self._run_completed.set(True)
            self._executing.set(False)
            raise

        finally:
            self._run_completed.set(True)
            self._executing.set(False)
        
            if self._termination_requested.get() or self._status.get() is ChartStatus.CANCELED:
                await self.finish()
            


class FinalState(State):
    """Final state that marks region completion."""
    
    async def execute(self, post: "Post", **inputs: Any) -> Optional[Any]:
        """FinalState has no work to do."""
        return None
    
    def is_final(self) -> bool:
        """FinalState is always final."""
        return True
