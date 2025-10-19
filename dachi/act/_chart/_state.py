from __future__ import annotations

from typing import Any, Dict, List, Union, Optional, Callable, Tuple
from abc import ABC
import typing as t
from abc import abstractmethod
from enum import Enum, auto
import asyncio
import logging

from dachi.core import Attr, Ctx, BaseModule
from ._event import Post
from dachi.utils._utils import resolve_fields, resolve_from_signature
from ._base import ChartBase, ChartStatus, InvalidTransition

logger = logging.getLogger("dachi.statechart")

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class RunResult(Enum):
    COMPLETED = auto()   # State finished normally
    PREEMPTED = auto()   # State was cancelled/preempted


class PseudoState(BaseModule):
    """Marker state that is entered but not executed.

    PseudoStates include initial states (READY) and final states.
    They are entered as part of the region lifecycle but do not
    have executable behavior.
    """

    name: str


class FinalState(PseudoState):
    """Final state that marks region completion."""

    name: str = "FINAL"
    status: Attr[ChartStatus] = Attr(ChartStatus.SUCCESS)


class ReadyState(PseudoState):
    """Built-in ready state. Region begins here before starting.

    READY is a marker state that does no work. It exists to ensure
    regions are always in a defined state, even at initialization.
    When region.start() is called, it automatically transitions from
    READY to the initial state. Follows the same lifecycle as other
    states (enter → run → exit) but execute() completes immediately.
    """
    name: str = "READY"

    @property
    def status(self) -> ChartStatus:
        return ChartStatus.WAITING


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
        self._finishing = False  # Guard against double-finish

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

    def run_completed(self) -> bool:
        """Return True if state has completed its run."""
        return self._run_completed.get()

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
        self._finishing = False

    def _check_execute_finish(self):
        """Check if state is ready to finish and schedule if so.

        State is ready to finish when BOTH:
        - _run_completed is True
        - _exiting is True

        Uses _finishing flag to prevent double-finish race condition.
        """
        if self._run_completed.get() and self._exiting.get() and not self._finishing:
            self._finishing = True
            loop = asyncio.get_running_loop()
            loop.create_task(self.finish())


class AtomState(BaseState, ABC):
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

    def exit(self, post: Post, ctx: Ctx) -> None:
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
            self._check_execute_finish()
        else:
            self._status.set(ChartStatus.PREEMPTING)
            self._termination_requested.set(True)


class State(AtomState):
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

            # Normal completion
            if self._exiting.get() and self._status.get().is_running():
                self._status.set(ChartStatus.SUCCESS)

        except asyncio.CancelledError:
            self._status.set(ChartStatus.CANCELED)

        except Exception as e:
            # Log the exception with full traceback
            logger.error(
                f"State '{self.name}' failed with {type(e).__name__}: {e}",
                exc_info=True,
                extra={
                    "state": self.name,
                    "exception_type": type(e).__name__,
                }
            )

            # Store exception details in context
            ctx["__exception__"] = {
                "message": str(e),
                "type": type(e).__name__,
                "state": self.name,
            }

            # Mark state as failed (don't re-raise)
            self._status.set(ChartStatus.FAILURE)

        finally:
            self._run_completed.set(True)
            self._executing.set(False)
            if self._termination_requested.get() or self._status.get().is_completed():
                # Direct finish for termination/cancellation (not via exit flow)
                await self.finish()


class BoundState(BaseState):
    """Wrap a State with variable bindings for input resolution.

    Bindings remap context variables when building inputs:
        {"local_name": "context_path"}

    Example:
        state = BoundState(
            state=ProcessingState(),
            bindings={"input": "sensor_data", "config": "../global_config"}
        )

    The wrapped state reads inputs from bound paths but writes outputs
    to the original context automatically (BoundCtx handles this).
    """
    state: State
    bindings: Dict[str, str]

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = self.state.name

    def can_enter(self) -> bool:
        """Check if state can be entered."""
        return self.state.can_enter()

    def can_run(self) -> bool:
        """Check if state can run."""
        return self.state.can_run()

    def can_exit(self) -> bool:
        """Check if state can exit."""
        return self.state.can_exit()

    def enter(self, post: Post, ctx: Ctx) -> None:
        """Delegate enter to wrapped state."""
        self.state.enter(post, ctx)
        self._status.set(self.state._status.get())
        self._entered.set(self.state._entered.get())

    async def execute(self, post: Post, **inputs: Any) -> Optional[Dict]:
        """Not used - run() delegates directly."""
        pass

    async def run(self, post: Post, ctx: Ctx) -> None:
        """Run wrapped state with bound context for input resolution."""
        bound_ctx = ctx.bind(self.bindings)
        await self.state.run(post, bound_ctx)

        # Sync all status flags from wrapped state
        self._status.set(self.state._status.get())
        self._run_completed.set(self.state._run_completed.get())
        self._executing.set(self.state._executing.get())
        self._exiting.set(self.state._exiting.get())
        self._termination_requested.set(self.state._termination_requested.get())

    def exit(self, post: Post, ctx: Ctx) -> None:
        """Delegate exit to wrapped state."""
        self.state.exit(post, ctx)
        self._status.set(self.state._status.get())
        self._exiting.set(self.state._exiting.get())

    def request_termination(self) -> None:
        """Delegate termination to wrapped state."""
        self.state.request_termination()
        self._termination_requested.set(True)

    def reset(self) -> None:
        """Reset both wrapper and wrapped state."""
        self.state.reset()
        super().reset()


class StreamState(AtomState, ABC):
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
        yielded_count = 0

        try:
            inputs = self.build_inputs(ctx)

            async for result in self.execute(post, **inputs):
                yielded_count += 1

                if result is not None:
                    ctx.update(result)

                if self._termination_requested.get():
                    break

        except asyncio.CancelledError:
            self._status.set(ChartStatus.CANCELED)

        except Exception as e:
            # Log the exception with yield count for debugging
            logger.error(
                f"StreamState '{self.name}' failed after {yielded_count} yields: {e}",
                exc_info=True,
                extra={
                    "state": self.name,
                    "exception_type": type(e).__name__,
                    "yielded_count": yielded_count,
                }
            )

            # Store exception details with progress tracking
            ctx["__exception__"] = {
                "message": str(e),
                "type": type(e).__name__,
                "state": self.name,
                "yielded_count": yielded_count,  # Track progress before failure
            }

            # Mark state as failed (don't re-raise)
            self._status.set(ChartStatus.FAILURE)

        finally:
            self._run_completed.set(True)
            self._executing.set(False)
            if self._termination_requested.get():
                self._status.set(ChartStatus.CANCELED)
            elif self._exiting.get() and self._status.get().is_running():
                self._status.set(ChartStatus.SUCCESS)

            if self._status.get().is_completed():
                # Direct finish for termination/cancellation (not via exit flow)
                await self.finish()


class BoundStreamState(BaseState):
    """Wrap a StreamState with variable bindings for input resolution.

    Like BoundState but for streaming states with preemption support.
    """
    state: StreamState
    bindings: Dict[str, str]

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = self.state.name

    def can_enter(self) -> bool:
        """Check if state can be entered."""
        return self.state.can_enter()

    def can_run(self) -> bool:
        """Check if state can run."""
        return self.state.can_run()

    def can_exit(self) -> bool:
        """Check if state can exit."""
        return self.state.can_exit()

    def enter(self, post: Post, ctx: Ctx) -> None:
        """Delegate enter to wrapped state."""
        self.state.enter(post, ctx)
        self._status.set(self.state._status.get())
        self._entered.set(self.state._entered.get())

    async def execute(self, post: Post, **inputs: Any) -> t.Iterator[Optional[Dict]]:
        """Not used - run() delegates directly."""
        pass

    async def run(self, post: Post, ctx: Ctx) -> None:
        """Run wrapped streaming state with bound context."""
        bound_ctx = ctx.bind(self.bindings)
        await self.state.run(post, bound_ctx)

        # Sync all status flags
        self._status.set(self.state._status.get())
        self._run_completed.set(self.state._run_completed.get())
        self._executing.set(self.state._executing.get())
        self._exiting.set(self.state._exiting.get())
        self._termination_requested.set(self.state._termination_requested.get())

    def exit(self, post: Post, ctx: Ctx) -> None:
        """Delegate exit to wrapped state."""
        self.state.exit(post, ctx)
        self._status.set(self.state._status.get())
        self._exiting.set(self.state._exiting.get())

    def request_termination(self) -> None:
        """Delegate termination to wrapped state."""
        self.state.request_termination()
        self._termination_requested.set(True)

    def reset(self) -> None:
        """Reset both wrapper and wrapped state."""
        self.state.reset()
        super().reset()


# class ReadyState(State):
#     """Built-in ready state. Region begins here before starting.

#     READY is a marker state that does no work. It exists to ensure
#     regions are always in a defined state, even at initialization.
#     When region.start() is called, it automatically transitions from
#     READY to the initial state. Follows the same lifecycle as other
#     states (enter → run → exit) but execute() completes immediately.
#     """

#     async def execute(self, post: Post, **inputs) -> None:
#         """READY does nothing - it exists only as a marker."""
#         return None


# class FinalState(LeafState):
#     """Final state that marks region completion."""

#     async def execute(self, post: "Post", **inputs: Any) -> Optional[Any]:
#         """FinalState has no work to do."""
#         return None

#     async def run(self, post: "Post", ctx: Ctx) -> None:
#         """FinalState immediately completes upon run."""
#         if not self.can_run():
#             raise InvalidTransition(
#                 f"Cannot run FinalState '{self.name}' from status {self._status.get()}. "
#                 f"State must be RUNNING, not executing, and not completed."
#             )

#         self._executing.set(True)
#         self._run_completed.set(True)
#         self._executing.set(False)
#         self._status.set(ChartStatus.SUCCESS)
#         await self.finish()

#     def can_exit(self):
#         return False

#     def is_final(self) -> bool:
#         """FinalState is always final."""
#         return True
