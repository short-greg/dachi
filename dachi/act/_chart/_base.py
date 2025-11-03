from __future__ import annotations

from typing import Any, Dict, List, Union, Optional, Callable, Tuple, Literal
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging

from dachi.core import BaseModule, Attr, Ctx, Scope, RestrictedSchemaMixin
from ._event import EventPost, EventQueue

logger = logging.getLogger("dachi.statechart")


JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class ChartStatus(Enum):
    """State lifecycle status."""
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTING = "preempting"
    SUCCESS = "success"
    CANCELED = "canceled"
    FAILURE = "failure"

    def is_waiting(self) -> bool:
        return self == ChartStatus.WAITING

    def is_running(self) -> bool:
        return self == ChartStatus.RUNNING

    def is_preempting(self) -> bool:
        return self == ChartStatus.PREEMPTING

    def is_success(self) -> bool:
        return self == ChartStatus.SUCCESS

    def is_canceled(self) -> bool:
        return self == ChartStatus.CANCELED

    def is_failure(self) -> bool:
        return self == ChartStatus.FAILURE

    def is_completed(self) -> bool:
        """Returns True if in a final state (SUCCESS, FAILURE, or CANCELED)."""
        return self in (ChartStatus.SUCCESS, ChartStatus.FAILURE, ChartStatus.CANCELED)


# TODO: Determine if StatusResult is needed
# class StatusResult:
#     """Result of a status check."""
#     def __init__(self, status: ChartStatus, message: Optional[str] = None):
#         self.status = status
#         self.message = message


class InvalidTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class Recoverable(ABC):
    """Protocol for objects that can recover their internal state.

    Implemented by:
    - Region: Recovers to last active state
    - CompositeState: Recovers child regions

    NOT implemented by:
    - StateChart: Top-level, never recovered (just calls recover on children)
    """

    @abstractmethod
    def can_recover(self) -> bool:
        """Check if recovery is possible.

        Returns:
            True if this object has state that can be recovered

        Examples:
            - Region: True if _last_active_state is not None
            - CompositeState: True if any child region can recover
        """
        pass

    @abstractmethod
    def recover(self, policy: Literal["shallow", "deep"]) -> None:
        """Recover internal state using the given policy.

        MUST call can_recover() first and raise error if False.

        Args:
            policy: Recovery policy
                - "shallow": Restore to last active immediate substate only
                - "deep": Recursively restore entire nested state tree

        Raises:
            RuntimeError: If can_recover() is False
        """
        pass


class ChartBase(BaseModule):
    """Base class for all state types."""
    name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = self.__class__.__name__
        self._status = Attr[ChartStatus](data=ChartStatus.WAITING)
        self._finish_callbacks: Dict[Callable, Tuple[tuple, dict]] = {}

    @property
    def status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()

    def is_running(self) -> bool:
        return self._status.get().is_running()

    def is_completed(self) -> bool:
        return self._status.get().is_completed()
    
    def is_waiting(self) -> bool:
        return self._status.get().is_waiting()

    def reset(self) -> None:
        """Reset state to WAITING. Must be implemented by subclasses."""
        self._status.set(ChartStatus.WAITING)

    def can_reset(self) -> bool:
        """Check if state can be reset."""
        return self._status.get().is_completed()

    async def cancel(self) -> None:
        """Cancel this component and all its resources.

        Only acts if not already in a finished state (SUCCESS, FAILURE, CANCELED).
        Subclasses should override to cancel their specific resources first,
        then call super().cancel() to set the status.
        """
        if self._status.get().is_completed():
            return

        self._status.set(ChartStatus.CANCELED)

    async def finish(self, post: EventPost | None=None, ctx: Ctx | None=None) -> None:
        """Mark as finished and invoke finish callbacks.

        Cancels all timers created by this component's Post instance,
        then invokes registered callbacks. Exceptions in callbacks are
        logged but don't prevent other callbacks from running.

        Args:
            post: Post object for event posting
            ctx: Context object for this component
        """
        if post is not None:
            post.cancel_all()

        callbacks_copy = list(self._finish_callbacks.items())
        for callback, (args, kwargs) in callbacks_copy:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Finish callback failed for '{self.name}': {e}",
                    exc_info=True,
                    extra={
                        "callback": callback.__name__ if hasattr(callback, "__name__") else str(callback),
                        "component_name": self.name,
                        "component_type": self.__class__.__name__
                    }
                )
                # Continue with remaining callbacks

    def register_finish_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Register a callback to be called when finish() is invoked.

        Note: If the same callback is registered multiple times with different
        args/kwargs, only the most recent registration will be kept. This
        simplifies callback removal but means you cannot register the same
        callback with different arguments.
        """
        self._finish_callbacks[callback] = (args, kwargs)

    def unregister_finish_callback(self, callback: Callable) -> None:
        """Unregister a finish callback."""
        if callback in self._finish_callbacks:
            del self._finish_callbacks[callback]

    def get_status(self) -> ChartStatus:
        """Get current state status."""
        return self._status.get()


class RestrictedStateSchemaMixin(RestrictedSchemaMixin):
    """
    Mixin for state charts with state-specific schema restrictions.

    Uses isinstance(variant, RestrictedStateSchemaMixin) for recursion checks.
    This ensures we only recurse on state-compatible classes, preventing
    task/state cross-contamination.

    This mixin provides the domain-specific behavior for state charts,
    inheriting all base functionality from core.RestrictedSchemaMixin.
    """

    @classmethod
    def restricted_schema(
        cls,
        *,
        states: list | None = None,
        _profile: str = "shared",
        _seen: dict | None = None,
        **kwargs
    ) -> dict:
        """
        Generate restricted schema for state chart states.

        Must be implemented by subclasses (e.g., Region, CompositeState, StateChart).

        Args:
            states: List of allowed state variants (can be State classes, StateSpec classes,
                   StateSpec instances, or schema dicts)
            _profile: "shared" (use $defs/Allowed_*) or "inline" (use oneOf)
            _seen: Cycle detection dict
            **kwargs: Additional arguments passed to nested restricted_schema() calls

        Returns:
            Restricted schema dict

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement restricted_schema()"
        )
