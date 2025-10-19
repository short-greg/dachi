from __future__ import annotations
import typing as t
import asyncio

from ._state import BaseState
from ._base import ChartStatus, InvalidTransition
from ._event import EventPost, ChartEventHandler, Event
from ._region import Region, ValidationResult
from dachi.core import Ctx, ModuleList


class CompositeState(BaseState, ChartEventHandler):
    """Composite state containing nested regions.

    Lifecycle:
    - run() starts all child regions and returns immediately (non-blocking)
    - Child regions execute in parallel via async tasks
    - When all children complete, _run_completed is set to True
    - Like all states, requires an event to trigger exit/transition
    - Parent region must have rules to transition composite to next state

    The composite "stays" in run() conceptually until all children finish,
    even though run() returns immediately. This avoids busy-waiting loops
    by using callbacks (finish_region) to track completion.
    """
    regions: ModuleList

    def __post_init__(self):

        super().__post_init__()
        if isinstance(self.regions, list):
            self.regions = ModuleList(items=self.regions)
        self._tasks = []
        self._finished_regions = set()

    def __getitem__(self, region_name: str) -> Region:
        """Get region by name.

        Args:
            region_name: Name of the region to retrieve

        Returns:
            The region instance

        Raises:
            KeyError: If region not found
        """
        for region in self.regions:
            if region.name == region_name:
                return region
        raise KeyError(f"Region '{region_name}' not found in composite state '{self.name}'")

    async def finish_region(self, region_name: str, post: 'EventPost', ctx: 'Ctx') -> None:
        """Handle completion of a region's task.
        
        Args:
            region_name (str): The name of the region that finished
            post: Post object for this state
            ctx: Ctx object for this state
        Raises:
            ValueError: If the region is unknown
        """
        self._finished_regions.add(region_name)

        # Find region by name and unregister callback
        for region in self.regions:
            if region.name == region_name:
                region.unregister_finish_callback(self.finish_region)
                break

        if len(self._finished_regions) == len(self.regions):
            # All regions have completed
            self._tasks = []
            self._run_completed.set(True)
            if self._exiting.get():
                self._status.set(ChartStatus.SUCCESS)
                await self.finish(post, ctx)

    def reset(self):
        """Reset composite state to initial state.

        Raises: InvalidTransition if state cannot be reset.
        """
        # Cancel all running tasks before clearing
        for task in self._tasks:
            if not task.done():
                task.cancel()

        super().reset()
        for region in self.regions:
            region.reset()
        self._tasks = []
        self._finished_regions = set()
        
    async def execute(self, post: "EventPost", ctx: Ctx) -> None:
        """Composite states do not have direct execution logic."""
        self._tasks = []
        for i, region in enumerate(self.regions):
            self._tasks.append(
                asyncio.create_task(
                    region.start(post.child(region.name), ctx.child(i))
                )
            )
            region.register_finish_callback(self.finish_region, region.name, post, ctx)
        return None

    def can_run(self) -> bool:
        """Check if the composite state can run."""
        return (self._entered.get() and
                not self._executing.get() and
                not self._run_completed.get())

    async def run(self, post: "EventPost", ctx: Ctx) -> None:
        """Run all regions and wait for them to complete."""
        if not self.can_run():
            raise RuntimeError(f"Cannot run state '{self.name}' in {self._status.get()} state")
        self._status.set(ChartStatus.RUNNING)
        if len(self.regions) == 0:
            self._status.set(ChartStatus.SUCCESS)
            self._run_completed.set(True)
            if self._exiting.get():
                await self.finish(post, ctx)
            return
        await self.execute(post, ctx)
        self._run_completed.set(False)
        # don't call finish until all regions have completed

    async def handle_event(self, event: Event, post: "EventPost", ctx: Ctx) -> None:
        """Handle events by dispatching to all running child regions."""
        if self._status.get() != ChartStatus.RUNNING:
            return

        for i, region in enumerate(self.regions):
            if region.is_running():
                child_post = post.child(region.name)
                child_ctx = ctx.child(i)
                await region.handle_event(event, child_post, child_ctx)

    def exit(self, post: EventPost, ctx: Ctx) -> None:
        """Called when exiting the state. Sets final status.

        Exit is synchronous - it initiates stopping of child regions but doesn't
        wait for them. When all regions finish, finish_region() will complete
        the exit process.

        Raises:
            InvalidStateTransition: If state cannot be exited.
        """
        if not self.can_exit():
            raise InvalidTransition(
                f"Cannot exit state '{self.name}' from status {self._status.get()}. "
                f"Must be entered and RUNNING, and not already exiting."
            )

        self._exiting.set(True)

        # Check completion status BEFORE stopping regions
        all_completed = all([region.is_completed() for region in self.regions])

        # Stop any running regions (scheduled as tasks, not awaited)
        for i, region in enumerate(self.regions):
            if region.is_running():
                region.unregister_finish_callback(self.finish_region)
                # Schedule stop as task, don't wait for it
                loop = asyncio.get_running_loop()
                loop.create_task(region.stop(post.child(region.name), ctx.child(i), preempt=True))

        self._termination_requested.set(True)

        if all_completed:
            # All regions already completed - trigger finish via _check_execute_finish
            self._run_completed.set(True)
            self._status.set(ChartStatus.SUCCESS)
            self._check_execute_finish(post, ctx)
        else:
            # Regions still running - will complete via finish_region callback
            self._status.set(ChartStatus.PREEMPTING)

    def validate(self) -> t.List[ValidationResult]:
        """Validate all child regions.

        Delegates validation to each child region and collects results.

        Returns:
            List of ValidationResult, one per child region
        """
        results = []
        for region in self.regions:
            results.append(region.validate())
        return results
