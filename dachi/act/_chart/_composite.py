from __future__ import annotations
import typing as t
import asyncio

from ._state import BaseState
from ._base import ChartStatus, InvalidTransition
from ._event import Post
from ._region import Region
from dachi.core import Ctx, ModuleList


class CompositeState(BaseState):
    """Composite state containing nested regions."""
    regions: ModuleList

    def __post_init__(self):

        super().__post_init__()
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

    async def finish_region(self, region_name: str) -> None:
        """Handle completion of a region's task."""
        self._finished_regions.add(region_name)

        # Find region by name and unregister callback
        for region in self.regions:
            if region.name == region_name:
                region.unregister_finish_callback(self.finish_region)
                break

        if len(self._finished_regions) == len(self.regions) and self._exiting.get():
            # All regions have completed
            self._tasks = []
            self._status.set(ChartStatus.SUCCESS)
            self._run_completed.set(True)
            await self.finish()

    def reset(self):
        """Reset composite state to initial state.

        Raises: InvalidTransition if state cannot be reset.
        """
        # Cancel all running tasks before clearing
        for task in self._tasks:
            if not task.done():
                task.cancel()

        super().reset()
        self._tasks = []
        self._finished_regions = set()
        
    async def execute(self, post: "Post", ctx: Ctx) -> None:
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

    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Run all regions and wait for them to complete."""
        if not self.can_run():
            raise RuntimeError(f"Cannot run state '{self.name}' in {self._status.get()} state")
        self._status.set(ChartStatus.RUNNING)
        if len(self.regions) == 0:
            self._status.set(ChartStatus.SUCCESS)
            await self.finish()
            return
        await self.execute(post, ctx)
        self._run_completed.set(False)
        # don't call finish until all regions have completed

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

        # Check completion status BEFORE stopping regions
        all_completed = all([region.is_completed() for region in self.regions])

        # Stop any running regions
        for i, region in enumerate(self.regions):
            if region.is_running():
                region.unregister_finish_callback(self.finish_region)
                await region.stop(post.child(region.name), ctx.child(i), preempt=True)

        self._termination_requested.set(True)

        if all_completed:
            self._run_completed.set(True)
            self._status.set(ChartStatus.SUCCESS)
            await self.finish()
        else:
            self._run_completed.set(False)
            self._status.set(ChartStatus.PREEMPTING)
