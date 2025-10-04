import typing as t
import asyncio

from ._state import BaseState, State, StateStatus
from ._event import Event, Post
from ._region import Region
from dachi.core import Ctx, ModuleList


class CompositeState(BaseState):
    """Composite state containing nested regions."""
    regions: ModuleList[Region]

    def __post_init__(self):
        
        super().__post_init__()
        self._tasks = []
        self._finished_regions = set()

    async def finish_region(self, region: str) -> None:
        """Handle completion of a region's task."""
        self._finished_regions.add(region)
        self.regions[region].unregister_finish_callback(self.finish_region)
        if len(self._finished_regions) == len(self.regions):
            # All regions have completed
            self._tasks = []
            self._status.set(StateStatus.COMPLETED)
            await self.finish()
        
    async def execute(self, post: "Post", ctx: Ctx) -> None:
        """Composite states do not have direct execution logic."""
        self._tasks = []
        for region in self.regions:
            self._tasks.append(
                asyncio.create_task(
                region.start(post.child(region.name, None), ctx)
            ))
            region.register_finish_callback(self.finish_region, region.name, post, ctx)
        return None
    
    async def run(self, post: "Post", ctx: Ctx) -> None:
        """Run all regions and wait for them to complete."""
        if len(self.regions) == 0:
            self._status.set(StateStatus.COMPLETED)
            await self.finish()
            return
        await self.execute(post, ctx)
        self._run_completed.set(False)
        # don't call finish until all regions have completed
