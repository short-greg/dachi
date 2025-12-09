import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import directly from internal modules to avoid broken import chain in dachi.act.__init__
from dachi.act.chart._build import (
    build_chart,
    build_region,
    build_composite,
    ChartBuilder,
    RegionBuilder,
    CompositeBuilder,
)
from dachi.act.chart._chart import StateChart
from dachi.act.chart._region import Region
from dachi.act.chart._composite import CompositeState
from dachi.act.chart._state import State


class IdleState(State):
    async def execute(self, post, **inputs):
        await post.aforward("start_work")
        return {"status": "idle_complete"}


class WorkingState(State):
    async def execute(self, post, **inputs):
        await post.aforward("work_done")
        return {"status": "work_complete"}


class TestBuildRegion:
    """Test build_region context manager."""

    def test_build_region_creates_region(self):
        """Test build_region creates a Region instance."""
        with build_region("test_region") as region:
            assert isinstance(region, RegionBuilder)
            assert region.region.name == "test_region"

    def test_build_region_adds_state(self):
        """Test build_region can add states."""
        with build_region("test_region") as region:
            idle_state = IdleState(name="idle")
            region.add_state(idle_state)

        assert "idle" in region.region.states
        assert region.region.states["idle"] == idle_state

    def test_build_region_adds_multiple_states(self):
        """Test build_region can add multiple states."""
        with build_region("test_region") as region:
            idle_state = IdleState(name="idle")
            working_state = WorkingState(name="working")
            region.add_state(idle_state)
            region.add_state(working_state)

        assert "idle" in region.region.states
        assert "working" in region.region.states

    def test_build_region_with_composite(self):
        """Test build_region can nest a composite."""
        with build_region("test_region") as region:
            with region.composite("composite1") as composite:
                assert isinstance(composite, CompositeBuilder)
                assert composite.composite.name == "composite1"

    def test_build_region_builds_on_exit(self):
        """Test build_region builds the region on context exit."""
        with build_region("test_region") as region:
            idle_state = IdleState(name="idle")
            region.add_state(idle_state)

        built_region = region.region
        assert isinstance(built_region, Region)
        assert "idle" in built_region.states


class TestBuildComposite:
    """Test build_composite context manager."""

    def test_build_composite_creates_composite(self):
        """Test build_composite creates a CompositeState instance."""
        with build_composite("test_composite") as composite:
            assert isinstance(composite, CompositeBuilder)
            assert composite.composite.name == "test_composite"

    def test_build_composite_adds_region(self):
        """Test build_composite can add regions."""
        with build_composite("test_composite") as composite:
            with composite.region("region1") as region:
                assert isinstance(region, RegionBuilder)
                assert region.region.name == "region1"

    def test_build_composite_adds_multiple_regions(self):
        """Test build_composite can add multiple regions."""
        with build_composite("test_composite") as composite:
            with composite.region("region1") as region1:
                idle1 = IdleState(name="idle1")
                region1.add_state(idle1)

            with composite.region("region2") as region2:
                idle2 = IdleState(name="idle2")
                region2.add_state(idle2)

        built_composite = composite.composite
        assert len(built_composite.regions) == 2
        assert built_composite.regions[0].name == "region1"
        assert built_composite.regions[1].name == "region2"

    def test_build_composite_builds_on_exit(self):
        """Test build_composite builds the composite on context exit."""
        with build_composite("test_composite") as composite:
            with composite.region("region1") as region:
                idle = IdleState(name="idle")
                region.add_state(idle)

        built_composite = composite.composite
        assert isinstance(built_composite, CompositeState)
        assert len(built_composite.regions) == 1


class TestBuildChart:
    """Test build_chart context manager."""

    def test_build_chart_creates_chart(self):
        """Test build_chart creates a StateChart instance."""
        chart = StateChart(name="test_chart")
        with build_chart(chart) as builder:
            assert isinstance(builder, ChartBuilder)
            assert builder.chart == chart

    def test_build_chart_adds_region(self):
        """Test build_chart can add regions."""
        chart = StateChart(name="test_chart")
        with build_chart(chart) as builder:
            with builder.region("region1") as region:
                assert isinstance(region, RegionBuilder)
                assert region.region.name == "region1"

    def test_build_chart_adds_multiple_regions(self):
        """Test build_chart can add multiple regions."""
        chart = StateChart(name="test_chart")
        with build_chart(chart) as builder:
            with builder.region("region1") as region1:
                idle1 = IdleState(name="idle1")
                region1.add_state(idle1)

            with builder.region("region2") as region2:
                idle2 = IdleState(name="idle2")
                region2.add_state(idle2)

        assert len(chart.regions) == 2
        assert chart.regions[0].name == "region1"
        assert chart.regions[1].name == "region2"

    def test_build_chart_builds_on_exit(self):
        """Test build_chart builds the chart on context exit."""
        chart = StateChart(name="test_chart")
        with build_chart(chart) as builder:
            with builder.region("region1") as region:
                idle = IdleState(name="idle")
                region.add_state(idle)

        assert len(chart.regions) == 1
        assert isinstance(chart.regions[0], Region)


class TestBuildChartNested:
    """Test nested builder pattern with full hierarchy."""

    def test_build_chart_with_nested_composite(self):
        """Test build_chart with nested composite and states."""
        chart = StateChart(name="test_chart")

        with build_chart(chart) as builder:
            with builder.region("main_region") as region:
                idle = IdleState(name="idle")
                region.add_state(idle)

                with region.composite("composite1") as composite:
                    with composite.region("subregion1") as subregion:
                        working = WorkingState(name="working")
                        subregion.add_state(working)

                working2 = WorkingState(name="working2")
                region.add_state(working2)

        assert len(chart.regions) == 1
        main_region = chart.regions[0]
        assert main_region.name == "main_region"
        assert "idle" in main_region.states
        assert "composite1" in main_region.states
        assert "working2" in main_region.states

        composite = main_region.states["composite1"]
        assert isinstance(composite, CompositeState)
        assert len(composite.regions) == 1
        assert composite.regions[0].name == "subregion1"
        assert "working" in composite.regions[0].states

    def test_build_chart_with_multiple_nested_composites(self):
        """Test build_chart with multiple nested composites."""
        chart = StateChart(name="test_chart")

        with build_chart(chart) as builder:
            with builder.region("main_region") as region:
                with region.composite("composite1") as composite1:
                    with composite1.region("sub1") as sub1:
                        idle1 = IdleState(name="idle1")
                        sub1.add_state(idle1)

                with region.composite("composite2") as composite2:
                    with composite2.region("sub2") as sub2:
                        idle2 = IdleState(name="idle2")
                        sub2.add_state(idle2)

        main_region = chart.regions[0]
        assert "composite1" in main_region.states
        assert "composite2" in main_region.states

        composite1 = main_region.states["composite1"]
        composite2 = main_region.states["composite2"]
        assert isinstance(composite1, CompositeState)
        assert isinstance(composite2, CompositeState)

    def test_build_chart_with_deeply_nested_composites(self):
        """Test build_chart with deeply nested composite hierarchy."""
        chart = StateChart(name="test_chart")

        with build_chart(chart) as builder:
            with builder.region("main_region") as region:
                with region.composite("composite1") as composite1:
                    with composite1.region("level1") as level1:
                        # Note: Currently CompositeState cannot contain nested CompositeStates
                        # This is a limitation of the current design
                        idle = IdleState(name="idle")
                        level1.add_state(idle)

        main_region = chart.regions[0]
        composite1 = main_region.states["composite1"]
        assert len(composite1.regions) == 1
        assert composite1.regions[0].name == "level1"


class TestBuilderExceptionHandling:
    """Test that builders don't swallow exceptions."""

    def test_build_region_propagates_exception(self):
        """Test build_region propagates exceptions raised inside context."""
        with pytest.raises(ValueError, match="test error"):
            with build_region("test_region") as region:
                raise ValueError("test error")

    def test_build_composite_propagates_exception(self):
        """Test build_composite propagates exceptions raised inside context."""
        with pytest.raises(ValueError, match="test error"):
            with build_composite("test_composite") as composite:
                raise ValueError("test error")

    def test_build_chart_propagates_exception(self):
        """Test build_chart propagates exceptions raised inside context."""
        chart = StateChart(name="test_chart")
        with pytest.raises(ValueError, match="test error"):
            with build_chart(chart) as builder:
                raise ValueError("test error")
