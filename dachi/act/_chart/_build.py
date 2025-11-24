from __future__ import annotations
from dachi.core import ModuleList
from . import _chart, _composite, _region, _state

"""

with build_chart() as chart:
    with chart.region('region1') as region:
        region.add_state(SomeState()) # adds the state
        with region.composite('composite1') as composite:
            with composite.region('subregion1') as subregion:
                subregion.add_state(AnotherState())
        region.add_state(ThirdState())

"""

# It should be able to build like above with context managers
# but also needs to handle events

# suggest the code below to enable this
# Currently the code below is not correct
# it doesn't follow the design


class CompositeBuilder:

    def __init__(self, name: str):
        self.name = name
        self.region_stack = []
        self._composite = None

    def region(self, name: str):
        """Create a child region builder for this composite.

        Args:
            name: Name of the region

        Returns:
            build_region context manager for the new region
        """
        region_builder = build_region(name)
        self.region_stack.append(region_builder)
        return region_builder

    def build(self):
        """Build the composite state with all child regions.

        Returns:
            Constructed CompositeState instance
        """
        regions = ModuleList(vals=[rb.region for rb in self.region_stack])
        self._composite = _composite.CompositeState(name=self.name, regions=regions)
        return self._composite

    @property
    def composite(self):
        """Get the built composite state."""
        if self._composite is None:
            self.build()
        return self._composite


class build_composite:
    """
    Context manager to build a Composite
    """

    def __init__(self, name: str):
        """
        Build a composite state from the states
        """
        self.composite_builder = CompositeBuilder(name)

    def __enter__(self):
        return self.composite_builder

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.composite_builder.build()
        return False

    @property
    def composite(self):
        """Get the built composite state."""
        return self.composite_builder.composite
                      

class RegionBuilder:

    def __init__(self, name: str):
        """Helper used by build_region to build a Region with a context manager

        Args:
            name (str): The name of the region to add
        """
        self._region = _region.Region(name=name, initial="READY", rules=[])
        self.composite_stack = []

    def add_state(self, state: _state.BaseState):
        """Add a state to the region.

        Args:
            state: State instance to add
        """
        self._region.add(state)

    def composite(self, name: str):
        """Create a composite in the region.

        Args:
            name (str): The name of the composite to add

        Returns:
            build_composite context manager
        """
        composite_builder = build_composite(name)
        self.composite_stack.append(composite_builder)
        return composite_builder

    def on(self, event_type: str):
        """Begin building a transition rule using fluent API.

        Args:
            event_type: Event type to match

        Returns:
            RuleBuilder for fluent API (from region.on())
        """
        return self._region.on(event_type)

    def set_initial(self, state_name: str):
        """Set the initial state for this region.

        Args:
            state_name: Name of the initial state
        """
        self._region.initial = state_name

    def build(self):
        """Build the region from the states and composites.

        Returns:
            The constructed Region instance
        """
        for composite_builder in self.composite_stack:
            built_composite = composite_builder.composite
            self._region[built_composite.name] = built_composite
        return self._region

    @property
    def region(self):
        """Get the region being built."""
        return self._region


class build_region:
    """
    Context manager to build a Region
    """
    def __init__(self, name: str):
        """Create a context manager to build a region

        Args:
            name (str): The name of the region to add
        """
        self.region_builder = RegionBuilder(name)

    def __enter__(self):
        return self.region_builder

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.region_builder.build()
        return False  # Propagate exceptions

    @property
    def region(self):
        """Get the built region."""
        return self.region_builder.region


class ChartBuilder:
    """Helper used by build_chart to build a StateChart with a context manager
    """

    def __init__(self, chart: _chart.StateChart):
        """Initialize chart builder.

        Args:
            chart: StateChart instance to populate
        """
        self.chart = chart
        self.region_stack = []

    def region(self, name: str):
        """Create a region builder for this chart.

        Args:
            name: Name of the region

        Returns:
            build_region context manager for the new region
        """
        region_builder = build_region(name)
        self.region_stack.append(region_builder)
        return region_builder

    def build(self):
        """Build the chart with all regions added.

        Returns:
            The populated StateChart instance
        """
        self.chart.regions = ModuleList(
            vals=[rb.region for rb in self.region_stack]
        )
        return self.chart


class build_chart:
    """
    Use to build a StateChart with a context manager
    Example:
    with build_chart() as chart:
        with chart.region('region1') as region:
            region.add_state(SomeState()) # adds the state
            with region.composite('composite1') as composite:
                with composite.region('subregion1') as subregion:
                    subregion.add_state(AnotherState())
            region.add_state(ThirdState())
    """
    def __init__(self, chart: _chart.StateChart):
        self.chart = ChartBuilder(chart)

    def __enter__(self):
        return self.chart

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.chart.build()
        return False  # Propagate exceptions
