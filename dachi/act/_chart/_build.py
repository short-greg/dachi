from . import _base as base, _chart, _composite, _region, _state

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
        self.composite = _composite.Composite(name=name)
        self.current_region = None
        self.region_stack = []

    def region(self, name: str):
        region_builder = build_region(name)
        self.region_stack.append(region_builder)
        return region_builder
    
    def build(self):
        # Build the composite from regions
        for region_builder in self.region_stack:
            self.composite.add_region(region_builder.region)
        return self.composite


class build_composite:
    """
    Context manager to build a Composite
    """

    def __init__(self, name: str):
        """
        Build a composite state from the states
        """
        self.composite = CompositeBuilder(name)
        self.region_stack = []

    def __enter__(self):
        return self.composite
    
    def region(self, name: str):
        self.region_stack.append(build_region(name))
        return self.region_stack[-1]

    def __exit__(self, exc_type, exc_value, traceback):
        self.composite.build()
        # TODO must not swallow exceptions
        pass
                      

class RegionBuilder:

    def __init__(self, name: str):
        """Helper used by build_region to build a Region with a context manager

        Args:
            name (str): The name of the region to add
        """
        self.region = _region.Region(name=name)
        self.composite_stack = []

    def add_state(self, state: _state.State):
        self.region.add_state(state)

    def composite(self, name: str):
        """
        Create a composite in the region
        Args:
            name (str): The name of the composite to add
        """
        composite_builder = build_composite(name)
        self.composite_stack.append(composite_builder)
        return composite_builder
    
    def build(self):
        """
        Build the region from the states
        """
        for composite_builder in self.composite_stack:
            self.region.add_composite(composite_builder.build())
        return self.region


class build_region:
    """
    Context manager to build a Region
    """
    def __init__(self, name: str):
        """Create a context manager to build a region

        Args:
            name (str): The name of the region to add
        """
        self.region = RegionBuilder(name)

    def __enter__(self):
        return self.region

    def __exit__(self, exc_type, exc_value, traceback):
        self.region.build()


class ChartBuilder:
    """Helper used by build_chart to build a StateChart with a context manager
    """

    def __init__(self, chart: _chart.StateChart):
        """

        Args:
            chart : 
        """
        self.chart = chart
        self.current_region = None
        self.region_stack = []
    
    def region(self, name: str):
        region_builder = build_region(name)
        self.region_stack.append(region_builder)
        return region_builder
    
    def build(self):
        # Build the chart from regions
        for region_builder in self.region_stack:
            self.chart.add_region(region_builder.region)
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
        self.chart.build()
        # TODO must not swallow exceptions
        pass
