from abc import ABC, abstractmethod
import typing
import asyncio


class APIAdapter(ABC):
    """APIAdapter allows one to adapt various WebAPI or otehr
    API for a consistent interface
    """

    @property
    @abstractmethod
    def output_schema(self) -> typing.Dict:
        pass

    @property
    @abstractmethod
    def input_schema(self) -> typing.Dict:
        pass

    @abstractmethod
    def query(self, data) -> typing.Dict:
        """Run a standard query to the API

        Args:
            data : Data to pass to the API

        Returns:
            typing.Dict: The result of the API call
        """
        pass

    def stream_query(self, data) -> typing.Iterator[typing.Dict]:
        """API that allows for streaming the response

        Args:
            data: Data to pass to the API

        Returns:
            typing.Iterator: Data representing the streamed response
            Uses 'delta' for the difference. Since the default
            behavior doesn't truly stream. This must be overridden 

        Yields:
            typing.Dict: The data
        """
        result = self.query(data)
        result['delta'] = None
        yield result
    
    async def async_query(self, data, **kwarg_override) -> typing.Dict:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.query(data, **kwarg_override)
    
    async def bulk_async_query(self, data, **kwarg_override) -> typing.List[typing.Dict]:
        """

        Args:
            data (_type_): 

        Returns:
            typing.List[typing.Dict]: 
        """
        tasks = []
        async with asyncio.TaskGroup() as tg:

            for data_i in data:
                tasks.append(
                    tg.create_task(self.async_query(data_i, **kwarg_override))
                )
        return tuple(
            task.result() for task in tasks
        )
    
    async def async_stream_query(self, data, **kwarg_override) -> typing.AsyncIterator[typing.Dict]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            data: The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.query(data, **kwarg_override)
        result['delta'] = None
        yield result

    async def _collect_results(generator, index, results, queue):
        async for item in generator:
            results[index] = item
            await queue.put(results[:])  # Put a copy of the current results
        results[index] = None  # Mark this generator as completed

    async def bulk_async_stream_query(self, data, **kwarg_override) -> typing.AsyncIterator[typing.List[typing.Dict]]:
        """

        Args:
            data (_type_): 

        Returns:
            typing.List[typing.Dict]: 
        """
        results = [None] * len(data)
        queue = asyncio.Queue()

        async with asyncio.TaskGroup() as tg:
            for index, data_i in enumerate(data):
                tg.create_task(self._collect_results(
                    self.async_stream_query(data_i, **kwarg_override), index, results, queue)
                )

        active_generators = len(data)
        while active_generators > 0:
            current_results = await queue.get()
            yield current_results
            active_generators = sum(result is not None for result in current_results)
