from abc import ABC, abstractmethod
import typing


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
    
    async def async_query(self, data) -> typing.Dict:
        """Run this query for asynchronous operations
        The default behavior is simply to call the query

        Args:
            data: Data to pass to the API

        Returns:
            typing.Any: 
        """
        return self.query(data)
    
    async def async_stream_query(self, data) -> typing.AsyncIterator[typing.Dict]:
        """Run this query for asynchronous streaming operations
        The default behavior is simply to call the query

        Args:
            data: The data to pass to the API

        Yields:
            typing.Dict: The data returned from the API
        """
        result = self.query(data)
        result['delta'] = None
        yield result
