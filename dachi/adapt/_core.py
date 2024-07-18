from abc import ABC, abstractmethod
import typing


class APIAdapter(ABC):
    """
    """

    @abstractmethod
    def query(self, data) -> typing.Dict:
        pass

    def stream_query(self, data) -> typing.Iterator[typing.Dict]:
        """

        Args:
            data: 

        Returns:
            typing.Iterator: 

        Yields:
            Iterator[typing.Iterator]: 
        """
        result = self.query(data)
        result['delta'] = None
        yield result
    
    async def async_query(self, data) -> typing.Dict:
        """

        Args:
            data: 

        Returns:
            typing.Any: 
        """
        return self.query(data)
    
    async def async_stream_query(self, data) -> typing.AsyncIterator[typing.Dict]:
        """

        Args:
            data: 

        Returns:
            typing.AsyncIterator: 

        Yields:
            Iterator[typing.AsyncIterator]: 
        """
        result = self.query(data)
        result['delta'] = None
        yield result
