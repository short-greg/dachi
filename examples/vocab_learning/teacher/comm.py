from abc import ABC, abstractmethod
import typing


class UIInterface(ABC):

    @abstractmethod
    def request_message(self, callback: typing.Callable[[str], None]):
        pass
