from abc import ABC, abstractmethod
import typing


class UI(ABC):
    """Defines an interface for creating a UI
    """

    @abstractmethod
    def request_message(self, callback: typing.Callable[[str], None]):
        pass

    @abstractmethod
    def post_message(self, speaker, message: str) -> bool:
        pass
