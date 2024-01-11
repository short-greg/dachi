from abc import ABC, abstractmethod
import typing


class UI(ABC):

    @abstractmethod
    def request_message(self, callback: typing.Callable[[str], None]):
        pass

    @abstractmethod
    def post_message(self, speaker, message: str):
        pass
