from abc import ABC, abstractmethod
import typing


class PromptComponent(ABC):

    @abstractmethod
    def as_text(self) -> str:
        pass

    @abstractmethod
    def as_dict(self) -> typing.Dict:
        pass

    @staticmethod
    def structure(text: str, heading: str=None):

        if heading is None:
            return f'{text}'
        
        return f"""
        {heading}
        {text}
        """
