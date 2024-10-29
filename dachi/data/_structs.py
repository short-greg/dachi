# 1st party
import typing

# 3rd party
import pydantic

# local
from ..utils._utils import (
    generic_class
)
from ..utils._utils import generic_class
from .._core._ai import Message


S = typing.TypeVar('S', bound=pydantic.BaseModel)


class Media:
    """Use to store media
    """
    descr: str
    data: str


class DataList(pydantic.BaseModel, typing.Generic[S]):
    """Create a list of data that inherit from pydantic.BaseModel
    """
    data: typing.List[S]

    def __getitem__(self, key) -> typing.Any:
        """Get a value in the data list

        Args:
            key: The index for the data

        Returns:
            typing.Any: The value at that index
        """
        return self.data[key]
    
    def __setitem__(self, key: typing.Optional[int], value: S) -> typing.Any:
        """Set a value in the 

        Args:
            key (str): The key for the value to set
            value : The value to set

        Returns:
            S: the value that was set
        """
        if key is None:
            self.data.append(value)
        else:
            self.data[key] = value
        return value
    
    @classmethod
    def load_records(cls, records: typing.List[typing.Dict]) -> 'DataList[S]':
        """Load the struct list from records

        Args:
            records (typing.List[typing.Dict]): The list of records to load

        Returns:
            StructList[S]: The list of structs
        """
        structs = []
        struct_cls: typing.Type[pydantic.BaseModel] = generic_class(S)
        for record in records:
            structs.append(struct_cls.load(record))
        return DataList[S](
            structs=structs
        )


class MediaMessage(Message):
    """A message that contains media such as an image
    """

    def __init__(self, source: str, media: typing.List[Media]):
        """Create a media message with a source and a media

        Args:
            source (str): The source for the message
            media (typing.List[Media]): The media to use
        """
        super().__init__(
            source=source,
            data={
                'media': media
            }
        )

    def render(self) -> str:
        """Render the media message

        Returns:
            str: The rendered message
        """
        return f'{self.source}: Media [{self.media}]'
