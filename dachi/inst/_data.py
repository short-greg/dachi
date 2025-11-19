from abc import abstractmethod, ABC
import pydantic
import typing as t
import pandas as pd
from dachi.core import Renderable, render


class Description(
    pydantic.BaseModel, Renderable, ABC
):
    """Provide context in the prompt template
    """
    name: str = pydantic.Field(description='The name of the description.')

    @abstractmethod
    def render(self) -> str:
        pass


class Ref(pydantic.BaseModel, Renderable):
    """Reference to another description.
    Useful when one only wants to include the 
    name of a description in part of the prompt
    """
    desc: Description

    @property
    def name(self) -> str:
        """Get the name of the ref

        Returns:
            str: The name of the ref
        """
        return self.desc.name

    def render(self) -> str:
        """Generate the text rendering of the ref

        Returns:
            str: The name for the ref
        """
        return self.desc.name


class Record(Renderable):
    """Use to create a pairwise object
    """

    def __init__(self, indexed: bool=False, **kwargs):
        """Create a pairwise object

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        super().__init__()
        self._data = pd.DataFrame(kwargs)
        self.indexed = indexed

    def extend(self, **kwargs) -> t.Self:
        """Extend the pairwise object with new items

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        self._data = pd.concat(
            [self._data, pd.DataFrame(kwargs)],
            ignore_index=True
        )

    def append(self, **kwargs) -> t.Self:
        """Append the pairwise object with new items

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        kwargs = {k: [v] for k, v in kwargs.items()}
        self._data = pd.concat(
            [self._data, pd.DataFrame(kwargs)],
            ignore_index=True
        )
    
    def join(self, **kwargs) -> 'Record':
        """Join the pairwise object with new items

        Args:
            items (typing.List[T]): The items to create the pairwise object from
        """
        data = self._data.copy()
        data = data.assign(**kwargs)
        record = Record()
        record._data = data
        return record
    
    def clear(self):
        """Reset the record
        """
        self._data = pd.DataFrame()
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the dataframe for the pairwise object

        Returns:
            pd.DataFrame: The dataframe for the pairwise object
        """
        return pd.DataFrame(self._items)
    
    def __getitem__(self, key: str):

        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if the record has the column specified by key

        Args:
            key (str): The key to check

        Returns:
            bool: Whether contained
        """
        return key in self._data.columns.values
    
    def __len__(self) -> int:
        """Get the length of the pairwise object

        Returns:
            int: The length of the pairwise object
        """
        return len(self._data.index)
    
    def render(self) -> str:
        """Render the pairwise object

        Returns:
            str: The rendered string
        """
        return render(self._data.to_dict(orient='records'))

    def top(self, field: str, largest: bool=True):

        if field not in self._data.columns:
            raise KeyError(f"Field '{field}' not found in the dataframe.")

        if largest:
            idx = self._data[field].idxmax()
        else:
            idx = self._data[field].idxmin()

        return self._data.loc[idx]
