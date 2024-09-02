# 1st party
import json
import typing

# 3rd party
import pandas as pd

# local
from ._structs import Message
from . import Module, processf


class CSV2DF(Module):
    """Converts a Message with a CSV to a Dataframe
    """

    def __init__(self, sep: str=',', key: str='text'):
        """Module to convert a CSV to a DataFrame

        Args:
            sep (str, optional): The separator for the CSV. Defaults to ','.
            key (str, optional): The key for the data. Defaults to 'text'.
        """
        self.delim = sep
        self.key = key

    def forward(self, message: Message) -> pd.DataFrame:
        """The message to convert

        Args:
            message (Message): The message 

        Returns:
            pd.DataFrame: The CSV as a DataFrame
        """
        return pd.read_csv(
            message[self.key], sep=self.delim
        )


@processf
def json_to_dict(message: Message, key: str='text') -> pd.DataFrame:
    """Convert a message containing a JSON string to a dict

    Args:
        message (Message): The message to convert
        key (str, optional): The key for the value to convert. Defaults to 'text'.

    Returns:
        dict: The dictionary loaded from the value
    """
    return json.loads(
        message.data[key]
    )


@processf
def csv_to_df(message: Message, sep: str=',', key: str='text') -> pd.DataFrame:
    """Convert a message to df

    Args:
        message (Message): The message to convert
        sep (str, optional): The separator. Defaults to ','.
        key (str, optional): The key for the data. Defaults to 'text'.

    Returns:
        pd.DataFrame: the csv as a DataFrame
    """
    return pd.read_csv(
        message.data[key], sep=sep
    )


@processf
def kv_to_dict(message: Message, sep: str='::', key: str='text') -> typing.Dict:
    """Convert a set of key values to a dictionary

    Args:
        message (Message): The message to convert
        sep: The separator for the keys and values. Defaults to '::'.
        key (str, optional): The key. Defaults to 'text'.

    Returns:
        typing.Dict: 
    """
    text = message.data[key]
    lines = text.splitlines()
    result = {}
    for line in lines:
        r = line.split(sep)
        if len(r) != 2:
            continue
        key, value = r
        result[key] = value

    return result
