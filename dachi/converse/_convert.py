from .._core._structs import Message
from .._core import Module, processf
import pandas as pd
import json


class CSV2DF(Module):

    def __init__(self, delim: str=',', key: str='text'):

        self.delim = delim
        self.key = key

    def forward(self, message: Message) -> pd.DataFrame:
        
        return pd.read_csv(
            message.data[self.key], sep=self.delim
        )


@processf
def json_to_dict(message: Message, key: str='text') -> pd.DataFrame:
    
    return json.loads(
        message.data[key]
    )


@processf
def csv_to_df(message: Message, sep: str=',', key: str='text') -> pd.DataFrame:

    return pd.read_csv(
        message.data[key], sep=sep
    )


@processf
def kv_to_dict(message: Message, sep: str='::', key: str='text') -> pd.DataFrame:

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
