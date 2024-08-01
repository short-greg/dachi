from ._structs import Message
from .._core import Module, processf
import pandas as pd
import json
import io
import string


class CSV2DF(Module):

    def __init__(self, delim: str=','):

        self.delim = delim

    def forward(self, message: Message) -> pd.DataFrame:
        
        return pd.read_csv(
            message.content, sep=self.delim
        )


@processf
def json_to_dict(message: Message, text_field: str='text') -> pd.DataFrame:
    
    return json.loads(
        message.content[text_field]
    )


@processf
def csv_to_df(message: Message, sep: str=',', text_field: str='text') -> pd.DataFrame:

    return pd.read_csv(
        message.content[text_field], sep=sep
    )


@processf
def kv_to_dict(message: Message, sep: str='::', text_field: str='text') -> pd.DataFrame:

    text = message.content[text_field]
    lines = text.splitlines()
    result = {}
    for line in lines:
        r = line.split(sep)
        if len(r) != 2:
            continue
        key, value = r
        result[key] = value

    return result
