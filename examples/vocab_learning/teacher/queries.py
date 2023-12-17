import typing
import threading

from openai import OpenAI

from typing import Dict
from dachi.comm import Query, Server

class LLMQuery(Query):

    def __init__(self, server: Server):
        self.client = OpenAI()
        self.server = server
        
    def query(
        self, contents: Dict, server: Server,
        on_post: typing.Union[str, typing.Callable]= None, 
        on_response: typing.Union[str, typing.Callable]=None
    ):
        self.callback(on_post, server)
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=contents['prompt']
        )
        self.callback(on_response, server, response)

    def post(self, contents: Dict, server: Server, on_post: typing.Union[str, typing.Callable]= None, on_response: typing.Union[str, typing.Callable]=None):
        
        thread = threading.Thread(self.query, server, args=[contents, on_post, on_response])
        thread.run()
        # threading.Thread()

