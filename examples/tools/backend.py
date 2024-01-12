import typing
import time


class ChatBackend:

    def post(self, message: str, chat_history: typing.List[str], callback) -> str:

        time.sleep(3)
        callback(speaker='Bot', message='My response')
