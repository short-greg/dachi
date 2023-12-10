import typing

from dachi.behavior import Server


class IOHandler:
    
    def __init__(self, server: Server, bot_name: str) -> None:
        
        self.server = server
        self.callback = None
        self.input_name = None
        self._bot_name = bot_name

    def connect_ui(self, callback):

        self.callback = callback
    
    def register_input(self, name: str):
        self.input_name = name

    def post_bot_message(self, bot_message: str) -> bool:
        if not self.callback:
            return False
        self.callback(self._bot_name, bot_message)
        return True

    def post_user_message(self, user_message: str) -> bool:

        if not self.input_name:
            return False
        
        self.server.shared[self.input_name] = user_message
        return True
