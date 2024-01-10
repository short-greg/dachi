from abc import ABC, abstractmethod
import typing


class UIInterface(ABC):

    @abstractmethod
    def request_message(self, callback: typing.Callable[[str], None]):
        pass


# class IOHandler:
    
#     def __init__(self, interface: ChatbotInterface, bot_name: str) -> None:
        
#         self.chatbot_interface = interface
#         self.backend_callback = None
#         # self.input_name = None
#         self._bot_name = bot_name

#     def connect_ui(self, callback):

#         self.ui_callback = callback

#     def post_bot_message(self, bot_message: str, backend_callback) -> bool:

#         if not self.ui_callback:
#             return False
#         self.ui_callback(self._bot_name, bot_message)
#         self.backend_callback = backend_callback
#         return True

#     def post_user_message(self, user_message: str) -> bool:

#         if self.input_name is None:
#             return False
#         if isinstance(self.backend_callback, str):
#             self.server.shared[self.backend_callback] = user_message
#         else:
#             self.backend_callback(user_message)
#         self.backend_callback = None
#         return True
