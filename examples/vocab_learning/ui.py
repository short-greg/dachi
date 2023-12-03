from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from backend import ChatBackend


class UI(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        self.chat_layout = BoxLayout(orientation='horizontal')
        self.add_widget(self.chat_layout)

        self.chat_history = BoxLayout()  # This will contain the chat bubbles
        self.chat_layout.add_widget(self.chat_history)

        self.interactive_pane = BoxLayout()  # This will contain interactive elements
        self.chat_layout.add_widget(self.interactive_pane)

        self.input_field = TextInput(multiline=False)
        self.send_button = Button(text='Send')
        self.bottom_bar = BoxLayout(size_hint_y=None, height=50)
        self.bottom_bar.add_widget(self.input_field)
        self.bottom_bar.add_widget(self.send_button)

        self.add_widget(self.bottom_bar)
        self.chat_backend = ChatBackend(self.update_chat_history)

        # Bind events like button click and text input here

    def on_send_pressed(self, instance):
        message = self.input_field.text
        self.chat_backend.send_message(message)
        self.input_field.text = ''  # Clear the input field after sending

    def update_chat_history(self, message):
        # Method to update chat history in the UI
        # You'll need to implement this to show messages
        pass
