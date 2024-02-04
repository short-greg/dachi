import tkinter as tk
from tkinter import scrolledtext
from tkinter import font as tkfont
import typing
from dachi.comm import UI


class ChatHistory(scrolledtext.ScrolledText):

    def __init__(self, frame):
        larger_font = tkfont.Font(family="Helvetica", size=14)
        super().__init__(
            frame, wrap=tk.WORD, font=larger_font
        )
        self.history = []
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def append(self, speaker, message):
        self.history.append({'speaker': speaker, 'message': message})
        self.insert(tk.END, f'{speaker}: {message}' + "\n")
        self.see(tk.END)

    def __getitem__(self, index):

        return self.history[index]


class ChatbotInterface(tk.Tk, UI):

    def __init__(self):
        super().__init__()
        # self._agent = agent

        self.title("Chatbot")
        self.geometry("800x600")  # Initial size of the window
        self.minsize(800, 600)    # Minimum size of the window

        # Main Frame
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # Chat History Pane
        self.chat_history = ChatHistory(self.main_frame)

        # Interactive Features Pane
        self.interactive_pane = tk.Frame(self.main_frame, bg='lightgray')
        self.interactive_pane.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(self.interactive_pane, text="Quick Reply 1").pack(pady=5)
        tk.Button(self.interactive_pane, text="Quick Reply 2").pack(pady=5)

        # Bottom Bar
        self.bottom_bar = tk.Frame(self)
        self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Text Input Field (Multiline)
        self.input_field = tk.Text(self.bottom_bar, height=3)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Shift-Return>", self.send_message)

        # Send Button
        self.send_button = tk.Button(self.bottom_bar, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
        self.waiting_for_response = False
        self.toggle_input_state(False)
        self._input_state = False
        # self._agent.io.connect_ui(self.bot_response)
        self.bot_message_buffer = []
        self._requests: typing.List[typing.Callable[[str], None]] = []

    def update(self) -> None:
        self.toggle_input_state(self._input_state)
        for speaker, message in self.bot_message_buffer:
            self.bot_response(speaker, message)
        self.bot_message_buffer.clear()
        return super().update()

    def request_message(self, callback: typing.Callable[[str], None]):

        self._requests.append(callback)
        self._input_state = True

    def send_message(self, event=None):
        if self.waiting_for_response:
            return  # Ignore additional input while waiting

        message = self.input_field.get("1.0", tk.END).strip()
        if message:
            request = self._requests[0]
            self._requests.pop(0)
            if len(self._requests) == 0:
                self.toggle_input_state(False)
            self.chat_history.append('You', message)
            self.input_field.delete("1.0", tk.END)
            self.waiting_for_response = True
            # self.toggle_input_state(False)
            self._input_state = False
            # Simulate bot response
            message = self.chat_history[-1]
            
            self.after(1, request, message['message'])
            # self.after(2000, self.bot_response)  # Replace with actual bot logic

    def bot_response(self, speaker, message):
        # Remove thinking dots
        self.thinking = False
        # self.chat_history.delete(self.thinking_message_id, tk.END)
        # Simulated bot response
        self.chat_history.append(speaker, message)
        self.waiting_for_response = False
        # self.toggle_input_state(True)
        self.thinking_dots = ""
    
    def post_message(self, speaker, message: str) -> bool:
        self.bot_message_buffer.append((speaker, message))
        # self.bot_response(speaker, message)
        return True
    
    def toggle_input_state(self, on: bool=True):
        self.send_button.config(state=tk.NORMAL if on else tk.DISABLED)
