
import tkinter as tk
from .agent import StoryWriter, AgentStatus
from ..tools.ui import ChatbotInterface
import time


class StoryWriterApp:

    def __init__(self):

        self._running = False
        self._ui = ChatbotInterface()
        self._agent = StoryWriter(self._ui, interval=1/60)
        self._interval = 1 / 60

    def run(self):
        self._running = True
        agent_stopped = False
        while True:

            while True:
                try:
                    self._ui.update_idletasks()
                    self._ui.update()
                    if not agent_stopped:
                        status = self._agent.act()
                        if status == AgentStatus.STOPPED:
                            self._agent.reset()
                except tk.TclError:
                    break
                time.sleep(1 / 60)
