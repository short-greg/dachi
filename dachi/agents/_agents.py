from ..behavior import Sango
from ..graph import Tako


class Agent(object):

    def __init__(self, tako: Tako):

        super().__init__()


    def exec(self):
        pass


class TakoAgent(Agent):

    def __init__(self, tako: Tako):

        super().__init__()
        self.tako = tako

    def run(self):
        self.tako()

    def step(self):
        self.tako()


class SangoAgent(Agent):

    def __init__(self, sango: Sango):

        super().__init__()
        self.sango = sango
        self.server = Server()

    def run(self):
        self.tako()

    def step(self):

        self.sango.tick()


