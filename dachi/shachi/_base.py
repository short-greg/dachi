from abc import ABC
from uuid import uuid4


# How to deal with waiting
# How to continue running

class Behavior(ABC):
    
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._id = str(uuid4())

    @property
    def id(self):
        return self._id
