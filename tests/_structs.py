from dachi.instruct._data import Description
from pydantic import BaseModel, Field
from dachi.utils import str_formatter


class SimpleStruct(BaseModel):

    x: str


class SimpleStruct2(BaseModel):

    x: str
    y: int


class NestedStruct(BaseModel):

    simple: SimpleStruct


class Role(Description):

    duty: str = Field(description='The duty of the role')

    def render(self) -> str:

        return f"""
        # Role {self.name}

        {self.duty}
        """
    
    def update(self, **kwargs) -> Description:
        return Role(name=self.name, duty=str_formatter(self.duty, **kwargs))
