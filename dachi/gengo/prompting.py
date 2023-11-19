import typing
from dataclasses import dataclass, Field


@dataclass
class Arg:
    
    name: str
    description: str = Field("")
    

class Prompt(object):

    def __init__(self, args: typing.List[typing.Union[Arg, str]], text: str):
        
        super().__init__()
        self._args = {}
        for arg in args:
            if isinstance(arg, str):
                self._args[arg] = Arg(arg)
            else:
                self._args[arg.name] = arg
        self._text = text

    def format(self, **kwargs):

        input_names = set(kwargs.keys())
        difference = input_names - set(self._args)
        if len(difference) != 0:
            raise ValueError(f'Input has keys that are not arguments to the prompt')
        inputs = {}
        for k, v in self._args.items():
            if k in kwargs:
                inputs[k] = v
            else:
                inputs[k] = "{{}}"
        return Prompt(
            self._text.format(**inputs)
        )
