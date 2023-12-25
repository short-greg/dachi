import typing
from dataclasses import dataclass, Field

from .base import PromptComponent


@dataclass
class Arg:
    
    name: str
    description: str = Field("")


class Prompt(PromptComponent):

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
    
    def as_text(self, heading: str=None) -> str:

        return self.structure(self._text, heading)

    def as_dict(self) -> str:

        return {
            "args": self._args,
            "text": self._text
        }


class Completion(PromptComponent):
    
    def __init__(self, prompt: Prompt, response: str):

        self.prompt = prompt
        self.response = response

    def as_text(
        self, 
        heading: str=None,
        prompt_heading: str="===Prompt===", 
        response_heading: str="===Response==="
    ) -> str:
        """
        Returns:
            typing.Dict: The completion object as a dict
        """
        body = f"""
        {self.structure(self.prompt.as_text(), prompt_heading)}
        {self.structure(self.response, response_heading)}
        """
        return self.structure(
            body, heading
        )
    
    def as_dict(self) -> typing.Dict:
        """
        Returns:
            typing.Dict: The completion object as a dict
        """
        return {
            "prompt": self.prompt,
            "response": self.response
        }
