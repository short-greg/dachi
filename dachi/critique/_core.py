from abc import ABC, abstractmethod
import typing
import json
from .._core import (
    Struct, Module,
    escape_curly_braces, Data,
    render, AIModel, TextMessage
)
# TODO: Add quotations


class Evaluation(Struct):

    values: typing.List[
        typing.Dict[str, typing.Any]
    ]


class Criterion(Struct, ABC):

    name: str

    @abstractmethod
    def criteria(self) -> typing.Dict[str, str]:
        pass

    @abstractmethod
    def out_format(self) -> typing.Dict:
        pass


class TextualCriterion(Criterion):

    desc: str

    def out_format(self) -> typing.Dict:
        
        return {
            self.name: "<Result>"
        }

    def criteria(self) -> typing.Dict[str, str]:
        
        return {
            self.name: self.desc
        }


class CriterionViewBase(Module, ABC):
    """Module responsible for rendering the criteria
    """
    def __init__(self, criterion: Criterion):
        self.criterion = criterion

    @abstractmethod
    def forward(
        self, x: Data, t: Data
    ) -> str:
        pass


class HeaderView(CriterionViewBase):

    def criteria_str(self) -> typing.Dict[str, str]:
        
        criteria = []
        for name, criterion in self.criterion.criteria().items():
            criteria.append(f"""
            # Criterion: {name}

            {criterion}
            """)
        return '\n\n'.join(criteria)

    def out_format_str(self) -> typing.Dict:

        out_format_dict = self.criterion.out_format()

        pre = {
            name: criterion
            for name, criterion in out_format_dict.items()
        }
        out_format = escape_curly_braces(pre)
        
        return f"""
        {{
            0: {out_format},
            ...
            N: {out_format}
        }}
        """
    
    def data_str(self, y: Data, t: Data) -> str:
        if not isinstance(y, typing.List):
            y = [y]
        if not isinstance(t, typing.List):
            t = [t]

        assert len(y) == len(t)
        result = []
        for t_i, y_i in zip(y, t):
            result.append({
                "y": render(y_i),
                "t": render(t_i)
            })
        return escape_curly_braces(result)
    
    def forward(
        self, y: Data, t: Data
    ) -> str:

        return f"""
        Evaluate the outputs according to the criteria below and
        output as a JSON

        Output: y
        Target: t

        # {self.criterion.name}

        {self.criteria_str()}

        # Out Format

        {self.out_format_str()}

        # Data

        {self.data_str(y, t)}

        """


class Critic(Module, ABC):

    @abstractmethod
    def forward(self, x: Data, t: Data=None) -> typing.Any:
        pass


class LLMCritic(Module, ABC):

    def __init__(self, ai_model: AIModel, criterion: Criterion):

        self.ai_model = ai_model
        self.criterion_view = HeaderView(criterion)
    
    def forward(self, y: Data, t: Data=None) -> Evaluation:
        
        instructions = self.criterion_view(
            y, t
        )
        result = self.ai_model(TextMessage('system', instructions))
        return Evaluation(data=json.loads(result))
