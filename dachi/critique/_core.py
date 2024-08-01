from abc import ABC, abstractmethod
import typing
import json
from ..converse import PromptModel, Message


from .._core import (
    Struct, Module, 
    escape_curly_braces, Data,
    render
)

# TODO: Add quotations


class Evaluation(Struct):

    data: typing.List[
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


class Textual(Criterion):

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
        for name, criterion in self.criteria().items():
            criteria.add(f"""
            # Criterion: {name}

            {criterion}
            """)
        return '\n\n'.join(criteria)

    def out_format_str(self) -> typing.Dict:

        out_format_dict = self.out_format()

        out_format = escape_curly_braces({
            name: escape_curly_braces(criterion)
            for name, criterion in out_format_dict.items()
        })
        
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

        # {self.name}

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

    def __init__(self, llm: PromptModel, criterion: Criterion):

        self.llm = llm
        self.criterion_view = HeaderView(criterion)
    
    def forward(self, y: Data, t: Data=None) -> Evaluation:
        
        instructions = self.criterion_view(
            y, t
        )
        result = self.llm(Message('System', instructions))
        return Evaluation(data=json.loads(result))


# class Evaluator(Module):

#     criterion: Criterion
    
#     VAR_NAMES = {'y', 't'}

#     @abstractmethod
#     def render_data(self, y: Data, t: Data) -> str:
#         pass

#     def forward(self, y: Data, t: Data) -> typing.Any:
        
#         criterion_str = self.criterion.render()

#         return f"""

#         {criterion_str}

#         # Data

#         {self.render_data}
#         """

# Move render data into the criterion
# I can have a decorator criterion to 
# add different types

#

# class Sample(Struct):

#     data: typing.Dict[str, Struct]

#     def render(self) -> str:

#         return escape_curly_braces(
#             self.data
#         )
    
#     @classmethod
#     def create(cls, **kwargs) -> 'Sample':

#         return Sample(
#             data=kwargs
#         )


# class Batch(Struct):

#     data: typing.List[typing.Dict[str, Struct]]

#     def render(self) -> str:
#         return escape_curly_braces(
#             self.data
#         )
    
#     @classmethod
#     def create(cls, **kwargs) -> 'Batch':

#         return Batch(
#             data=[
#                 dict(zip(kwargs,t))
#                 for t in zip(*kwargs.values())]
#         )

#     @classmethod
#     def from_samples(cls, samples: typing.List[Sample]) -> 'Batch':

#         return Batch(
#             data=[
#                 sample.data for sample in samples
#             ]
#         )



# class EvaluatorBase(Description, Module):

#     VAR_NAMES = {}

#     @abstractmethod
#     def out_format(self) -> typing.Dict:
#         pass

#     @classmethod
#     def var_name_str(cls) -> str:
#         return '\n'.join(
#             f'{k}: {v}' for k, v in cls.VAR_NAMES.items()
#         )

#     def out_format_str(self) -> str:
#         base_out = escape_curly_braces(
#             self.out_format()
#         )
#         return f"""
#         {{
#             0: {base_out},
#             ...
#             N: {base_out}
#         }}
#         """

#     @abstractmethod
#     def criteria(self) -> typing.List[Criterion]:
#         pass

#     def additional(self) -> typing.Dict:
#         return {}

#     def additional_str(self) -> str:
#         additional = self.additional()
#         if len(additional) == 0:
#             return 'None.'
#         out_str = ""
#         for k, v in additional.items():
#             out_str += f'{k}: {escape_curly_braces(v)}\n\n'
#         return out_str

#     def render(self) -> str:

#         # target var name = y
#         # output

#         rendered = """
#         Evaluate each sample from 1 to N where N is the number of
#         samples

#         # Variables

#         {var_names}

#         # Method

#         {criteria}

#         # Data
#         {data}

#         Output with this format.

#         {format}

#         # Additional

#         {additional}
#         """
#         return str_formatter(
#             rendered,
#             var_names=self.var_name_str(),
#             criteria=escape_curly_braces(self.criteria()),
#             format=escape_curly_braces(self.out_format_str()),
#             additional=escape_curly_braces(self.additional())
#         )
    
#     @abstractmethod
#     def forward(self, y: Data, t: Data) -> typing.Any:
#         pass



    # @pydantic.field_validator('how', mode='before')
    # def validate_names_types_data(cls, values):
    
    #     variables = set(get_str_variables(cls.how))
        
    #     if variables != cls.var_names:
    #         raise ValueError(
    #             "The description must have these variable "
    #             f"names {cls.var_names}"
    #         )

    #     return values
    # @classmethod
    # def var_name_str(cls) -> str:
    #     return '\n'.join(
    #         f'{k}: {v}' for k, v in cls.VAR_NAMES.items()
    #     )
    
    # def out_format_str(self) -> str:
    #     # TODO: Use the criteria

    #     out_format = escape_curly_braces({
    #         criterion.name: "<result>"
    #         for criterion in self.criteria

    #     })
        
    #     return f"""
    #     {{
    #         0: {out_format},
    #         ...
    #         N: {out_format}
    #     }}
    #     """

    # def criteria_str(self) -> str:
    #     # TODO: Use the criteria
    #     return '\n'.join(
    #         criterion.render(extra_info=False)
    #         for criterion in self.criteria
    #     )

    # def additional_str(self) -> str:

    #     return '\n'.join(
    #         criterion.render(criteria=False)
    #         for criterion in self.criteria
    #     )
    
    #     # additional = self.additional()
    #     # if len(additional) == 0:
    #     #     return 'None.'
    #     # out_str = ""
    #     # for k, v in additional.items():
    #     #     out_str += f'{k}: {escape_curly_braces(v)}\n\n'
    #     # return out_str

    # def render(self) -> str:
    #     # target var name = y
    #     # output

    #     rendered = """
    #     Evaluate each sample from 1 to N where N is the number of
    #     samples

    #     # Variables

    #     {var_names}

    #     # Method

    #     {criteria}

    #     # Data
    #     {data}

    #     Output with this format.

    #     {format}

    #     # Additional

    #     {additional}
    #     """
    #     return str_formatter(
    #         rendered,
    #         var_names=self.var_name_str(),
    #         criteria=self.criteria_str(),
    #         format=self.out_format_str(),
    #         additional=self.additional_str()
    #     )
    
    #     # return escape_curly_braces(self.out_format_str())

# """

# Evaluate each of the following criteria and output to a JSON as specified below.

# # Criterion 1: <Name>

# Evaluate on a scale of -1 to 1 if the output is semantically similar
# to the target

# -1: Disagree
# 0: Neither agree nor disagree
# 1: Agree

# Disagree: If ...
# Neither agree nor disagree: If ...
# Agree: ...


# # Output 

# {
#     0: {
#         'Name1': <Evaluation>,
    
#     },
#     ...
#     N: {

    
#     }
# }

# """

# class Rating(Criterion):
    
#     min_val: float = 0.0
#     max_val: float = 1.0
#     examples: typing.Optional[typing.Dict[float, typing.List[str]]] = None

#     @abstractmethod
#     def render(self, criteria: bool=True, extra_info: bool=True) -> str:
#         pass



# class Likert(Criterion):
    
#     start: int = 0
#     levels: typing.List[str]
#     examples: typing.Optional[typing.Dict[int, typing.List[str]]] = None

#     def extra_info(self) -> typing.Optional[str]:
#         return None

#     def render(self, criteria: bool=True, extra_info: bool=True) -> str:
#         pass


# class Diagnosis(Criterion):
    
#     requirements: str
#     criteria: typing.List[str]
    
#     examples: typing.Optional[typing.Dict[int, typing.List[str]]] = None

#     @abstractmethod
#     def render(self, criteria: bool=True, extra_info: bool=True) -> str:
#         pass




# AGREE_5 = [
#     'strongly disagree',
#     'disagree',
#     'neither agree or disagree',
#     'agree',
#     'strongly agree'
# ]


# QUALITY_5 = [
#     'very bad',
#     'bad',
#     'average',
#     'good',
#     'very good'
# ]



# class CompositeEvaluator(EvaluatorBase):

#     def __init__(self, evaluators: typing.List[Evaluator]):
        
#         self.evaluators = evaluators

#     def out_format(self) -> typing.Dict:
        
#         format = {}
#         for evaluator in self.evaluators:
#             format.update(
#                 evaluator.out_format()
#             )
#         return format

#     def criteria(self) -> typing.List[Criterion]:

#         pass
#         # criteria = {}
#         # for evaluator in self.evaluators:
#         #     criteria.update(
#         #         evaluator.criteria()
#         #     )
#         # return criteria

#     def additional(self) -> typing.List[typing.List[str]]:

#         additional = []
#         for evaluator in self.evaluators:
#             additional.append(
#                 *evaluator.additional()
#             )
#         return additional

#     def render(self) -> str:

#         return escape_curly_braces(
#             self.out_format_str()
#         )
    
#     def forward(self, y: Data, t: Data=None) -> typing.Any:
        
#         result = self.render()
#         variables = get_str_variables(result)
#         if 't' in variables and t is not None:
#             return result.format(
#                 y=y, t=t
#             )
#         return result.format(y=y)


# class Supervised(Evaluator):

#     def var_name_str(cls) -> typing.Set[str]:
#         return set(['y', 't'])

#     def out_format(self) -> typing.Dict:
        
#         return {self.name: '<Evaluation>'}

#     def criteria(self) -> typing.List[Criterion]:
#         pass
#         # return {
#         #     self.name: f'{self.name}: how well the output matches the target'
#         #                f'according to: {self.how}'
#         # }

#     def forward(self, y: Data, t: Data) -> typing.Any:
        
#         rendered = self.render()
#         data = y.merge(t)
#         return rendered.format(data=data.render())


# class Quality(Evaluator):

#     def out_format(self) -> typing.Dict:
        
#         return {self.name: '<Evaluation>'}
    
#     def criteria(self) -> typing.Dict:
#         return {
#             self.name: f'Evaluate the input y according to '
#                        f'this regularzation. {self.regularization}'
#         }

#     def forward(self, y: Data, t: Data=None) -> typing.Any:
        
#         rendered = self.render()
#         data = y.merge(t)
#         return rendered.format(data=data.render())


# class Style(Evaluator):

#     out_examples: typing.List[Struct]

#     def additional(self) -> typing.List[typing.List[str]]:
#         return [[
#             escape_curly_braces(example)
#             for example in self.examples
#         ]]

#     def out_format(self) -> typing.Dict:
        
#         return {self.name: '<Evaluation>'}

#     def criteria(self) -> typing.List[Criterion]:
#         pass
#         # return {
#         #     self.name: f'{self.name}: how well the output matches the target'
#         #                f'according to: {self.how}'
#         # }

#     def forward(self, y: Data, t: Data=None) -> typing.Any:
        
#         rendered = self.render()
#         data = y.merge(t)
#         return rendered.format(data=data.render())

