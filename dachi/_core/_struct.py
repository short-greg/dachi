

# class ValidateStrMixin:

#     @pydantic.field_validator('*', mode='before')
#     def convert_to_string_template(cls, v, info: pydantic.ValidationInfo):
    
#         outer_type = cls.model_fields[info.field_name].annotation
#         if (inspect.isclass(outer_type) and issubclass(outer_type, Str)) and not isinstance(v, Str) and not isinstance(v, typing.Dict):
#             return Str(text=v)
#         return v


# class Str(pydantic.BaseModel, TextMixin):

#     text: str
#     vars: typing.List[str] = Field(default_factory=list)

#     def forward(self, **kwargs):

#         format = {}
#         remaining_vars = []
#         for var in self.vars:
#             if var in kwargs:
#                 format[var] = to_text(kwargs[var])
#             else:
#                 format[var] = '{' + var + '}' 
#                 remaining_vars.append(var)
#         return Str(
#             text=self.text.format(**format),
#             vars=remaining_vars
#         )
    
#     def to_text(self):

#         return self.text
    
#     def __call__(self, **kwargs):
#         return self.forward(**kwargs)
