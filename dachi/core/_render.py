# 1st party
import inspect
from . import (
    Renderable, TemplateField,
)
from ..utils import escape_curly_braces, unescape_curly_braces, is_primitive
import typing
from typing import Self, get_type_hints
import json

# 3rd party
import pydantic


def model_template(model_cls: typing.Type[pydantic.BaseModel]) -> str:
    """Get the template for a pydantic.BaseModel

    Args:
        model_cls (typing.Type[pydantic.BaseModel]): The model to retrieve for

    Returns:
        str: The model template string
    """
    template = {}
    for name, field_type in get_type_hints(model_cls).items():
        
        if inspect.isclass(field_type) and issubclass(field_type, pydantic.BaseModel):
            template[name] = model_template(field_type)
        else:
            template[name] = {
                "is_required": model_cls.model_fields[name].is_required(),
                "type": field_type
            }
    return template


def model_to_text(model: pydantic.BaseModel, escape: bool=False) -> str:
    """Dump the struct to a string

    Returns:
        str: The string
    """
    if escape:  
        return escape_curly_braces(model.model_dump())
    return model.model_dump_json()


def model_from_text(model_cls: typing.Type[pydantic.BaseModel], data: str, escaped: bool=False) -> Self:
    """Load the struct from a string

    Args:
        data (str): The data for the struct

    Returns:
        Self: The loaded struct
    """
    if escaped:
        data = unescape_curly_braces(data)
    return model_cls(**json.loads(data))


def struct_template(model: pydantic.BaseModel) -> typing.Dict:
    """Get the template for the Struct

    Returns:
        typing.Dict: The template 
    """
    template = {}
    
    base_template = model_template(model)
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        if isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel):

            template[field_name] = struct_template(field_type)
        else:

            if 'is_required' in base_template[field_name]:
                is_required = base_template[field_name]['is_required']
            else:
                is_required = True
            template[field_name] = TemplateField(
                type_=field.annotation,
                description=field.description,
                default=field.default if field.default is not None else None,
                is_required=is_required
            )

    return template


def render(
    x: typing.Any, escape_braces: bool=True, 
    template_render: typing.Optional[typing.Callable[[TemplateField], str]]=None
) -> typing.Union[str, typing.List[str]]:
    """Convert an input to text. Will use the text for a cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    if isinstance(x, TemplateField):
        if template_render is not None:
            x = template_render(x)
        else: 
            x = x.render()

    if isinstance(x, Renderable):
        return x.render()

    elif isinstance(x, pydantic.BaseModel):
        return model_to_text(x, escape_braces)
    elif is_primitive(x):
        return str(x)
    elif isinstance(x, typing.Dict):
        items = {}
        for k, v in x.items():
            if isinstance(v, str):
                v = f'"{v}"'
            else:
                v = render(v, escape_braces)
            items[k] = v    
        items = ', '.join(
            f'"{k}": {v}' 
            for k, v in items.items()
        )

        if escape_braces:
            return f"{{{{{items}}}}}"
        else:
            return f'{{{items}}}'
    elif isinstance(x, typing.List):

        items = []
        for v in x:
            if isinstance(v, str):
                v = f'"{v}"'
            else:
                v = render(v, escape_braces)
            items.append(v)

        return '[{}]'.format(', '.join(render(v) for v in items))
    elif isinstance(x, Renderable):
        return x.render()
    return str(x)


def is_renderable(obj: typing.Any) -> bool:
    """Return whether an object is renderable

    Args:
        obj (typing.Any): The object to check

    Returns:
        bool: whether the object is renderable
    """

    return (
        isinstance(obj, Renderable)
        or is_primitive(obj)
        or isinstance(obj, list)
        or isinstance(obj, dict)
        or isinstance(obj, pydantic.BaseModel)
    )


def render_multi(xs: typing.Iterable[typing.Any]) -> typing.List[str]:
    """Convert an input to text. Will use the text for an cue,
    the render() method for a description and convert any other value to
    text with str()

    Args:
        value (X): The input

    Returns:
        str: The resulting text
    """
    return [
        render(x) for x in xs
    ]

