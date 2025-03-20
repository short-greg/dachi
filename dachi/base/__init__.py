from ._base import (
    Storable, Renderable,
    Templatable, Trainable
)
from ._core import (
    render,
    render_multi, 
    is_renderable,
    Renderable, model_template,
    struct_template,
    model_to_text, model_from_text,
    StructLoadException,
    TemplateField, doc
)

