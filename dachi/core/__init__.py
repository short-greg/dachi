from dataclasses import InitVar
from typing import ClassVar
from ._base import (
    to_kind,
    END_TOK,
    NULL_TOK,
    TextMsg,
    Msg,
    Inp,
    Renderable,
    Templatable,
    ExampleMixin,
)
from ._module import (
    PrivateParam,
    PrivateRuntime,
    PrivateShared,
    Module,
    AdaptModule,
    Checkpoint,
)
from ._registry import (
    Registry,
    RegistryEntry
)
from ._shareable import (
    ShareableItem,
    Param,
    Runtime,
    Shared,
    ParamSet,
    PARAM,
    Trainable
)

from ._structs import (
    ModuleDict,
    ModuleList,
)
# from ._dialog import (
#     Tree,
#     END_TOK,
#     NULL_TOK,
#     TextMsg,
#     Inp
#     # FieldRenderer
# )
from ._render import (
    TemplateField,
    model_to_text,
    render,
    render_multi,
    is_renderable,
    generate_numbered_list,
    parse_function_spec,
    Styling,
    style_formatter,
    numbered,
    bullet,
    bold,
    italic,
    DEFAULT_STYLE,
    struct_template
)
from ._scope import (
    Scope,
    Ctx
)
