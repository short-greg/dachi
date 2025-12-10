import inspect
import typing
import pydantic


def is_generic_type(annotation) -> bool:
    """Detect whether an annotation represents a generic type.

    A generic type is a parameterized type like list[int], Dict[str, Any],
    or ModuleList[Task]. This function handles various forms of annotations
    including strings and ForwardRefs.

    Args:
        annotation: Type annotation (could be type, string, ForwardRef, etc.)

    Returns:
        True if annotation appears to be a generic type, False otherwise

    Examples:
        >>> is_generic_type(int)
        False
        >>> is_generic_type(list[int])
        True
        >>> is_generic_type(Dict[str, int])
        True
        >>> is_generic_type("List[int]")
        True
        >>> is_generic_type("int")
        False
        >>> is_generic_type(ForwardRef("Dict[str, Any]"))
        True

    Notes:
        - For string annotations, uses heuristic (presence of '[')
        - For ForwardRef, checks the forward argument string
        - For type annotations, uses typing.get_origin()
        - Unsubscripted generic classes (e.g., list, dict) return False
    """
    if isinstance(annotation, str):
        return '[' in annotation

    if isinstance(annotation, typing.ForwardRef):
        return '[' in annotation.__forward_arg__

    origin = typing.get_origin(annotation)
    return origin is not None

def generic_class(t: typing.TypeVar, idx: int=0) -> typing.Type:
    """Gets the generic type for a class assuming that it only has 
    one.

    Args:
        t (typing.TypeVar): The class to get the generic type for
        idx (int, optional): . Defaults to 0.

    Returns:
        typing.Type: the type specified by the generic class
    """
    return t.__orig_class__.__args__[idx]

def resolve_fields(ctx, cls) -> dict:
    """Resolve fields from a context dict based on class definition.
    
    Takes a dictionary of context data and a class definition, returns only 
    the fields the class defines, with defaults applied.
    
    Args:
        ctx: Dictionary of context data (may contain extra fields) or dict-like object
        cls: Class definition with annotations and/or attributes as defaults
        
    Returns:
        dict: Dictionary containing only the fields defined by cls, with defaults
        
    Raises:
        TypeError: If ctx is not dict-like
        KeyError: If required field (no default) is missing from ctx
    """
    # Support dict-like objects (like Ctx) with __getitem__ and __contains__
    if not (hasattr(ctx, '__getitem__') and hasattr(ctx, '__contains__')):
        raise TypeError("ctx must be dict-like (support __getitem__ and __contains__)")

    # Gather declared keys: annotations + simple attributes
    ann = getattr(cls, "__annotations__", {})
    keys = set(ann.keys())
    for k, v in cls.__dict__.items():
        if k.startswith("__"):
            continue
        if callable(v) or isinstance(v, (staticmethod, classmethod, property)):
            continue
        keys.add(k)

    out = {}
    for k in keys:
        if k in ctx:
            out[k] = ctx[k]
        elif k in cls.__dict__:
            out[k] = cls.__dict__[k]
        else:
            raise KeyError(f"Missing required key: {k}")
    return out


def resolve_from_signature(ctx, func, exclude_params=None) -> dict:
    """Resolve function parameters from context data using function signature.
    
    Extracts parameters from function signature and resolves their values
    from the context dictionary, applying defaults where available.
    
    Args:
        ctx: Dictionary of context data (may contain extra fields) or dict-like object
        func: Function whose signature to inspect for parameters
        exclude_params: Set of parameter names to exclude from resolution
        
    Returns:
        dict: Dictionary of resolved parameters ready for **kwargs
        
    Raises:
        TypeError: If ctx is not dict-like
        KeyError: If required parameter (no default) is missing from ctx
    """
    # Support dict-like objects (like Ctx) with __getitem__ and __contains__
    if not (hasattr(ctx, '__getitem__') and hasattr(ctx, '__contains__')):
        raise TypeError("ctx must be dict-like (support __getitem__ and __contains__)")
    
    if exclude_params is None:
        exclude_params = set()
    
    sig = inspect.signature(func)
    out = {}
    
    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter for methods
        if param_name == 'self':
            continue
            
        # Skip excluded parameters
        if param_name in exclude_params:
            continue
            
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        
        if param_name in ctx:
            # Use value from context
            out[param_name] = ctx[param_name]
        elif param.default is not param.empty:
            # Use default value from function signature
            out[param_name] = param.default
        else:
            # Required parameter missing
            raise KeyError(f"Missing required parameter: {param_name}")

    return out


def python_type_to_json_schema_type(python_type: str) -> str:
    """Convert Python type name to JSON Schema type.

    Args:
        python_type: Python type name (e.g., 'str', 'int', 'bool')

    Returns:
        Corresponding JSON Schema type (e.g., 'string', 'integer', 'boolean')

    Example:
        >>> python_type_to_json_schema_type('str')
        'string'
        >>> python_type_to_json_schema_type('int')
        'integer'
    """
    type_mapping = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object',
        'NoneType': 'null'
    }
    return type_mapping.get(python_type, 'string')


def python_type_to_json_schema(python_type: typing.Any) -> dict:
    """Convert Python type to JSON schema dict.

    Handles basic types (int, str, float, bool, list, dict, None) and
    typing module types (List[T], Dict[K,V], Optional[T], Union[T1, T2]).

    Args:
        python_type: Python type or typing generic

    Returns:
        JSON schema dict representation

    Example:
        >>> python_type_to_json_schema(int)
        {'type': 'integer'}
        >>> python_type_to_json_schema(typing.List[str])
        {'type': 'array', 'items': {'type': 'string'}}
        >>> python_type_to_json_schema(typing.Union[int, str])
        {'oneOf': [{'type': 'integer'}, {'type': 'string'}]}
    """
    type_map = {
        int: {"type": "integer"},
        str: {"type": "string"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"}
    }

    if python_type in type_map:
        return type_map[python_type]

    origin = getattr(python_type, '__origin__', None)

    if origin is list:
        args = getattr(python_type, '__args__', ())
        schema = {"type": "array"}
        if args:
            schema["items"] = python_type_to_json_schema(args[0])
        return schema

    if origin is dict:
        args = getattr(python_type, '__args__', ())
        schema = {"type": "object"}
        if args and len(args) == 2:
            schema["additionalProperties"] = python_type_to_json_schema(args[1])
        return schema

    if origin is typing.Union:
        args = getattr(python_type, '__args__', ())
        return {"oneOf": [python_type_to_json_schema(arg) for arg in args]}

    return {"type": "string"}


def is_generic_type(annotation) -> bool:
    """Detect whether an annotation represents a generic type.

    A generic type is a parameterized type like list[int], Dict[str, Any],
    or ModuleList[Task]. This function handles various forms of annotations
    including strings and ForwardRefs.

    Args:
        annotation: Type annotation (could be type, string, ForwardRef, etc.)

    Returns:
        True if annotation appears to be a generic type, False otherwise

    Examples:
        >>> is_generic_type(int)
        False
        >>> is_generic_type(list[int])
        True
        >>> is_generic_type(Dict[str, int])
        True
        >>> is_generic_type("List[int]")
        True
        >>> is_generic_type("int")
        False
        >>> is_generic_type(ForwardRef("Dict[str, Any]"))
        True

    Notes:
        - For string annotations, uses heuristic (presence of '[')
        - For ForwardRef, checks the forward argument string
        - For type annotations, uses typing.get_origin()
        - Unsubscripted generic classes (e.g., list, dict) return False
    """
    if isinstance(annotation, str):
        return '[' in annotation

    if isinstance(annotation, typing.ForwardRef):
        return '[' in annotation.__forward_arg__

    origin = typing.get_origin(annotation)
    return origin is not None


def generic_class(t: typing.TypeVar, idx: int=0) -> typing.Type:
    """Gets the generic type for a class assuming that it only has 
    one.

    Args:
        t (typing.TypeVar): The class to get the generic type for
        idx (int, optional): . Defaults to 0.

    Returns:
        typing.Type: the type specified by the generic class
    """
    return t.__orig_class__.__args__[idx]


def is_nested_model(
    pydantic_model_cls: typing.Type[pydantic.BaseModel]
) -> bool:
    """Helper function to check if it is a nested model

    Args:
        pydantic_model_cls (typing.Type[pydantic.BaseModel]): The class to check if it is a nested model

    Returns:
        bool: If it is a nested model
    """
    for field in pydantic_model_cls.model_fields.values():
        
        if isinstance(field.annotation, type) and issubclass(field.annotation, pydantic.BaseModel):
            return True
    return False
