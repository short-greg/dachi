"""Dictionary utilities and data storage for Dachi communication system.

Provides utility functions for dictionary manipulation, accumulation patterns,
and buffered data collection commonly used in AI processing pipelines.
"""

import typing

"""Constant for UNDEFINED. usage: value is UNDEFINED"""
UNDEFINED = object()

primitives = (bool, str, int, float, type(None))
"""a list of primitive types"""


def is_primitive(obj) -> bool:
    """Utility to check if a value is a primitive

    Args:
        obj: Value to check

    Returns:
        bool: If it is a "primitive"
    """
    return type(obj) in primitives


def get_member(obj, loc: str):
    """Get a member from an object recursively

    Args:
        obj : the object
        loc (str): the location as a string, use '.' to indicate sub objects

    Returns:
        Any: The member
    """
    locs = loc.split('.')
    for loc in locs:
        obj = getattr(obj, loc)
    return obj


def get_or_set(d: typing.Dict, key, value) -> typing.Any:
    """Get existing value from dictionary or set and return new value if key doesn't exist.
    
    This is commonly used for lazy initialization patterns in communication systems.

    Args:
        d: The target dictionary
        key: The key to check/set
        value: The value to set if key doesn't exist

    Returns:
        The existing value if key exists, otherwise the newly set value
        
    Example:
        state = {}
        count = get_or_set(state, 'message_count', 0)  # Returns 0, sets state['message_count'] = 0
        count = get_or_set(state, 'message_count', 5)  # Returns 0 (existing value)
    """
    if key in d:
        return d[key]
    d[key] = value
    return value


def sub_dict(d: typing.Dict, key: str) -> typing.Dict:
    """Get or create a nested dictionary for hierarchical data organization.
    
    Essential for organizing complex communication state like processor outputs,
    request metadata, and nested configuration structures.
    
    Args:
        d: The parent dictionary
        key: The key for the nested dictionary
        
    Returns:
        The nested dictionary (existing or newly created)
        
    Raises:
        ValueError: If the key exists but points to a non-dictionary value
        
    Example:
        state = {}
        req_state = sub_dict(state, 'requests')  # Creates state['requests'] = {}
        req_state['req_1'] = 'pending'
        
        # Later access
        req_state = sub_dict(state, 'requests')  # Returns existing dict
    """
    if key in d:
        if not isinstance(d[key], typing.Dict):
            raise ValueError(
                f'The field pointed to be {key} is not a dict.'
            )
    else:
        d[key] = {}
    return d[key]


def get_or_setf(d: typing.Dict, key, f: typing.Callable[[], typing.Any]) -> typing.Any:
    """Get existing value or set value from factory function if key doesn't exist.
    
    Used for expensive initialization where the default value requires computation
    or resource allocation (like creating processors, connections, etc.).

    Args:
        d: The target dictionary
        key: The key to check/set
        f: Factory function to call if key doesn't exist (no arguments)

    Returns:
        The existing value if key exists, otherwise result of calling f()
        
    Example:
        processors = {}
        # Only creates OpenAI client if not already cached
        llm = get_or_setf(processors, 'openai', lambda: OpenAIChat())
    """
    if key in d:
        return d[key]
    d[key] = f()
    return d[key]


def call_or_set(d: typing.Dict, key, value, f: typing.Callable[[typing.Any, typing.Any], typing.Any]) -> typing.Any:
    """Set initial value or combine with existing value using a function.
    
    Useful for accumulation patterns like counting, merging results, or combining
    data from multiple sources in communication pipelines.

    Args:
        d: The target dictionary
        key: The key to check/set
        value: The new value to set or combine
        f: Function to combine existing and new values (existing_value, new_value) -> combined

    Returns:
        The newly set value (if key didn't exist) or combined value (if it did)
        
    Example:
        stats = {}
        # First call sets initial value
        total = call_or_set(stats, 'tokens', 100, lambda old, new: old + new)  # Returns 100
        # Second call combines values  
        total = call_or_set(stats, 'tokens', 50, lambda old, new: old + new)   # Returns 150
    """
    if key not in d:
        d[key] = value
        return value
    d[key] = f(d[key], value)
    return d[key]


def acc(
    d: typing.Dict, key, value, init_val: str=''
) -> typing.Any:
    """Accumulate values using addition operator (commonly for string/number accumulation).
    
    Specialized accumulation function optimized for text streaming, token counting,
    and other additive operations in AI processing.

    Args:
        d: The target dictionary
        key: The key for accumulation
        value: The value to add (ignored if UNDEFINED)
        init_val: Initial value if key doesn't exist (default: empty string)

    Returns:
        The accumulated value after addition
        
    Example:
        # Text streaming accumulation
        text_buffer = {}
        acc(text_buffer, 'response', 'Hello', '')     # 'Hello'
        acc(text_buffer, 'response', ' world', '')   # 'Hello world'
        
        # Token counting
        token_counts = {}
        acc(token_counts, 'total', 150, 0)  # 150
        acc(token_counts, 'total', 75, 0)   # 225
    """

    if key not in d:
        d[key] = init_val
    
    if value is not UNDEFINED:
        d[key] = d[key] + value
    return d[key]


def get_or_spawn(state: typing.Dict, child: str) -> typing.Dict:
    """Get or create a child dictionary for hierarchical state management.
    
    Alias for sub_dict() with clearer semantics for agent/task state organization.
    Commonly used in behavior trees and multi-agent systems for isolated state spaces.

    Args:
        state: The parent state dictionary
        child: The name/ID of the child state space

    Returns:
        The child state dictionary (existing or newly created)
        
    Example:
        # Agent state management
        system_state = {}
        agent1_state = get_or_spawn(system_state, 'agent_1')
        agent1_state['status'] = 'active'
        
        # Task state in behavior trees
        task_states = {}
        monitor_state = get_or_spawn(task_states, 'monitor_task')
    """
    if child not in state:
        state[child] = {}
    return state[child]
