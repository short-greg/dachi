"""Dictionary utilities and data storage for Dachi communication system.

Provides utility functions for dictionary manipulation, accumulation patterns,
and buffered data collection commonly used in AI processing pipelines.
"""

import typing
from ._utils import UNDEFINED


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


# 1st party
import typing
from functools import reduce

# 3rd party
import pydantic
import typing
# local

T = typing.TypeVar("T")
K = bool | str | None | int | float


class Buffer(pydantic.BaseModel):
    """Thread-safe buffer for collecting and processing streaming data.
    
    Essential component for handling streaming AI responses, accumulating tokens,
    collecting multi-step results, and managing data flow in async pipelines.
    Supports controlled access patterns with open/close semantics.
    
    Example:
        # Streaming response collection
        response_buffer = Buffer(buffer=[])
        
        for chunk in llm.stream(message):
            if response_buffer.is_open():
                response_buffer.add(chunk.delta.text)
        
        response_buffer.close()
        full_response = ''.join(response_buffer.get())
    """
    buffer: typing.List
    opened: bool = True
        
    def add(self, *data):
        """Add one or more data items to the buffer.
        
        Commonly used for accumulating streaming tokens, collecting intermediate
        results, or gathering data from multiple async sources.

        Args:
            *data: Variable number of items to add to buffer
            
        Raises:
            RuntimeError: If buffer is closed (no longer accepting data)
            
        Example:
            buffer.add("Hello", " ", "world")  # Adds 3 items
            buffer.add(chunk.text)              # Adds streaming chunk
        """
        
        if not self.opened:
            raise RuntimeError(
                'The buffer is currently closed so cannot add data to it.'
            )

        self.buffer.extend(data)

    def open(self):
        """Open buffer to allow data additions.
        
        Used to resume data collection after temporary closure or
        to reset buffer state for new processing cycles.
        """
        self.opened = True

    def close(self):
        """Close buffer to prevent further data additions.
        
        Typically called when streaming is complete or when transitioning
        to read-only access patterns. Existing data remains accessible.
        """
        self.opened = False

    def is_open(self) -> bool:
        """Check if buffer accepts new data.

        Returns:
            True if buffer is open and accepts new data, False otherwise
        """
        return self.opened
    
    def clear(self):
        """Remove all data from buffer while preserving open/closed state.
        
        Useful for resetting buffer contents between processing cycles
        without changing the buffer's operational state.
        """
        self.buffer.clear()
        
    def it(self) -> 'BufferIter':
        """Create an iterator for sequential buffer access.
        
        Returns:
            BufferIter: Iterator for reading buffer contents sequentially
            
        Example:
            iterator = buffer.it()
            while not iterator.end():
                item = iterator.read()
                process(item)
        """
        return BufferIter(self)

    def __getitem__(self, idx) -> typing.Union[typing.Any, typing.List]:
        """Access buffer data using index, slice, or iterable of indices.

        Args:
            idx: Index (int), slice, or iterable of indices

        Returns:
            Single item, list of items, or sliced data
            
        Example:
            item = buffer[0]           # First item
            items = buffer[1:5]        # Slice 
            items = buffer[[0, 2, 4]]  # Multiple indices
        """
        if isinstance(idx, slice):
            return self.buffer[idx]
        if isinstance(idx, typing.Iterable):
            return [self.buffer[i] for i in idx]
        return self.buffer[idx]
        
    def __len__(self) -> int:
        """Get number of items in buffer.

        Returns:
            Number of items currently stored in buffer
        """
        return len(self.buffer)
    
    def get(self) -> typing.List:
        """Get copy of all buffer contents as list.
        
        Returns:
            New list containing all buffer items (safe from external modification)
            
        Example:
            all_chunks = response_buffer.get()
            full_text = ''.join(all_chunks)
        """
        return [*self.buffer]


class BufferIter(pydantic.BaseModel):
    """Sequential iterator for Buffer data with streaming-aware read patterns.
    
    Provides controlled sequential access to buffer contents with support for
    batch reading, reduction operations, and streaming consumption patterns
    commonly needed in AI pipeline processing.
    
    Example:
        buffer = Buffer(buffer=['chunk1', 'chunk2', 'chunk3'])
        iterator = buffer.it()
        
        # Sequential reading
        while not iterator.end():
            chunk = iterator.read()
            process_chunk(chunk)
            
        # Batch processing
        remaining = iterator.read_all()
        final_result = ''.join(remaining)
    """

    buffer: Buffer
    i: int = 0

    def start(self) -> bool:
        """Check if iterator is at the beginning of buffer.
        
        Returns:
            True if no items have been read yet, False otherwise
        """
        return self.i == 0

    def end(self) -> bool:
        """Check if iterator has reached the end of available data.

        Returns:
            True if all available buffer data has been consumed, False otherwise
        """
        return self.i >= (len(self._buffer) - 1)
    
    def is_open(self) -> bool:
        """Check if underlying buffer accepts new data.
        
        Useful for streaming scenarios where buffer may receive more data
        while iterator is being consumed.

        Returns:
            True if underlying buffer is open and may receive more data
        """
        return self._buffer.is_open()

    def read(self) -> typing.Any:
        """Read next item from buffer sequentially.
        
        Advances iterator position and returns the next available item.
        Essential for processing streaming data item by item.

        Returns:
            The next item in the buffer
            
        Raises:
            StopIteration: If all available data has been consumed
            
        Example:
            try:
                while True:
                    chunk = iterator.read()
                    accumulated_text += chunk
            except StopIteration:
                # All data consumed
                pass
        """
        if self.i < (len(self._buffer) - 1):
            self.i += 1
            return self._buffer[self.i - 1]
        raise StopIteration('Reached the end of the buffer')

    def read_all(self) -> typing.List:
        """Read all remaining items from current position to end.
        
        Consumes all remaining buffer data in one operation. Useful for
        batch processing or when switching from streaming to bulk processing.

        Returns:
            List of all remaining items from current iterator position
            
        Example:
            # Process first few items individually
            first_item = iterator.read()
            second_item = iterator.read()
            
            # Process remaining items in batch
            remaining_items = iterator.read_all()
            batch_process(remaining_items)
        """
        
        if self.i < len(self._buffer):
            i = self.i
            self.i = len(self._buffer)
            return self._buffer[i:]
            
        return []

    def read_reduce(self, f, init_val=None) -> typing.Any:
        """Apply reduction function to all remaining buffer items.
        
        Combines read_all() with functional reduction for efficient processing
        of accumulated data. Common for text joining, numerical aggregation, etc.

        Args:
            f: Reduction function taking (accumulator, item) -> new_accumulator
            init_val: Initial value for reduction (None uses first item)

        Returns:
            Result of applying reduction function to all remaining items
            
        Example:
            # Sum numerical tokens
            total = iterator.read_reduce(lambda acc, x: acc + x, 0)
            
            # Join text chunks
            text = iterator.read_reduce(lambda acc, chunk: acc + chunk, '')
        """
        data = self.read_all()
        return reduce(
            f, data, init_val
        )

    def read_map(self, f) -> typing.Iterator:
        """Apply transformation function to all remaining buffer items.
        
        Combines read_all() with functional mapping for efficient processing
        pipelines. Returns iterator for memory-efficient processing.

        Args:
            f: Transformation function taking item -> transformed_item

        Returns:
            Iterator of transformed items
            
        Example:
            # Transform text chunks to uppercase
            upper_chunks = iterator.read_map(str.upper)
            
            # Parse JSON chunks
            parsed_chunks = iterator.read_map(json.loads)
            
            # Convert to list if needed
            all_transformed = list(iterator.read_map(transform_func))
        """
        data = self.read_all()
        return map(
            f, data
        )
