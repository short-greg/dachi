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
