import pytest
from dachi.act.comm._buffer import Buffer, BufferIter


class TestBuffer:

    def test_buffer_creation(self):
        buffer = Buffer(buffer=[])
        assert isinstance(buffer, Buffer)
        assert len(buffer) == 0
        assert buffer.is_open() == True

    def test_buffer_add_items(self):
        buffer = Buffer(buffer=[])
        buffer.add("hello", "world")
        assert len(buffer) == 2
        assert buffer.get() == ["hello", "world"]

    def test_buffer_add_when_closed_raises_error(self):
        buffer = Buffer(buffer=[])
        buffer.close()
        with pytest.raises(RuntimeError, match="The buffer is currently closed"):
            buffer.add("item")

    def test_buffer_open_close(self):
        buffer = Buffer(buffer=[])
        assert buffer.is_open() == True
        
        buffer.close()
        assert buffer.is_open() == False
        
        buffer.open()
        assert buffer.is_open() == True

    def test_buffer_clear(self):
        buffer = Buffer(buffer=["a", "b", "c"])
        assert len(buffer) == 3
        
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.get() == []

    def test_buffer_indexing(self):
        buffer = Buffer(buffer=["a", "b", "c", "d"])
        
        # Single index
        assert buffer[0] == "a"
        assert buffer[2] == "c"
        
        # Slice
        assert buffer[1:3] == ["b", "c"]
        
        # Multiple indices
        assert buffer[[0, 2]] == ["a", "c"]

    def test_buffer_get_returns_copy(self):
        buffer = Buffer(buffer=["a", "b", "c"])
        items = buffer.get()
        items.append("d")
        
        # Original buffer should be unchanged
        assert buffer.get() == ["a", "b", "c"]
        assert len(buffer) == 3


class TestBufferIter:

    def test_buffer_iter_creation(self):
        buffer = Buffer(buffer=["a", "b", "c"])
        iterator = buffer.it()
        assert isinstance(iterator, BufferIter)
        assert iterator.start() == True
        assert iterator.end() == False

    def test_buffer_iter_read_sequential(self):
        buffer = Buffer(buffer=["a", "b", "c"])
        iterator = buffer.it()
        
        assert iterator.read() == "a"
        assert iterator.read() == "b"
        assert iterator.read() == "c"
        
        with pytest.raises(StopIteration):
            iterator.read()

    def test_buffer_iter_read_all(self):
        buffer = Buffer(buffer=["a", "b", "c", "d"])
        iterator = buffer.it()
        
        # Read first item
        assert iterator.read() == "a"
        
        # Read all remaining
        remaining = iterator.read_all()
        assert remaining == ["b", "c", "d"]
        
        # Should be empty now
        assert iterator.read_all() == []

    def test_buffer_iter_read_reduce(self):
        buffer = Buffer(buffer=[1, 2, 3, 4])
        iterator = buffer.it()
        
        # Read first item
        assert iterator.read() == 1
        
        # Reduce remaining items
        total = iterator.read_reduce(lambda acc, x: acc + x, 0)
        assert total == 9  # 2 + 3 + 4
        
    def test_buffer_iter_read_map(self):
        buffer = Buffer(buffer=["hello", "world", "test"])
        iterator = buffer.it()
        
        # Read first item
        assert iterator.read() == "hello"
        
        # Map remaining items
        upper_items = list(iterator.read_map(str.upper))
        assert upper_items == ["WORLD", "TEST"]

    def test_buffer_iter_end_detection(self):
        buffer = Buffer(buffer=["a", "b"])
        iterator = buffer.it()
        
        assert iterator.end() == False
        iterator.read()  # "a"
        assert iterator.end() == False
        iterator.read()  # "b"
        assert iterator.end() == True

    def test_buffer_iter_is_open_reflects_buffer_state(self):
        buffer = Buffer(buffer=["a", "b"])
        iterator = buffer.it()
        
        assert iterator.is_open() == True
        buffer.close()
        assert iterator.is_open() == False
        buffer.open()
        assert iterator.is_open() == True

    def test_buffer_iter_with_empty_buffer(self):
        buffer = Buffer(buffer=[])
        iterator = buffer.it()
        
        assert iterator.start() == True
        assert iterator.end() == True
        assert iterator.read_all() == []
        
        with pytest.raises(StopIteration):
            iterator.read()


class TestBufferStreamingScenarios:
    """Test scenarios common in streaming AI applications."""

    def test_streaming_text_accumulation(self):
        # Simulate streaming text response
        buffer = Buffer(buffer=[])
        
        chunks = ["Hello", " ", "world", "!", " How", " are", " you", "?"]
        for chunk in chunks:
            buffer.add(chunk)
        
        buffer.close()
        
        # Reconstruct full text
        full_text = ''.join(buffer.get())
        assert full_text == "Hello world! How are you?"

    def test_streaming_with_iterator_processing(self):
        # Simulate processing chunks as they arrive
        buffer = Buffer(buffer=[])
        chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
        
        for chunk in chunks:
            buffer.add(chunk)
        
        buffer.close()
        
        # Process with iterator
        iterator = buffer.it()
        processed = []
        while not iterator.end():
            try:
                chunk = iterator.read()
                processed.append(f"processed_{chunk}")
            except StopIteration:
                break
                
        assert processed == ["processed_chunk1", "processed_chunk2", "processed_chunk3", "processed_chunk4"]

    def test_partial_consumption_then_bulk_read(self):
        # Simulate reading some chunks individually, then bulk processing rest
        buffer = Buffer(buffer=["token1", "token2", "token3", "token4", "token5"])
        iterator = buffer.it()
        
        # Process first two individually
        first = iterator.read()
        second = iterator.read()
        
        # Process remaining in bulk
        remaining = iterator.read_all()
        
        assert first == "token1"
        assert second == "token2"
        assert remaining == ["token3", "token4", "token5"]

    def test_streaming_json_like_processing(self):
        # Simulate processing JSON-like chunks
        buffer = Buffer(buffer=[
            {"delta": "Hello"},
            {"delta": " world"},
            {"delta": "!"},
            {"finish_reason": "stop"}
        ])
        
        iterator = buffer.it()
        text_chunks = []
        
        while not iterator.end():
            try:
                chunk = iterator.read()
                if "delta" in chunk:
                    text_chunks.append(chunk["delta"])
                elif "finish_reason" in chunk:
                    break
            except StopIteration:
                break
                
        full_text = ''.join(text_chunks)
        assert full_text == "Hello world!"