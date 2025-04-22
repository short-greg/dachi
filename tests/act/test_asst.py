import pytest
from dachi.act import _asst as F
from dachi.act import TaskStatus
from dachi.store import _data as store
import typing
from ..asst.test_ai import DummyAIModel
from dachi.msg._messages import Msg
from dachi import store
import time
from dachi.utils import Args
from dachi.asst import Op, Threaded, NullOut, ToText


class TestStreamModel:

    def test_stream_assist_initial_status(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.stream_assist(model, message, buffer, ctx, out='content', interval=1./40.)
        
        status = stream()
        assert status == TaskStatus.RUNNING

    def test_stream_assist_success_status(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.stream_assist(
            model, message, buffer, 
            ctx, out='content', 
            interval=1./400.
        )
        
        stream()
        time.sleep(0.15)
        status = stream()
        assert status == TaskStatus.SUCCESS

    def test_stream_assist_buffer_content(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.stream_assist(model, message, buffer, ctx, out='content', interval=1./400.)
        
        stream()
        time.sleep(0.1)
        stream()
        result = ''.join(buffer.get())
        assert result == 'Great!'

    def test_stream_assist_with_empty_prompt(self):
        buffer = store.Buffer()
        model = DummyAIModel(target='')
        message = Msg(role='user', text='')
        ctx = store.Context()
        stream = F.stream_assist(model, message, buffer, ctx, out='content', interval=1./400.)
        
        stream()
        time.sleep(0.1)
        stream()
        result = ''.join(buffer.get())
        assert result == ''

    def test_stream_assist_with_invalid_interval(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        with pytest.raises(ValueError):
            F.stream_assist(model, message, buffer, ctx, out='content', interval=-1)

    def test_stream_assist_with_large_output(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text' * 1000)
        ctx = store.Context()
        stream = F.stream_assist(model, message, buffer, ctx, out='content', interval=1./400.)
        
        stream()
        time.sleep(0.2)
        result = ''.join(buffer.get())
        assert len(result) > 0

    def test_stream_assist_with_multiple_calls(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.stream_assist(model, message, buffer, ctx, out='content', interval=1./400.)
        
        for _ in range(5):
            stream()
            time.sleep(0.05)
        result = ''.join(buffer.get())
        assert result == 'Great!'

    def test_stream_assist_with_invalid_context(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = None
        with pytest.raises(AttributeError):
            F.stream_assist(buffer, model, message, ctx, out='content', interval=1./400.)

    def test_stream_assist_with_non_string_output(self):
        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.stream_assist(
            model, message, buffer, 
            ctx, out='content', interval=1./400.
        )
        
        stream()
        time.sleep(0.15)
        stream()
        result = ''.join(buffer.get())
        assert result == 'Great!'

    def test_buffer_returns_correct_Status(self):

        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(
            role='user', text='text'
        )
        ctx = store.Context()
        stream = F.stream_assist(
            model, message, buffer, ctx, interval=1./400.,
            out='content'
        )
        res = stream()
        time.sleep(0.15)
        res = stream()

        assert res == TaskStatus.SUCCESS

    def test_buffer_has_correct_value(self):

        buffer = store.Buffer()
        model = DummyAIModel()
        message = Msg(
            role='user', text='text'
        )
        ctx = store.Context()
        stream = F.stream_assist(
            model, message, buffer, ctx, out='content', interval=1./400.
        )
        stream()
        time.sleep(0.1)
        stream()
        res = ''.join((r for r in buffer.get()))

        assert res == 'Great!'


class TestRunAssist:

    def test_run_assist_initial_status(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        status = stream()
        assert status == TaskStatus.RUNNING

    def test_run_assist_success_status(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        stream()
        time.sleep(0.15)
        status = stream()
        assert status == TaskStatus.SUCCESS

    def test_run_assist_shared_data(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        stream()
        time.sleep(0.1)
        stream()
        assert shared.data == 'Great!'

    def test_run_assist_with_empty_prompt(self):
        shared = store.Shared()
        model = DummyAIModel(target='')
        message = Msg(role='user', text='')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        stream()
        time.sleep(0.1)
        stream()
        assert shared.data == ''

    def test_run_assist_with_large_output(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text' * 1000)
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        stream()
        time.sleep(0.2)
        assert len(shared.data) > 0

    def test_run_assist_with_multiple_calls(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        for _ in range(5):
            stream()
            time.sleep(0.05)
        assert shared.data == 'Great!'

    # FIGURE THIS OUT
    # def test_run_assist_with_invalid_context(self):
    #     shared = store.Shared()
    #     model = DummyAIModel()
    #     message = Msg(role='user', text='text')
    #     with pytest.raises(AttributeError):
    #         run = F.run_assist(model, message, shared, None, out='content', interval=1./400.)
    #         run()

    def test_run_assist_with_non_string_output(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        stream()
        time.sleep(0.1)
        stream()
        assert shared.data == 'Great!'

    def test_run_assist_thread_reuse(self):
        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(role='user', text='text')
        ctx = store.Context()
        stream = F.run_assist(model, message, shared, ctx, out='content')
        
        stream()
        time.sleep(0.1)
        thread_before = ctx['_thread']
        stream()
        thread_after = ctx['_thread']
        assert thread_before is thread_after

    def test_run_assist_returns_correct_status(self):

        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(
            role='user', text='text'
        )
        ctx = store.Context()
        stream = F.run_assist(
            model, message, shared, ctx, out='content'
        )
        res = stream()
        time.sleep(0.1)
        res = stream()

        assert res == TaskStatus.SUCCESS

    def test_run_assist(self):

        shared = store.Shared()
        model = DummyAIModel()
        message = Msg(
            role='user', text='text'
        )
        ctx = store.Context()
        run = F.run_assist(
            model, message, shared, ctx, out='content'
        )
        run()
        time.sleep(0.1)
        run()
        assert shared.data == 'Great!'


class TestStreamOp:

    def test_stream_op_initial_status(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel(
            "", proc=[NullOut('out', 'content')]), 
            ToText(), out='out'
        )
        stream = F.stream_op(op, buffer, ctx, out='out')
        
        status = stream()
        assert status == TaskStatus.RUNNING

    def test_stream_op_initial_status_raises_error(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel(
            "", proc=[NullOut('out', 'content')]), 
            ToText(), out='out'
        )
        
        with pytest.raises(TypeError):
            stream = F.stream_op(op, buffer, ctx, x=2, out='out')
            status = stream()
            time.sleep(0.15)
            stream()

    def test_stream_op_success_status(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel(
            "Hello", proc=[NullOut('out', 'content')]), 
            ToText(), out='out'
        )
        stream = F.stream_op(
            op, buffer, ctx, Args(text='Hi!'),
            out='out', 
            
            interval=1./400.
        )
        
        stream()
        time.sleep(0.1)
        status = stream()
        assert status == TaskStatus.SUCCESS

    def test_stream_op_buffer_content(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel(
            "Hello", 
            proc=[NullOut('out', 'content')]), ToText(), out='out')
        stream = F.stream_op(
            op, buffer, ctx, Args(text='hi'), interval=1./400.
        )
        
        stream()
        time.sleep(0.15)
        stream()
        result = ''.join(buffer.get())
        assert result == 'Hello'

    def test_stream_op_with_empty_input(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(), out='out')
        stream = F.stream_op(op, buffer, ctx, out='out', interval=1./400.)
        
        stream()
        time.sleep(0.1)
        result = ''.join(buffer.get())
        assert result == ''

    def test_stream_op_with_invalid_interval(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(), out='out')
        with pytest.raises(ValueError):
            F.stream_op(op, buffer, ctx, out='out', interval=-1)

    # def test_stream_op_with_large_output(self):
    #     buffer = store.Buffer()
    #     ctx = store.Context()
    #     op = Op(DummyAIModel("Hello" * 1000, proc=[NullOut('out', 'content')]), ToText(), out='out')
    #     stream = F.stream_op(op, buffer, ctx, Args(text='hi'), out='out', interval=1./400.)
        
    #     stream()
    #     time.sleep(0.)
    #     stream()
    #     result = ''.join(buffer.get())
    #     assert len(result) > 0

    def test_stream_op_with_multiple_calls(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel("Hello", proc=[NullOut('out', 'content')]), ToText(), out='out')
        stream = F.stream_op(op, buffer, ctx, Args(text='hi'), out='out', interval=1./400.)
        
        for _ in range(5):
            stream()
            time.sleep(0.05)
        result = ''.join(buffer.get())
        assert result == 'Hello'

    def test_stream_op_with_invalid_context(self):
        buffer = store.Buffer()
        ctx = None
        op = Op(DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(), out='out')
        with pytest.raises(AttributeError):
            F.stream_op(op, buffer, ctx, out='out', interval=1./400.)

    def test_stream_op_thread_reuse(self):
        buffer = store.Buffer()
        ctx = store.Context()
        op = Op(DummyAIModel("Hello", proc=[NullOut('out', 'content')]), ToText(), out='out')
        stream = F.stream_op(op, buffer, ctx, Args(text='text'), out='out', interval=1./400.)
        
        stream()
        time.sleep(0.1)
        thread_before = ctx['_thread']
        stream()
        thread_after = ctx['_thread']
        assert thread_before is thread_after


class TestRunOp:

    def test_run_op_initial_status(self):
        shared = store.Shared()
        ctx = store.Context()
        op = Op(DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(),  out='out')
        run = F.run_op(op, shared, ctx, Args(text='text'),)
        
        status = run()
        assert status == TaskStatus.RUNNING

    def test_run_op_success_status(self):
        shared = store.Shared()
        ctx = store.Context()
        op = Op(DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(), out='out')
        run = F.run_op(op, shared, ctx, Args(text='text'))
        
        run()
        time.sleep(0.1)
        status = run()
        assert status == TaskStatus.SUCCESS

    def test_run_op_shared_data(self):
        shared = store.Shared()
        ctx = store.Context()
        op = Op(DummyAIModel("Hello", proc=[NullOut('out', 'content')]), ToText(), out='out')
        run = F.run_op(op, shared, ctx, Args(text='text'))
        
        run()
        time.sleep(0.1)
        assert shared.data == 'Hello'

    def test_run_op_with_empty_input(self):
        shared = store.Shared()
        ctx = store.Context()
        op = Op(DummyAIModel("", proc=[NullOut('out', 'content')]), ToText(), out='out')
        run = F.run_op(op, shared, ctx, Args(text='text'))
        
        run()
        time.sleep(0.1)
        assert shared.data == ''

    def test_run_op_with_threaded_model(self):
        shared = store.Shared()
        ctx = store.Context()
        threaded = Threaded(
            DummyAIModel("Hello", proc=[NullOut('out', 'content')]),
            router={"user": ToText("user"), "system": ToText("system")},
            out='out'
        )
        run = F.run_op(threaded, shared, ctx, Args(text='text', route='user'))
        
        run()
        time.sleep(0.1)
        run()
        assert shared.data == 'Hello'

    def test_run_op_thread_reuse(self):
        shared = store.Shared()
        ctx = store.Context()
        op = Op(DummyAIModel("Hello", proc=[NullOut('out', 'content')]), ToText(), out='out')
        run = F.run_op(op, shared, ctx, Args(text='text'))
        
        run()
        time.sleep(0.1)
        thread_before = ctx['_thread']
        run()
        thread_after = ctx['_thread']
        assert thread_before is thread_after

    def test_run_op_with_multiple_calls(self):
        shared = store.Shared()
        ctx = store.Context()
        op = Op(DummyAIModel("Hello", proc=[NullOut('out', 'content')]), ToText(), out='out')
        run = F.run_op(op, shared, ctx, Args(text='text'))
        
        for _ in range(5):
            run()
            time.sleep(0.05)
        assert shared.data == 'Hello'
