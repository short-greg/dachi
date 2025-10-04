"""Unit tests for StateChart event system components.

Tests cover EventQueue, Post, Timer, and MonotonicClock classes following the
framework testing conventions.
"""

import asyncio
import pytest
import time
# Import directly to avoid circular import issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dachi.act._chart import _event
EventQueue = _event.EventQueue
Event = _event.Event  
MonotonicClock = _event.MonotonicClock


class TestEventQueue:
    
    def test_post_nowait_with_string_creates_event(self):
        queue = EventQueue()
        result = queue.post_nowait("test_event")
        assert result is True
        assert queue.size() == 1
    
    def test_post_nowait_with_event_object_adds_to_queue(self):
        queue = EventQueue()
        event = Event(type="test", ts=time.monotonic())
        result = queue.post_nowait(event)
        assert result is True
        assert queue.size() == 1
    
    def test_post_nowait_when_full_drop_newest_returns_false(self):
        queue = EventQueue(maxsize=1, overflow="drop_newest")
        queue.post_nowait("event1")
        result = queue.post_nowait("event2")
        assert result is False
        assert queue.size() == 1
    
    def test_post_nowait_when_full_drop_oldest_removes_first(self):
        queue = EventQueue(maxsize=1, overflow="drop_oldest")
        queue.post_nowait("event1")
        result = queue.post_nowait("event2")
        assert result is True
        assert queue.size() == 1
    
    def test_pop_nowait_returns_first_event(self):
        queue = EventQueue()
        queue.post_nowait("first")
        queue.post_nowait("second")
        event = queue.pop_nowait()
        assert event["type"] == "first"
        assert queue.size() == 1
    
    def test_pop_nowait_when_empty_raises_indexerror(self):
        queue = EventQueue()
        with pytest.raises(IndexError, match="pop from an empty queue"):
            queue.pop_nowait()
    
    def test_empty_returns_true_when_no_events(self):
        queue = EventQueue()
        assert queue.empty() is True
    
    def test_empty_returns_false_when_has_events(self):
        queue = EventQueue()
        queue.post_nowait("test")
        assert queue.empty() is False
    
    def test_clear_removes_all_events(self):
        queue = EventQueue()
        queue.post_nowait("event1")
        queue.post_nowait("event2")
        queue.clear()
        assert queue.empty() is True
    
    def test_state_dict_captures_queue_state(self):
        queue = EventQueue(maxsize=512, overflow="drop_oldest")
        queue.post_nowait("event1")
        queue.post_nowait("event2")
        
        state = queue.state_dict()
        
        assert state["maxsize"] == 512
        assert state["overflow"] == "drop_oldest"
        assert len(state["events"]) == 2
        assert state["events"][0]["type"] == "event1"
        assert state["events"][1]["type"] == "event2"
    
    def test_load_state_dict_restores_queue_state(self):
        queue = EventQueue()
        
        state = {
            "maxsize": 256,
            "overflow": "block",
            "events": [
                {"type": "restored_event", "ts": 123.45},
                {"type": "another_event", "ts": 567.89}
            ]
        }
        
        queue.load_state_dict(state)
        
        assert queue.maxsize == 256
        assert queue.overflow == "block"
        assert queue.size() == 2
        
        event1 = queue.pop_nowait()
        assert event1["type"] == "restored_event"
        assert event1["ts"] == 123.45
        
        event2 = queue.pop_nowait()
        assert event2["type"] == "another_event"


class TestMonotonicClock:
    
    def test_now_returns_float(self):
        clock = MonotonicClock()
        now = clock.now()
        assert isinstance(now, float)
        assert now > 0
    
    def test_now_advances_over_time(self):
        clock = MonotonicClock()
        first = clock.now()
        time.sleep(0.001)  # 1ms sleep
        second = clock.now()
        assert second > first


@pytest.mark.asyncio
class TestMonotonicClockAsync:
    
    async def test_sleep_until_waits_correct_duration(self):
        clock = MonotonicClock()
        start = clock.now()
        target = start + 0.01  # 10ms
        await clock.sleep_until(target)
        elapsed = clock.now() - start
        assert elapsed >= 0.01  # Should wait at least 10ms
    
    async def test_sleep_until_past_time_returns_immediately(self):
        clock = MonotonicClock()
        start = clock.now()
        past_time = start - 1.0  # 1 second ago
        await clock.sleep_until(past_time)
        elapsed = clock.now() - start
        assert elapsed < 0.001  # Should return almost immediately


@pytest.mark.asyncio  
class TestEventQueueAsync:
    
    async def test_post_calls_post_nowait(self):
        queue = EventQueue()
        result = await queue.post("test_event")
        assert result is True
        assert queue.size() == 1


class TestPost:
    
    def test_register_finish_callback_adds_callback(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")], epoch=1)

        callback = lambda: None
        post.register_finish_callback(callback)

        assert callback in post._finish_callbacks
    
    def test_register_finish_callback_prevents_duplicates(self):
        queue = EventQueue()
        post = _event.Post(queue=queue)
        
        callback = lambda: None
        post.register_finish_callback(callback)
        post.register_finish_callback(callback)
        
        assert len(post._finish_callbacks) == 1
    
    def test_unregister_finish_callback_removes_callback(self):
        queue = EventQueue()
        post = _event.Post(queue=queue)
        
        callback = lambda: None
        post.register_finish_callback(callback)
        post.unregister_finish_callback(callback)
        
        assert callback not in post._finish_callbacks
    
    @pytest.mark.asyncio
    async def test_aforward_posts_event_with_source_info(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")], epoch=42)

        result = await post.aforward("TestEvent", {"key": "value"})

        assert result is True
        assert queue.size() == 1

        event = queue.pop_nowait()
        assert event["type"] == "TestEvent"
        assert event["payload"] == {"key": "value"}
        assert event["source"] == [("region1", "state1")]
        assert event["epoch"] == 42
        assert event["scope"] == "chart"  # default
    
    @pytest.mark.asyncio
    async def test_aforward_with_custom_scope(self):
        queue = EventQueue()
        post = _event.Post(queue=queue)
        
        await post.aforward("TestEvent", scope="parent")
        
        event = queue.pop_nowait()
        assert event["scope"] == "parent"
    
    @pytest.mark.asyncio
    async def test_finish_calls_registered_callbacks_and_posts_event(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")])

        callback_called = False
        def callback():
            nonlocal callback_called
            callback_called = True

        post.register_finish_callback(callback)
        await post.finish()

        # Check callback was called
        assert callback_called is True

        # Check Finished event was posted
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "Finished"
        assert event["source"] == [("region1", "state1")]
    
    @pytest.mark.asyncio
    async def test_finish_calls_async_callbacks(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")])

        callback_called = False
        async def async_callback():
            nonlocal callback_called
            callback_called = True

        post.register_finish_callback(async_callback)
        await post.finish()

        assert callback_called is True
    
    @pytest.mark.asyncio
    async def test_call_delegates_to_aforward(self):
        queue = EventQueue()
        post = _event.Post(queue=queue)

        result = await post("TestEvent", {"data": "test"})

        assert result is True
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "TestEvent"
        assert event["payload"] == {"data": "test"}


class TestPostChild:

    def test_child_extends_parent_source_list(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")])

        child_post = post.child("child_region_0", "child_state_0")

        assert child_post.source == [("region1", "state1"), ("child_region_0", "child_state_0")]

    def test_child_creates_deeply_nested_source(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "composite1")])

        child_0 = post.child("child_region_0", "state0")
        child_0_1 = child_0.child("child_region_1", "state1")

        assert child_0_1.source == [("region1", "composite1"), ("child_region_0", "state0"), ("child_region_1", "state1")]

    def test_child_with_multiple_names_creates_distinct_sources(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("main", "root")])

        child_a = post.child("child_a", "state_a")
        child_b = post.child("child_b", "state_b")
        child_c = post.child("child_c", "state_c")

        assert child_a.source == [("main", "root"), ("child_a", "state_a")]
        assert child_b.source == [("main", "root"), ("child_b", "state_b")]
        assert child_c.source == [("main", "root"), ("child_c", "state_c")]

    def test_child_has_empty_finish_callbacks(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("main", "root")])

        parent_callback = lambda: None
        post.register_finish_callback(parent_callback)

        child_post = post.child("child", "state")

        assert len(child_post._finish_callbacks) == 0

    def test_child_shares_parent_event_queue(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("main", "root")])

        child_post = post.child("child", "state")

        assert child_post.queue is queue

    def test_post_stores_source_list(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")])

        assert post.source == [("region1", "state1")]

    @pytest.mark.asyncio
    async def test_aforward_includes_source_in_event(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[("region1", "state1")])

        await post.aforward("TestEvent", {"data": "test"})

        event = queue.pop_nowait()
        assert event["source"] == [("region1", "state1")]

    def test_child_from_empty_source_list(self):
        queue = EventQueue()
        post = _event.Post(queue=queue, source=[])

        child_post = post.child("child_name", "child_state")

        assert child_post.source == [("child_name", "child_state")]


@pytest.mark.asyncio
class TestTimer:
    
    async def test_start_creates_timer_and_returns_id(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        timer_id = timer.start("test_tag", 0.001, owner_region="region1", owner_state="state1")
        
        assert timer_id.startswith("timer_")
        assert len(timer._timers) == 1
        assert timer._timers[timer_id]["tag"] == "test_tag"
    
    async def test_timer_fires_and_posts_event(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        timer_id = timer.start("test_tag", 0.001, owner_region="region1", owner_state="state1")
        
        # Wait for timer to fire
        await asyncio.sleep(0.01)
        
        # Check timer event was posted
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "Timer"
        assert event["payload"]["tag"] == "test_tag"
        assert event["payload"]["timer_id"] == timer_id
        assert event["source"] == [("region1", "state1")]
        
        # Timer should be cleaned up
        assert timer_id not in timer._timers
    
    async def test_cancel_stops_timer(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        timer_id = timer.start("test_tag", 0.1, owner_region="region1", owner_state="state1")
        result = timer.cancel(timer_id)
        
        assert result is True
        assert timer_id not in timer._timers
        
        # Wait to ensure timer doesn't fire
        await asyncio.sleep(0.01)
        assert queue.empty()
    
    async def test_cancel_nonexistent_timer_returns_false(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        result = timer.cancel("nonexistent")
        
        assert result is False
    
    async def test_cancel_owned_removes_matching_timers(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        # Start multiple timers with different owners
        timer1 = timer.start("tag1", 0.1, owner_region="region1", owner_state="state1")
        timer2 = timer.start("tag2", 0.1, owner_region="region1", owner_state="state1")
        timer3 = timer.start("tag3", 0.1, owner_region="region2", owner_state="state2")
        
        count = timer.cancel_owned("region1", "state1")
        
        assert count == 2
        assert timer1 not in timer._timers
        assert timer2 not in timer._timers
        assert timer3 in timer._timers
    
    async def test_list_returns_timer_info(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        timer_id = timer.start("test_tag", 1.0, owner_region="region1", owner_state="state1")
        
        timers = timer.list()
        
        assert len(timers) == 1
        assert timers[0]["id"] == timer_id
        assert timers[0]["tag"] == "test_tag"
        assert timers[0]["remaining"] > 0.9  # Should be close to 1.0
    
    async def test_timer_with_custom_payload(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        timer.start("test_tag", 0.001, 
                   owner_region="region1", 
                   owner_state="state1",
                   payload={"custom": "data"})
        
        await asyncio.sleep(0.01)
        
        event = queue.pop_nowait()
        assert event["payload"]["custom"] == "data"
        assert event["payload"]["tag"] == "test_tag"
    
    async def test_timer_with_parent_scope(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        timer.start("test_tag", 0.001, 
                   owner_region="region1", 
                   owner_state="state1",
                   scope="parent")
        
        await asyncio.sleep(0.01)
        
        event = queue.pop_nowait()
        assert event["scope"] == "parent"
    
    async def test_state_dict_captures_timer_metadata(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        # Start some timers
        timer1_id = timer.start("tag1", 1.0, owner_region="region1", owner_state="state1")
        timer2_id = timer.start("tag2", 2.0, owner_region="region2", owner_state="state2")
        
        state = timer.state_dict()
        
        assert state["next_id"] == 2  # Should have incremented
        assert len(state["timer_metadata"]) == 2
        
        # Check timer metadata (should not include 'task')
        timer1_meta = state["timer_metadata"][timer1_id]
        assert timer1_meta["tag"] == "tag1"
        assert timer1_meta["owner_region"] == "region1"
        assert timer1_meta["owner_state"] == "state1"
        assert "task" not in timer1_meta  # Runtime task should be excluded
    
    async def test_load_state_dict_restores_timer_metadata(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        
        # Start a timer to increment next_id
        timer.start("temp", 0.001, owner_region="temp", owner_state="temp")
        
        state = {
            "next_id": 42,
            "timer_metadata": {
                "timer_1": {
                    "tag": "restored_tag",
                    "when": clock.now() + 1.0,
                    "owner_region": "restored_region",
                    "owner_state": "restored_state"
                }
            }
        }
        
        timer.load_state_dict(state)
        
        assert timer._next_id == 42
        assert len(timer._timers) == 0  # Active timers should be cleared