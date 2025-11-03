"""Unit tests for StateChart event system components.

Tests cover EventQueue, Post, Timer, and MonotonicClock classes following the
framework testing conventions.
"""

import asyncio
import pytest
import pytest_asyncio
import time
import sys
import os

from dachi.act._chart import _event
from dachi.act._chart._event import EventQueue, Event, EventPost, Timer, MonotonicClock

Post = EventPost  # Alias for backward compatibility in tests


class TestEventQueue:

    def test_init_creates_queue_with_default_params(self):
        queue = EventQueue()
        assert queue.maxsize == 1024
        assert queue.overflow == "drop_newest"
        assert queue.empty() is True

    def test_init_creates_queue_with_custom_maxsize(self):
        queue = EventQueue(maxsize=512)
        assert queue.maxsize == 512

    def test_init_creates_queue_with_drop_oldest_overflow(self):
        queue = EventQueue(overflow="drop_oldest")
        assert queue.overflow == "drop_oldest"

    def test_init_creates_queue_with_block_overflow(self):
        queue = EventQueue(overflow="block")
        assert queue.overflow == "block"

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

    def test_post_nowait_when_full_block_returns_false(self):
        queue = EventQueue(maxsize=1, overflow="block")
        queue.post_nowait("event1")
        result = queue.post_nowait("event2")
        assert result is False
        assert queue.size() == 1

    def test_post_nowait_with_event_populates_all_fields(self):
        queue = EventQueue()
        event = Event(
            type="test",
            payload={"key": "value"},
            port="output",
            scope="parent",
            source=[("region1", "state1")],
            epoch=42,
            meta={"info": "data"},
            ts=123.45
        )
        queue.post_nowait(event)
        retrieved = queue.pop_nowait()
        assert retrieved["type"] == "test"
        assert retrieved["payload"] == {"key": "value"}
        assert retrieved["port"] == "output"
        assert retrieved["scope"] == "parent"
        assert retrieved["source"] == [("region1", "state1")]
        assert retrieved["epoch"] == 42
        assert retrieved["meta"] == {"info": "data"}
        assert retrieved["ts"] == 123.45

    def test_post_nowait_preserves_event_fields(self):
        queue = EventQueue()
        event = Event(type="original", ts=100.0)
        queue.post_nowait(event)
        retrieved = queue.pop_nowait()
        assert retrieved["type"] == "original"
        assert retrieved["ts"] == 100.0

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

    def test_pop_nowait_maintains_fifo_order_with_multiple_events(self):
        queue = EventQueue()
        queue.post_nowait("first")
        queue.post_nowait("second")
        queue.post_nowait("third")
        assert queue.pop_nowait()["type"] == "first"
        assert queue.pop_nowait()["type"] == "second"
        assert queue.pop_nowait()["type"] == "third"

    def test_size_returns_zero_when_empty(self):
        queue = EventQueue()
        assert queue.size() == 0

    def test_size_returns_correct_count_after_adding_events(self):
        queue = EventQueue()
        queue.post_nowait("event1")
        queue.post_nowait("event2")
        queue.post_nowait("event3")
        assert queue.size() == 3

    def test_size_decreases_after_pop(self):
        queue = EventQueue()
        queue.post_nowait("event1")
        queue.post_nowait("event2")
        assert queue.size() == 2
        queue.pop_nowait()
        assert queue.size() == 1

    def test_empty_returns_true_when_no_events(self):
        queue = EventQueue()
        assert queue.empty() is True

    def test_empty_returns_false_when_has_events(self):
        queue = EventQueue()
        queue.post_nowait("test")
        assert queue.empty() is False

    def test_capacity_returns_maxsize(self):
        queue = EventQueue()
        assert queue.capacity() == 1024

    def test_capacity_returns_custom_maxsize(self):
        queue = EventQueue(maxsize=256)
        assert queue.capacity() == 256

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

    def test_state_dict_with_empty_queue(self):
        queue = EventQueue()
        state = queue.state_dict()
        assert state["events"] == []

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

    def test_load_state_dict_roundtrip_preserves_state(self):
        queue1 = EventQueue(maxsize=128, overflow="drop_oldest")
        queue1.post_nowait("event1")
        queue1.post_nowait("event2")

        state = queue1.state_dict()

        queue2 = EventQueue()
        queue2.load_state_dict(state)

        assert queue2.state_dict() == state


class TestMonotonicClock:

    def test_now_returns_float(self):
        clock = MonotonicClock()
        now = clock.now()
        assert isinstance(now, float)
        assert now > 0

    def test_now_advances_over_time(self):
        clock = MonotonicClock()
        first = clock.now()
        time.sleep(0.001)
        second = clock.now()
        assert second > first

    def test_now_is_monotonic_never_decreases(self):
        clock = MonotonicClock()
        values = [clock.now() for _ in range(100)]
        for i in range(1, len(values)):
            assert values[i] >= values[i-1]


@pytest.mark.asyncio
class TestMonotonicClockAsync:

    async def test_sleep_until_waits_correct_duration(self):
        clock = MonotonicClock()
        start = clock.now()
        target = start + 0.01
        await clock.sleep_until(target)
        elapsed = clock.now() - start
        assert elapsed >= 0.01

    async def test_sleep_until_past_time_returns_immediately(self):
        clock = MonotonicClock()
        start = clock.now()
        past_time = start - 1.0
        await clock.sleep_until(past_time)
        elapsed = clock.now() - start
        assert elapsed < 0.001

    async def test_sleep_until_with_current_time_returns_immediately(self):
        clock = MonotonicClock()
        start = clock.now()
        current_time = clock.now()
        await clock.sleep_until(current_time)
        elapsed = clock.now() - start
        assert elapsed < 0.001


@pytest.mark.asyncio
class TestEventQueueAsync:

    async def test_post_adds_string_event_to_queue(self):
        queue = EventQueue()
        result = await queue.post("test_event")
        assert result is True
        assert queue.size() == 1

    async def test_post_with_event_object_adds_to_queue(self):
        queue = EventQueue()
        event = Event(type="test", ts=time.monotonic())
        result = await queue.post(event)
        assert result is True
        assert queue.size() == 1

    async def test_post_when_full_drop_newest_returns_false(self):
        queue = EventQueue(maxsize=1, overflow="drop_newest")
        await queue.post("event1")
        result = await queue.post("event2")
        assert result is False
        assert queue.size() == 1


class TestPost:

    def test_post_init_sets_preempting_lambda(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        assert callable(post.preempting)
        assert post.preempting() is False

    def test_post_init_with_empty_source(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[])
        assert post.source == []

    def test_post_init_with_source_list(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("region1", "state1")])
        assert post.source == [("region1", "state1")]

    def test_post_init_with_epoch(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, epoch=42)
        assert post.epoch == 42


@pytest.mark.asyncio
class TestPostAforward:

    async def test_aforward_posts_event_with_empty_source(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[])
        result = await post.aforward("TestEvent")
        assert result is None
        event = queue.pop_nowait()
        assert event["source"] == []

    async def test_aforward_posts_event_with_hierarchical_source(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", "s1"), ("r2", "s2")])
        await post.aforward("TestEvent")
        event = queue.pop_nowait()
        assert event["source"] == [("r1", "s1"), ("r2", "s2")]

    async def test_aforward_with_payload_includes_data(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", {"key": "value"})
        event = queue.pop_nowait()
        assert event["payload"] == {"key": "value"}

    async def test_aforward_with_none_payload_creates_empty_dict(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", None)
        event = queue.pop_nowait()
        assert event["payload"] == {}

    async def test_aforward_with_port_includes_port(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", port="output1")
        event = queue.pop_nowait()
        assert event["port"] == "output1"

    async def test_aforward_with_scope_chart_sets_scope(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", scope="chart")
        event = queue.pop_nowait()
        assert event["scope"] == "chart"

    async def test_aforward_with_scope_parent_sets_scope(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", scope="parent")
        event = queue.pop_nowait()
        assert event["scope"] == "parent"

    async def test_aforward_includes_epoch_in_event(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, epoch=42)
        await post.aforward("TestEvent")
        event = queue.pop_nowait()
        assert event["epoch"] == 42

    async def test_aforward_includes_timestamp_in_event(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        before = time.monotonic()
        await post.aforward("TestEvent")
        after = time.monotonic()
        event = queue.pop_nowait()
        assert isinstance(event["ts"], float)
        assert before <= event["ts"] <= after

    async def test_aforward_includes_empty_meta_in_event(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent")
        event = queue.pop_nowait()
        assert event["meta"] == {}

    async def test_aforward_returns_none_when_no_delay(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        result = await post.aforward("TestEvent")
        assert result is None

    async def test_aforward_returns_none_when_queue_full(self):
        queue = EventQueue(maxsize=1, overflow="drop_newest")
        post = _event.EventPost(queue=queue)
        await post.aforward("Event1")
        result = await post.aforward("Event2")
        assert result is None

    async def test_aforward_posts_complete_event_structure(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("region1", "state1")], epoch=42)
        await post.aforward("TestEvent", {"key": "value"}, scope="parent", port="output")
        event = queue.pop_nowait()
        assert event["type"] == "TestEvent"
        assert event["payload"] == {"key": "value"}
        assert event["scope"] == "parent"
        assert event["port"] == "output"
        assert event["source"] == [("region1", "state1")]
        assert event["epoch"] == 42
        assert event["meta"] == {}
        assert isinstance(event["ts"], float)

    async def test_aforward_default_scope_is_chart(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent")
        event = queue.pop_nowait()
        assert event["scope"] == "chart"


class TestPostChild:

    def test_child_adds_region_with_none_state_to_source(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[])
        child = post.child("region1")
        assert child.source == [("region1", None)]

    def test_child_from_empty_source_creates_single_tuple(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[])
        child = post.child("region1")
        assert child.source == [("region1", None)]

    def test_child_from_existing_source_extends_hierarchy(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", "s1")])
        child = post.child("r2")
        assert child.source == [("r1", "s1"), ("r2", None)]

    def test_child_preserves_queue_reference(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        child = post.child("region1")
        assert child.queue is queue

    def test_child_preserves_epoch(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, epoch=42)
        child = post.child("region1")
        assert child.epoch == 42

    def test_child_creates_new_post_instance(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        child = post.child("region1")
        assert child is not post

    def test_child_with_deep_hierarchy_extends_correctly(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", "s1")])
        child1 = post.child("r2")
        child2 = child1.child("r3")
        assert child2.source == [("r1", "s1"), ("r2", None), ("r3", None)]

    def test_child_multiple_children_from_same_parent(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("main", "root")])
        child_a = post.child("child_a")
        child_b = post.child("child_b")
        assert child_a.source == [("main", "root"), ("child_a", None)]
        assert child_b.source == [("main", "root"), ("child_b", None)]


class TestPostSibling:

    def test_sibling_replaces_none_with_state_name(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", None)])
        sibling = post.sibling("state1")
        assert sibling.source == [("r1", "state1")]

    def test_sibling_replaces_existing_state_name(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", "s1")])
        sibling = post.sibling("s2")
        assert sibling.source == [("r1", "s2")]

    def test_sibling_updates_only_last_tuple(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", "s1"), ("r2", None)])
        sibling = post.sibling("state")
        assert sibling.source == [("r1", "s1"), ("r2", "state")]

    def test_sibling_raises_valueerror_when_source_empty(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[])
        with pytest.raises(ValueError, match="Cannot add state without a region in the source"):
            post.sibling("state1")

    def test_sibling_preserves_queue_reference(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", None)])
        sibling = post.sibling("state1")
        assert sibling.queue is queue

    def test_sibling_preserves_epoch(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", None)], epoch=42)
        sibling = post.sibling("state1")
        assert sibling.epoch == 42

    def test_sibling_creates_new_post_instance(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", None)])
        sibling = post.sibling("state1")
        assert sibling is not post


@pytest.mark.asyncio
class TestPostCall:

    async def test_call_delegates_to_aforward(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("r1", "s1")])
        result = await post("TestEvent", {"data": "test"})
        assert result is None
        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "TestEvent"
        assert event["payload"] == {"data": "test"}
        assert event["source"] == [("r1", "s1")]


@pytest.mark.asyncio
class TestPostTimerDelay:

    async def test_aforward_with_delay_returns_timer_id(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=0.01)
        assert timer_id is not None
        assert timer_id.startswith("timer_")

    async def test_aforward_with_delay_zero_returns_none(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=0)
        assert timer_id is None
        assert queue.size() == 1

    async def test_aforward_with_delay_none_returns_none(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=None)
        assert timer_id is None
        assert queue.size() == 1

    async def test_aforward_with_negative_delay_raises_valueerror(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        with pytest.raises(ValueError, match="delay must be >= 0.0"):
            await post.aforward("TestEvent", delay=-1.0)

    async def test_aforward_with_delay_fires_after_duration(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, source=[("region1", "state1")])
        timer_id = await post.aforward("DelayedEvent", {"key": "value"}, delay=0.01)

        assert queue.empty()
        await asyncio.sleep(0.02)

        assert queue.size() == 1
        event = queue.pop_nowait()
        assert event["type"] == "DelayedEvent"
        assert event["payload"] == {"key": "value"}
        assert event["source"] == [("region1", "state1")]
        assert event["meta"]["timer_id"] == timer_id

    async def test_aforward_with_delay_includes_timer_id_in_meta(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=0.01)

        await asyncio.sleep(0.02)

        event = queue.pop_nowait()
        assert "timer_id" in event["meta"]
        assert event["meta"]["timer_id"] == timer_id

    async def test_aforward_with_delay_respects_scope(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", delay=0.01, scope="parent")

        await asyncio.sleep(0.02)

        event = queue.pop_nowait()
        assert event["scope"] == "parent"

    async def test_aforward_with_delay_respects_port(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        await post.aforward("TestEvent", delay=0.01, port="output1")

        await asyncio.sleep(0.02)

        event = queue.pop_nowait()
        assert event["port"] == "output1"

    async def test_aforward_with_delay_includes_epoch(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue, epoch=42)
        await post.aforward("TestEvent", delay=0.01)

        await asyncio.sleep(0.02)

        event = queue.pop_nowait()
        assert event["epoch"] == 42

    async def test_multiple_delayed_events_fire_independently(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        id1 = await post.aforward("Event1", delay=0.01)
        id2 = await post.aforward("Event2", delay=0.01)
        id3 = await post.aforward("Event3", delay=0.01)

        assert id1 != id2 != id3
        await asyncio.sleep(0.02)

        assert queue.size() == 3


@pytest.mark.asyncio
class TestPostTimerCancel:

    async def test_cancel_stops_delayed_event(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=0.1)

        result = post.cancel(timer_id)

        assert result is True
        await asyncio.sleep(0.02)
        assert queue.empty()

    async def test_cancel_nonexistent_timer_returns_false(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        result = post.cancel("nonexistent_timer")

        assert result is False

    async def test_cancel_removes_timer_from_tracking(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=0.1)

        post.cancel(timer_id)

        assert timer_id not in post._timers

    async def test_cancel_already_fired_timer_returns_false(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        timer_id = await post.aforward("TestEvent", delay=0.01)

        await asyncio.sleep(0.02)
        result = post.cancel(timer_id)

        assert result is False

    async def test_cancel_one_timer_does_not_affect_others(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        id1 = await post.aforward("Event1", delay=0.1)
        id2 = await post.aforward("Event2", delay=0.1)

        post.cancel(id1)
        await asyncio.sleep(0.02)

        assert id1 not in post._timers
        assert id2 in post._timers

        post.cancel_all()


@pytest.mark.asyncio
class TestPostTimerCancelAll:

    async def test_cancel_all_stops_all_timers(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        await post.aforward("Event1", delay=0.1)
        await post.aforward("Event2", delay=0.1)
        await post.aforward("Event3", delay=0.1)

        count = post.cancel_all()

        assert count == 3
        await asyncio.sleep(0.02)
        assert queue.empty()

    async def test_cancel_all_clears_timers_dict(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        await post.aforward("Event1", delay=0.1)
        await post.aforward("Event2", delay=0.1)

        post.cancel_all()

        assert len(post._timers) == 0

    async def test_cancel_all_returns_zero_when_no_timers(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        count = post.cancel_all()

        assert count == 0

    async def test_cancel_all_after_some_timers_fired(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        await post.aforward("Event1", delay=0.01)
        await post.aforward("Event2", delay=0.1)
        await post.aforward("Event3", delay=0.1)

        await asyncio.sleep(0.02)

        count = post.cancel_all()

        assert count == 2
        assert queue.size() == 1

    async def test_cancel_all_does_not_affect_child_post_timers(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)
        child = post.child("region1")

        await post.aforward("ParentEvent", delay=0.1)
        await child.aforward("ChildEvent", delay=0.1)

        post.cancel_all()

        assert len(post._timers) == 0
        assert len(child._timers) == 1


@pytest.mark.asyncio
class TestPostTimerEdgeCases:

    async def test_timer_id_increments_correctly(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        id1 = await post.aforward("Event1", delay=0.1)
        id2 = await post.aforward("Event2", delay=0.1)
        id3 = await post.aforward("Event3", delay=0.1)

        assert id1 == "timer_0"
        assert id2 == "timer_1"
        assert id3 == "timer_2"

    async def test_timer_cleanup_after_firing(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        timer_id = await post.aforward("TestEvent", delay=0.01)

        await asyncio.sleep(0.02)

        assert timer_id not in post._timers

    async def test_delay_with_zero_point_zero_returns_none(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        result = await post.aforward("TestEvent", delay=0.0)

        assert result is None
        assert queue.size() == 1

    async def test_very_small_positive_delay_creates_timer(self):
        queue = EventQueue()
        post = _event.EventPost(queue=queue)

        result = await post.aforward("TestEvent", delay=0.001)

        assert result is not None
        assert result.startswith("timer_")


@pytest_asyncio.fixture
async def timer_fixture():
    """Fixture that creates a Timer and cleans it up after the test."""
    queue = EventQueue()
    clock = MonotonicClock()
    timer = _event.Timer(queue=queue, clock=clock)
    yield timer
    timer.clear()
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
class TestTimer:

    async def test_init_creates_empty_timers_dict(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        assert timer._timers == {}
        timer.clear()

    async def test_init_sets_next_id_to_zero(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        assert timer._next_id == 0

    async def test_init_stores_queue_reference(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        assert timer.queue is queue

    async def test_init_stores_clock_reference(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)
        assert timer.clock is clock

    async def test_start_creates_timer_and_returns_id(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer_id = timer.start("test_tag", 0.001, owner_region="region1", owner_state="state1")

        assert timer_id.startswith("timer_")
        assert len(timer._timers) == 1
        assert timer._timers[timer_id]["tag"] == "test_tag"

    async def test_start_increments_next_id_on_each_call(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        id1 = timer.start("tag1", 0.1, owner_region="r1", owner_state="s1")
        id2 = timer.start("tag2", 0.1, owner_region="r2", owner_state="s2")
        id3 = timer.start("tag3", 0.1, owner_region="r3", owner_state="s3")

        assert id1 == "timer_0"
        assert id2 == "timer_1"
        assert id3 == "timer_2"

        timer.clear()

    async def test_start_with_none_owner_creates_empty_source(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer.start("test_tag", 0.001, owner_region=None, owner_state=None)
        await asyncio.sleep(0.01)

        event = queue.pop_nowait()
        assert event["source"] == []

    async def test_start_with_default_scope_uses_chart(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer.start("test_tag", 0.001, owner_region="r1", owner_state="s1")
        await asyncio.sleep(0.01)

        event = queue.pop_nowait()
        assert event["scope"] == "chart"

    async def test_start_without_payload_creates_event_with_tag_only(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer_id = timer.start("test_tag", 0.001, owner_region="r1", owner_state="s1")
        await asyncio.sleep(0.01)

        event = queue.pop_nowait()
        assert "tag" in event["payload"]
        assert "timer_id" in event["payload"]
        assert len(event["payload"]) == 2
    
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

        timer.clear()
    
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

        timer.clear()
    
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

        timer.clear()
    
    async def test_load_state_dict_restores_timer_metadata(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

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
        assert len(timer._timers) == 0

    async def test_cancel_owned_returns_zero_when_no_matches(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer.start("tag1", 0.1, owner_region="region1", owner_state="state1")
        count = timer.cancel_owned("region2", "state2")

        assert count == 0

        timer.clear()

    async def test_cancel_owned_with_partial_owner_match_does_not_cancel(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer1 = timer.start("tag1", 0.1, owner_region="region1", owner_state="state1")
        timer2 = timer.start("tag2", 0.1, owner_region="region1", owner_state="state2")

        count = timer.cancel_owned("region1", "state1")

        assert count == 1
        assert timer1 not in timer._timers
        assert timer2 in timer._timers

        timer.clear()

    async def test_list_returns_empty_list_when_no_timers(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timers = timer.list()

        assert timers == []

    async def test_list_calculates_remaining_time_correctly(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer_id = timer.start("test_tag", 0.5, owner_region="r1", owner_state="s1")
        await asyncio.sleep(0.01)

        timers = timer.list()
        assert 0.4 < timers[0]["remaining"] < 0.5

        timer.clear()

    async def test_state_dict_with_no_active_timers(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        state = timer.state_dict()

        assert state["next_id"] == 0
        assert state["timer_metadata"] == {}

    async def test_snapshot_returns_empty_list(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        result = timer.snapshot()

        assert result == []

    async def test_restore_accepts_list_and_does_nothing(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer.restore([{"some": "data"}])

    async def test_multiple_timers_with_same_tag_different_owners(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        id1 = timer.start("same_tag", 0.001, owner_region="r1", owner_state="s1")
        id2 = timer.start("same_tag", 0.001, owner_region="r2", owner_state="s2")

        await asyncio.sleep(0.01)

        assert queue.size() == 2
        event1 = queue.pop_nowait()
        event2 = queue.pop_nowait()
        assert event1["payload"]["tag"] == "same_tag"
        assert event2["payload"]["tag"] == "same_tag"
        assert event1["source"] != event2["source"]

    async def test_timer_fires_after_cancel_does_not_post_event(self):
        queue = EventQueue()
        clock = MonotonicClock()
        timer = _event.Timer(queue=queue, clock=clock)

        timer_id = timer.start("test_tag", 0.001, owner_region="r1", owner_state="s1")
        timer.cancel(timer_id)

        await asyncio.sleep(0.01)

        assert queue.empty()
