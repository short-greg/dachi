from __future__ import annotations

# 1st party
import threading
import time
from typing import List
from unittest.mock import Mock

# 3rd party
import pytest
from pydantic import BaseModel

# local
from dachi.utils import Bulletin, Post, singleton


class Essay(BaseModel):
    title: str
    content: str
    author: str = "Anonymous"
    priority: int = 1


class Task(BaseModel):
    name: str
    priority: int = 1
    completed: bool = False
    urgency: int = 1
    importance: int = 1


class Message(BaseModel):
    text: str
    sender: str
    timestamp: float = 0.0
    priority: int = 1


@singleton
class EssayBulletin(Bulletin[Essay]):
    pass


class TestBulletin:
    
    def test_bulletin_creation(self):
        bulletin = Bulletin[Essay]()
        assert isinstance(bulletin, Bulletin)
        assert bulletin.get_stats()["total_posts"] == 0
    
    def test_publish_item(self):
        bulletin = Bulletin[Essay]()
        essay = Essay(title="Test Essay", content="This is a test")
        
        post_id = bulletin.publish(essay)
        assert isinstance(post_id, str)
        assert len(post_id) > 0
        
        stats = bulletin.get_stats()
        assert stats["total_posts"] == 1
        assert stats["locked_posts"] == 1
    
    def test_publish_non_basemodel_fails(self):
        bulletin = Bulletin[Essay]()
        
        with pytest.raises(TypeError, match="Published items must be BaseModel instances"):
            bulletin.publish("not a basemodel")
    
    def test_retrieve_first_by_id(self):
        bulletin = Bulletin[Essay]()
        essay = Essay(title="Test Essay", content="This is a test")
        
        post_id = bulletin.publish(essay)
        retrieved = bulletin.retrieve_first(id=post_id)
        
        assert retrieved is not None
        assert retrieved["id"] == post_id
        assert retrieved["item"] == essay
        assert retrieved["locked"] == True
        assert "timestamp" in retrieved
        assert "expires_at" in retrieved
    
    def test_retrieve_first_by_filter(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="Python Guide", content="Learn Python")
        essay2 = Essay(title="Java Tutorial", content="Learn Java")
        
        bulletin.publish(essay1)
        bulletin.publish(essay2)
        
        retrieved = bulletin.retrieve_first(filter_func=lambda item: "Python" in item.title)
        assert retrieved is not None
        assert retrieved["item"].title == "Python Guide"
    
    def test_retrieve_first_by_type(self):
        bulletin = Bulletin()
        essay = Essay(title="Essay", content="Content")
        task = Task(name="Complete work")
        
        bulletin.publish(essay)
        bulletin.publish(task)
        
        retrieved = bulletin.retrieve_first(item_type=Task)
        assert retrieved is not None
        assert retrieved["item"].name == "Complete work"
        assert isinstance(retrieved["item"], Task)
    
    def test_retrieve_all(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="First", content="Content 1")
        essay2 = Essay(title="Second", content="Content 2")
        essay3 = Essay(title="Third", content="Content 3")
        
        bulletin.publish(essay1)
        bulletin.publish(essay2)
        bulletin.publish(essay3)
        
        all_posts = bulletin.retrieve_all()
        assert len(all_posts) == 3
        titles = {post["item"].title for post in all_posts}
        assert titles == {"First", "Second", "Third"}
    
    def test_retrieve_all_with_filter(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="Python Guide", content="Learn Python")
        essay2 = Essay(title="Java Tutorial", content="Learn Java")
        essay3 = Essay(title="Python Advanced", content="Advanced Python")
        
        bulletin.publish(essay1)
        bulletin.publish(essay2)
        bulletin.publish(essay3)
        
        python_posts = bulletin.retrieve_all(filter_func=lambda item: "Python" in item.title)
        assert len(python_posts) == 2
        titles = {post["item"].title for post in python_posts}
        assert titles == {"Python Guide", "Python Advanced"}
    
    def test_retrieve_all_by_type(self):
        bulletin = Bulletin()
        essay = Essay(title="Essay", content="Content")
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2")
        
        bulletin.publish(essay)
        bulletin.publish(task1)
        bulletin.publish(task2)
        
        task_posts = bulletin.retrieve_all(item_type=Task)
        assert len(task_posts) == 2
        task_names = {post["item"].name for post in task_posts}
        assert task_names == {"Task 1", "Task 2"}
    
    def test_retrieve_all_exclude_locked(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="Locked", content="Content 1")
        essay2 = Essay(title="Unlocked", content="Content 2")
        
        id1 = bulletin.publish(essay1, lock=True)
        id2 = bulletin.publish(essay2, lock=False)
        
        unlocked_posts = bulletin.retrieve_all(include_locked=False)
        assert len(unlocked_posts) == 1
        assert unlocked_posts[0]["item"].title == "Unlocked"
    
    def test_release_post(self):
        bulletin = Bulletin[Essay]()
        essay = Essay(title="Test", content="Content")
        
        post_id = bulletin.publish(essay, lock=True)
        stats = bulletin.get_stats()
        assert stats["locked_posts"] == 1
        
        success = bulletin.release(post_id)
        assert success == True
        
        stats = bulletin.get_stats()
        assert stats["locked_posts"] == 0
        assert stats["available_posts"] == 1
    
    def test_publish_with_timestamp_and_ttl(self):
        bulletin = Bulletin[Essay]()
        essay = Essay(title="Test Essay", content="This is a test")
        
        before_time = time.time()
        post_id = bulletin.publish(essay, ttl=2.0)
        after_time = time.time()
        
        post = bulletin.retrieve_first(id=post_id)
        assert post is not None
        assert before_time <= post["timestamp"] <= after_time
        assert post["expires_at"] is not None
        assert post["expires_at"] > before_time
    
    def test_release_nonexistent_post(self):
        bulletin = Bulletin[Essay]()
        success = bulletin.release("nonexistent-id")
        assert success == False
    
    def test_remove_post(self):
        bulletin = Bulletin[Essay]()
        essay = Essay(title="Test", content="Content")
        
        post_id = bulletin.publish(essay)
        assert bulletin.get_stats()["total_posts"] == 1
        
        success = bulletin.remove(post_id)
        assert success == True
        assert bulletin.get_stats()["total_posts"] == 0
        
        retrieved = bulletin.retrieve_first(id=post_id)
        assert retrieved is None
    
    def test_retrieve_first_with_ordering(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="Low Priority", content="Content", priority=1)
        essay2 = Essay(title="High Priority", content="Content", priority=10)
        essay3 = Essay(title="Medium Priority", content="Content", priority=5)
        
        bulletin.publish(essay1)
        time.sleep(0.01)
        bulletin.publish(essay2)
        time.sleep(0.01) 
        bulletin.publish(essay3)
        
        # Get highest priority first
        retrieved = bulletin.retrieve_first(order_func=lambda item: -item.priority)
        assert retrieved is not None
        assert retrieved["item"].title == "High Priority"
    
    def test_remove_nonexistent_post(self):
        bulletin = Bulletin[Essay]()
        success = bulletin.remove("nonexistent-id")
        assert success == False
    
    def test_clear_bulletin(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="First", content="Content 1")
        essay2 = Essay(title="Second", content="Content 2")
        
        bulletin.publish(essay1)
        bulletin.publish(essay2)
        assert bulletin.get_stats()["total_posts"] == 2
        
        cleared_count = bulletin.clear()
        assert cleared_count == 2
        assert bulletin.get_stats()["total_posts"] == 0
    
    def test_stats_with_mixed_types(self):
        bulletin = Bulletin()
        essay = Essay(title="Essay", content="Content")
        task = Task(name="Task")
        message = Message(text="Hello", sender="User")
        
        bulletin.publish(essay, lock=True)
        bulletin.publish(task, lock=False)
        bulletin.publish(message, lock=True)
        
        stats = bulletin.get_stats()
        assert stats["total_posts"] == 3
        assert stats["locked_posts"] == 2
        assert stats["available_posts"] == 1
        assert Essay in stats["type_breakdown"]
        assert Task in stats["type_breakdown"]
        assert Message in stats["type_breakdown"]
    
    def test_thread_safety(self):
        bulletin = Bulletin[Essay]()
        results = []
        errors = []
        
        def publisher_thread(thread_id: int):
            try:
                for i in range(10):
                    essay = Essay(title=f"Essay {thread_id}-{i}", content=f"Content from thread {thread_id}")
                    post_id = bulletin.publish(essay)
                    results.append(post_id)
            except Exception as e:
                errors.append(e)
        
        def retriever_thread():
            try:
                for _ in range(50):
                    posts = bulletin.retrieve_all()
                    if posts:
                        bulletin.release(posts[0]["id"])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=publisher_thread, args=(i,))
            threads.append(t)
        
        for i in range(2):
            t = threading.Thread(target=retriever_thread)
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 50
        assert bulletin.get_stats()["total_posts"] == 50

    def test_expiration_cleanup(self):
        bulletin = Bulletin[Essay]()
        essay = Essay(title="Expiring", content="Will expire soon")
        
        # Publish with very short TTL
        post_id = bulletin.publish(essay, ttl=0.05)
        assert bulletin.get_stats()["total_posts"] == 1
        
        # Wait for expiration
        time.sleep(0.1)
        
        # Trigger cleanup by calling retrieve_first
        bulletin.retrieve_first()
        assert bulletin.get_stats()["total_posts"] == 0

    def test_retrieve_all_with_ordering(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="Low", content="Content", priority=1)
        essay2 = Essay(title="High", content="Content", priority=10)
        essay3 = Essay(title="Medium", content="Content", priority=5)
        
        bulletin.publish(essay1)
        bulletin.publish(essay2)
        bulletin.publish(essay3)
        
        # Get all ordered by priority (descending)
        all_posts = bulletin.retrieve_all(order_func=lambda item: -item.priority)
        assert len(all_posts) == 3
        titles = [post["item"].title for post in all_posts]
        assert titles == ["High", "Medium", "Low"]

    def test_retrieve_first_default_fifo_ordering(self):
        bulletin = Bulletin[Essay]()
        essay1 = Essay(title="First", content="Content")
        essay2 = Essay(title="Second", content="Content")
        
        bulletin.publish(essay1)
        time.sleep(0.01)
        bulletin.publish(essay2)
        
        # Should get first published (FIFO)
        retrieved = bulletin.retrieve_first()
        assert retrieved is not None
        assert retrieved["item"].title == "First"


class TestBulletin_Callbacks:
    
    def test_register_callback(self):
        bulletin = Bulletin[Essay]()
        callback = Mock()
        
        success = bulletin.register_callback(callback)
        assert success == True
        
        # Registering same callback again should return False
        success = bulletin.register_callback(callback)
        assert success == False
    
    def test_unregister_callback(self):
        bulletin = Bulletin[Essay]()
        callback = Mock()
        
        bulletin.register_callback(callback)
        success = bulletin.unregister_callback(callback)
        assert success == True
        
        # Unregistering non-existent callback should return False
        success = bulletin.unregister_callback(callback)
        assert success == False
    
    def test_publish_callback_triggered(self):
        bulletin = Bulletin[Essay]()
        callback = Mock()
        bulletin.register_callback(callback)
        
        essay = Essay(title="Test", content="Content")
        post_id = bulletin.publish(essay)
        
        callback.assert_called_once()
        call_args = callback.call_args[0]
        post, event = call_args
        assert post["id"] == post_id
        assert post["item"] == essay
        assert event == bulletin.ON_PUBLISH
    
    def test_release_callback_triggered(self):
        bulletin = Bulletin[Essay]()
        callback = Mock()
        bulletin.register_callback(callback)
        
        essay = Essay(title="Test", content="Content")
        post_id = bulletin.publish(essay, lock=True)
        
        bulletin.release(post_id)
        
        # Should be called twice: once for publish, once for release
        assert callback.call_count == 2
        release_call = callback.call_args_list[1][0]
        post, event = release_call
        assert post["id"] == post_id
        assert post["locked"] == False
        assert event == bulletin.ON_RELEASE
    
    def test_remove_callback_triggered(self):
        bulletin = Bulletin[Essay]()
        callback = Mock()
        bulletin.register_callback(callback)
        
        essay = Essay(title="Test", content="Content")
        post_id = bulletin.publish(essay)
        
        bulletin.remove(post_id)
        
        # Should be called twice: once for publish, once for remove
        assert callback.call_count == 2
        remove_call = callback.call_args_list[1][0]
        post, event = remove_call
        assert post["id"] == post_id
        assert event == bulletin.ON_REMOVE
    
    def test_expire_callback_triggered(self):
        bulletin = Bulletin[Essay]()
        callback = Mock()
        bulletin.register_callback(callback)
        
        essay = Essay(title="Expiring", content="Will expire")
        post_id = bulletin.publish(essay, ttl=0.05)
        
        # Wait for expiration
        time.sleep(0.1)
        
        # Trigger cleanup
        bulletin.retrieve_first()
        
        # Should be called twice: once for publish, once for expire
        assert callback.call_count == 2
        expire_call = callback.call_args_list[1][0]
        post, event = expire_call
        assert post["id"] == post_id
        assert event == bulletin.ON_EXPIRE
    
    def test_callback_error_handling(self):
        bulletin = Bulletin[Essay]()
        
        def failing_callback(post, event):
            raise Exception("Callback error")
        
        bulletin.register_callback(failing_callback)
        
        essay = Essay(title="Test", content="Content")
        # Should not raise exception even though callback fails
        post_id = bulletin.publish(essay)
        assert post_id is not None
    
    def test_multiple_callbacks(self):
        bulletin = Bulletin[Essay]()
        callback1 = Mock()
        callback2 = Mock()
        
        bulletin.register_callback(callback1)
        bulletin.register_callback(callback2)
        
        essay = Essay(title="Test", content="Content")
        bulletin.publish(essay)
        
        callback1.assert_called_once()
        callback2.assert_called_once()


class TestBulletin_ComplexScenarios:
    
    def test_priority_queue_simulation(self):
        bulletin = Bulletin[Task]()
        
        # Add tasks with different priorities
        tasks = [
            Task(name="Critical bug fix", urgency=10, importance=10),
            Task(name="Minor feature", urgency=2, importance=3),
            Task(name="Documentation", urgency=1, importance=5),
            Task(name="Performance optimization", urgency=7, importance=8),
        ]
        
        for task in tasks:
            bulletin.publish(task, lock=False)
        
        # Retrieve tasks by priority score (urgency * importance)
        priority_tasks = bulletin.retrieve_all(
            order_func=lambda item: -(item.urgency * item.importance)
        )
        
        assert priority_tasks[0]["item"].name == "Critical bug fix"  # 100
        assert priority_tasks[1]["item"].name == "Performance optimization"  # 56
    
    def test_work_queue_with_expiration(self):
        bulletin = Bulletin[Task]()
        callback = Mock()
        
        bulletin.register_callback(callback)
        
        # Add work items with different expiration times
        task1 = Task(name="Quick task", urgency=5)
        task2 = Task(name="Slow task", urgency=3)
        
        bulletin.publish(task1, ttl=0.1)  # Expires quickly
        bulletin.publish(task2, ttl=1.0)  # Expires later
        
        # Should be called twice for publishing
        assert callback.call_count == 2
        
        # Wait for first task to expire
        time.sleep(0.15)
        
        # Trigger cleanup
        remaining = bulletin.retrieve_all()
        
        assert len(remaining) == 1
        assert remaining[0]["item"].name == "Slow task"
        
        # Should now be called 3 times: 2 publishes + 1 expire
        assert callback.call_count == 3
        
        # Check that the last call was an expire event
        last_call = callback.call_args_list[-1][0]
        post, event = last_call
        assert event == bulletin.ON_EXPIRE


class TestSingletonBulletin:
    
    def test_singleton_bulletin_creation(self):
        bulletin1 = EssayBulletin.obj
        bulletin2 = EssayBulletin.obj
        assert bulletin1 is bulletin2
    
    def test_singleton_bulletin_functionality(self):
        bulletin = EssayBulletin.obj
        bulletin.clear()  # Clear any existing posts
        
        essay = Essay(title="Singleton Test", content="Testing singleton pattern")
        post_id = bulletin.publish(essay)
        
        retrieved = bulletin.retrieve_first(id=post_id)
        assert retrieved is not None
        assert retrieved["item"].title == "Singleton Test"
    
    def test_singleton_direct_instantiation_fails(self):
        with pytest.raises(TypeError, match="EssayBulletin is a singleton"):
            EssayBulletin()


class TestGenericBulletinUsage:
    
    def test_multiple_bulletin_types(self):
        essay_bulletin = Bulletin[Essay]()
        task_bulletin = Bulletin[Task]()
        
        essay = Essay(title="Essay", content="Content")
        task = Task(name="Task")
        
        essay_id = essay_bulletin.publish(essay)
        task_id = task_bulletin.publish(task)
        
        essay_post = essay_bulletin.retrieve_first(id=essay_id)
        task_post = task_bulletin.retrieve_first(id=task_id)
        
        assert isinstance(essay_post["item"], Essay)
        assert isinstance(task_post["item"], Task)
        
        # Cross-bulletin retrieval should return None
        assert essay_bulletin.retrieve_first(id=task_id) is None
        assert task_bulletin.retrieve_first(id=essay_id) is None
    
    def test_untyped_bulletin_accepts_any_basemodel(self):
        bulletin = Bulletin()
        
        essay = Essay(title="Essay", content="Content")
        task = Task(name="Task")
        message = Message(text="Hello", sender="User")
        
        essay_id = bulletin.publish(essay)
        task_id = bulletin.publish(task)
        message_id = bulletin.publish(message)
        
        assert bulletin.retrieve_first(id=essay_id)["item"] == essay
        assert bulletin.retrieve_first(id=task_id)["item"] == task
        assert bulletin.retrieve_first(id=message_id)["item"] == message