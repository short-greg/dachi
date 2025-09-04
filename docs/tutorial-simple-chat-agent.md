# Tutorial: Simple Chat Agent

This tutorial shows how to build a simple chat agent using Dachi's AsyncDispatcher for AI processing and Blackboard for conversation state management. You'll learn how to create a responsive chat interface that maintains conversation context and handles errors gracefully.

## Overview

We'll build a chat agent with the following features:

- Maintains conversation history using Blackboard
- Processes AI requests asynchronously with AsyncDispatcher
- Handles multiple concurrent conversations
- Provides both synchronous and asynchronous interfaces
- Includes error handling and recovery

## Prerequisites

Make sure you have:
- Dachi installed with OpenAI adapter support
- OpenAI API key configured
- Basic familiarity with async/await patterns

## Step 1: Basic Chat Agent Structure

Let's start with a simple chat agent that can handle single conversations:

```python
# chat_agent.py
from __future__ import annotations

import time
import asyncio
from typing import Optional, Dict, Any

from dachi.comm import Blackboard, AsyncDispatcher, RequestStatus
from dachi.proc import OpenAIChat
from dachi.utils import Msg

class SimpleChatAgent:
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        # Initialize communication components
        self.blackboard = Blackboard()
        self.dispatcher = AsyncDispatcher(max_concurrent_requests=3)
        
        # Initialize AI processor
        self.ai_processor = OpenAIChat(
            model=model,
            temperature=temperature
        )
        
        # Initialize conversation state
        self.blackboard.set("conversation_history", [])
        self.blackboard.set("agent_status", "ready")
        self.blackboard.set("total_messages", 0)
    
    def chat(self, user_message: str, timeout: float = 30.0) -> str:
        """
        Process user message and return AI response.
        This is a synchronous interface that waits for completion.
        """
        # Update conversation history
        history = self.blackboard.get("conversation_history", [])
        history.append({"role": "user", "content": user_message})
        self.blackboard.set("conversation_history", history)
        self.blackboard.set("agent_status", "processing")
        
        # Build context for AI
        context = self._build_context(history)
        message = Msg(
            content=user_message,
            context=context
        )
        
        # Submit async AI request
        request_id = self.dispatcher.submit_proc(
            self.ai_processor,
            message,
            callback_id=f"chat_msg_{len(history)}"
        )
        
        try:
            # Wait for completion
            result = self._wait_for_result(request_id, timeout)
            
            # Update conversation with AI response
            history.append({"role": "assistant", "content": result.content})
            self.blackboard.set("conversation_history", history)
            self.blackboard.set("agent_status", "ready")
            
            # Update statistics
            total_messages = self.blackboard.get("total_messages", 0) + 1
            self.blackboard.set("total_messages", total_messages)
            
            return result.content
            
        except Exception as e:
            self.blackboard.set("agent_status", "error")
            self.blackboard.set("last_error", str(e))
            raise
    
    def _build_context(self, history: list) -> dict:
        """Build context information for the AI request"""
        return {
            "conversation_length": len(history),
            "total_messages": self.blackboard.get("total_messages", 0),
            "recent_topics": self._extract_recent_topics(history[-5:])  # Last 5 messages
        }
    
    def _extract_recent_topics(self, recent_history: list) -> list:
        """Extract topics from recent conversation (simplified)"""
        topics = []
        for msg in recent_history:
            if msg["role"] == "user":
                # Simple keyword extraction (in real app, use NLP)
                words = msg["content"].lower().split()
                topics.extend([w for w in words if len(w) > 4])
        return list(set(topics))[:5]  # Return unique topics, max 5
    
    def _wait_for_result(self, request_id: str, timeout: float):
        """Wait for async request to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.dispatcher.status(request_id)
            if status == RequestStatus.DONE:
                return self.dispatcher.result(request_id)
            elif status == RequestStatus.ERROR:
                error = self.dispatcher.result(request_id)
                raise Exception(f"AI request failed: {error}")
            elif status == RequestStatus.CANCELLED:
                raise Exception("AI request was cancelled")
            
            time.sleep(0.1)  # Poll every 100ms
        
        # Cancel the request if it times out
        self.dispatcher.cancel(request_id)
        raise TimeoutError(f"AI request timed out after {timeout} seconds")
    
    def get_conversation_history(self) -> list:
        """Get the current conversation history"""
        return self.blackboard.get("conversation_history", [])
    
    def get_status(self) -> dict:
        """Get current agent status and statistics"""
        return {
            "status": self.blackboard.get("agent_status"),
            "total_messages": self.blackboard.get("total_messages", 0),
            "conversation_length": len(self.blackboard.get("conversation_history", [])),
            "last_error": self.blackboard.get("last_error")
        }
    
    def clear_conversation(self):
        """Clear the current conversation"""
        self.blackboard.set("conversation_history", [])
        self.blackboard.set("agent_status", "ready")
        self.blackboard.delete("last_error")
```

## Step 2: Testing the Basic Agent

Create a simple test script to verify the chat agent works:

```python
# test_chat_agent.py
from chat_agent import SimpleChatAgent

def main():
    # Create chat agent
    agent = SimpleChatAgent(model="gpt-4", temperature=0.7)
    
    print("Chat Agent Ready! Type 'quit' to exit.")
    print("=" * 40)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Get AI response
            print("AI is thinking...")
            response = agent.chat(user_input)
            print(f"AI: {response}")
            
            # Show status
            status = agent.get_status()
            print(f"[Messages: {status['total_messages']}, Status: {status['status']}]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Agent status:", agent.get_status())

if __name__ == "__main__":
    main()
```

## Step 3: Multi-Conversation Agent

Now let's extend our agent to handle multiple conversations simultaneously:

```python
# multi_chat_agent.py
from __future__ import annotations

import uuid
from typing import Dict, Optional, Callable
from chat_agent import SimpleChatAgent

class MultiChatAgent:
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        # Shared components
        self.dispatcher = AsyncDispatcher(max_concurrent_requests=10)
        self.ai_processor = OpenAIChat(model=model, temperature=temperature)
        
        # Per-conversation blackboards (scoped by conversation ID)
        self.conversations: Dict[str, Blackboard] = {}
        
        # Global statistics blackboard
        self.stats_board = Blackboard()
        self.stats_board.set("total_conversations", 0)
        self.stats_board.set("total_messages", 0)
        self.stats_board.set("active_conversations", 0)
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation and return its ID"""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        # Create scoped blackboard for this conversation
        conversation_board = Blackboard(scope=conversation_id)
        conversation_board.set("history", [])
        conversation_board.set("status", "ready")
        conversation_board.set("created_at", time.time())
        conversation_board.set("message_count", 0)
        
        self.conversations[conversation_id] = conversation_board
        
        # Update global stats
        total_convos = self.stats_board.get("total_conversations", 0) + 1
        active_convos = self.stats_board.get("active_conversations", 0) + 1
        self.stats_board.set("total_conversations", total_convos)
        self.stats_board.set("active_conversations", active_convos)
        
        return conversation_id
    
    def chat(self, conversation_id: str, user_message: str, timeout: float = 30.0) -> str:
        """Send a message in a specific conversation"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        board = self.conversations[conversation_id]
        
        # Update conversation history
        history = board.get("history", [])
        history.append({"role": "user", "content": user_message})
        board.set("history", history)
        board.set("status", "processing")
        
        # Build context
        context = self._build_context(conversation_id, history)
        message = Msg(content=user_message, context=context)
        
        # Submit async request
        request_id = self.dispatcher.submit_proc(
            self.ai_processor,
            message,
            callback_id=f"conv_{conversation_id}_msg_{len(history)}"
        )
        
        try:
            # Wait for completion
            result = self._wait_for_result(request_id, timeout)
            
            # Update conversation with response
            history.append({"role": "assistant", "content": result.content})
            board.set("history", history)
            board.set("status", "ready")
            
            # Update statistics
            msg_count = board.get("message_count", 0) + 1
            board.set("message_count", msg_count)
            
            total_msgs = self.stats_board.get("total_messages", 0) + 1
            self.stats_board.set("total_messages", total_msgs)
            
            return result.content
            
        except Exception as e:
            board.set("status", "error")
            board.set("last_error", str(e))
            raise
    
    def chat_async(self, conversation_id: str, user_message: str, 
                   callback: Callable[[str, str], None]) -> str:
        """
        Send a message asynchronously and call callback when complete.
        Returns request_id for tracking.
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        board = self.conversations[conversation_id]
        
        # Update conversation history
        history = board.get("history", [])
        history.append({"role": "user", "content": user_message})
        board.set("history", history)
        board.set("status", "processing")
        
        # Build context and message
        context = self._build_context(conversation_id, history)
        message = Msg(content=user_message, context=context)
        
        # Define completion callback
        def on_completion(request_id: str, result: Any, status: RequestStatus):
            if status == RequestStatus.DONE:
                # Update conversation with response
                history = board.get("history", [])
                history.append({"role": "assistant", "content": result.content})
                board.set("history", history)
                board.set("status", "ready")
                
                # Update statistics
                msg_count = board.get("message_count", 0) + 1
                board.set("message_count", msg_count)
                
                total_msgs = self.stats_board.get("total_messages", 0) + 1
                self.stats_board.set("total_messages", total_msgs)
                
                # Call user callback
                callback(conversation_id, result.content)
                
            elif status == RequestStatus.ERROR:
                board.set("status", "error")
                board.set("last_error", str(result))
                callback(conversation_id, f"Error: {result}")
        
        # Submit async request with callback
        request_id = self.dispatcher.submit_proc(
            self.ai_processor,
            message,
            callback=on_completion,
            callback_id=f"async_conv_{conversation_id}_msg_{len(history)}"
        )
        
        return request_id
    
    def get_conversation_history(self, conversation_id: str) -> list:
        """Get conversation history"""
        if conversation_id not in self.conversations:
            return []
        return self.conversations[conversation_id].get("history", [])
    
    def get_conversation_status(self, conversation_id: str) -> dict:
        """Get status for a specific conversation"""
        if conversation_id not in self.conversations:
            return {"error": "Conversation not found"}
        
        board = self.conversations[conversation_id]
        return {
            "id": conversation_id,
            "status": board.get("status"),
            "message_count": board.get("message_count", 0),
            "created_at": board.get("created_at"),
            "last_error": board.get("last_error")
        }
    
    def list_conversations(self) -> list:
        """List all active conversations"""
        return [
            self.get_conversation_status(conv_id) 
            for conv_id in self.conversations.keys()
        ]
    
    def get_global_stats(self) -> dict:
        """Get global statistics"""
        return {
            "total_conversations": self.stats_board.get("total_conversations", 0),
            "active_conversations": len(self.conversations),
            "total_messages": self.stats_board.get("total_messages", 0),
            "dispatcher_stats": {
                "max_concurrent": self.dispatcher.max_concurrent_requests,
                "active_requests": len([
                    req for req in self.dispatcher.requests.values() 
                    if not req.status.is_complete()
                ])
            }
        }
    
    def close_conversation(self, conversation_id: str):
        """Close and remove a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            active_convos = self.stats_board.get("active_conversations", 1) - 1
            self.stats_board.set("active_conversations", max(0, active_convos))
    
    def _build_context(self, conversation_id: str, history: list) -> dict:
        """Build context for AI request"""
        board = self.conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "conversation_length": len(history),
            "message_count": board.get("message_count", 0),
            "created_at": board.get("created_at"),
            "global_stats": self.get_global_stats()
        }
    
    def _wait_for_result(self, request_id: str, timeout: float):
        """Wait for async request to complete (same as SimpleChatAgent)"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.dispatcher.status(request_id)
            if status == RequestStatus.DONE:
                return self.dispatcher.result(request_id)
            elif status == RequestStatus.ERROR:
                error = self.dispatcher.result(request_id)
                raise Exception(f"AI request failed: {error}")
            elif status == RequestStatus.CANCELLED:
                raise Exception("AI request was cancelled")
            
            time.sleep(0.1)
        
        self.dispatcher.cancel(request_id)
        raise TimeoutError(f"AI request timed out after {timeout} seconds")
```

## Step 4: Testing Multi-Conversation Agent

Create a test script for the multi-conversation agent:

```python
# test_multi_chat.py
from multi_chat_agent import MultiChatAgent
import threading
import time

def test_concurrent_conversations():
    """Test multiple conversations running concurrently"""
    agent = MultiChatAgent()
    
    # Create multiple conversations
    conv1 = agent.create_conversation()
    conv2 = agent.create_conversation()
    conv3 = agent.create_conversation()
    
    print(f"Created conversations: {conv1[:8]}, {conv2[:8]}, {conv3[:8]}")
    
    # Define callback for async messages
    responses = {}
    def handle_response(conv_id: str, response: str):
        responses[conv_id] = response
        print(f"[{conv_id[:8]}] AI: {response[:50]}...")
    
    # Send messages to all conversations asynchronously
    req1 = agent.chat_async(conv1, "Tell me about artificial intelligence", handle_response)
    req2 = agent.chat_async(conv2, "Explain quantum physics", handle_response)
    req3 = agent.chat_async(conv3, "What is the weather like?", handle_response)
    
    print("Sent async messages to all conversations...")
    
    # Wait for all responses
    start_time = time.time()
    while len(responses) < 3 and time.time() - start_time < 30:
        time.sleep(0.5)
        print(f"Waiting... ({len(responses)}/3 responses received)")
    
    # Show results
    print("\nFinal Results:")
    for conv_id in [conv1, conv2, conv3]:
        status = agent.get_conversation_status(conv_id)
        print(f"Conversation {conv_id[:8]}: {status['message_count']} messages, {status['status']}")
    
    print("\nGlobal Stats:", agent.get_global_stats())

def test_interactive_multi_chat():
    """Interactive test with multiple conversations"""
    agent = MultiChatAgent()
    conversations = {}
    
    print("Multi-Chat Agent Ready!")
    print("Commands: 'new' (new conversation), 'list' (list conversations), 'switch <id>' (switch conversation), 'quit' (exit)")
    print("=" * 60)
    
    current_conv = None
    
    while True:
        try:
            if current_conv:
                prompt = f"[{current_conv[:8]}] You: "
            else:
                prompt = "Command: "
            
            user_input = input(prompt).strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            elif user_input.lower() == 'new':
                conv_id = agent.create_conversation()
                conversations[conv_id] = conv_id
                current_conv = conv_id
                print(f"Created new conversation: {conv_id[:8]}")
            
            elif user_input.lower() == 'list':
                convos = agent.list_conversations()
                print(f"Active conversations ({len(convos)}):")
                for conv in convos:
                    print(f"  {conv['id'][:8]}: {conv['message_count']} messages, {conv['status']}")
            
            elif user_input.lower().startswith('switch '):
                conv_prefix = user_input.split(' ', 1)[1]
                found_conv = None
                for conv_id in conversations:
                    if conv_id.startswith(conv_prefix):
                        found_conv = conv_id
                        break
                
                if found_conv:
                    current_conv = found_conv
                    print(f"Switched to conversation: {current_conv[:8]}")
                else:
                    print(f"Conversation starting with '{conv_prefix}' not found")
            
            elif user_input.lower() == 'stats':
                stats = agent.get_global_stats()
                print("Global Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            elif current_conv is None:
                print("No conversation selected. Use 'new' to create one.")
            
            elif user_input:
                # Send message to current conversation
                print("AI is thinking...")
                response = agent.chat(current_conv, user_input)
                print(f"AI: {response}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Concurrent conversations test")
    print("2. Interactive multi-chat")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_concurrent_conversations()
    elif choice == "2":
        test_interactive_multi_chat()
    else:
        print("Invalid choice")
```

## Step 5: Advanced Features

### Adding Conversation Persistence

```python
# persistent_chat_agent.py
import json
import os
from datetime import datetime
from multi_chat_agent import MultiChatAgent

class PersistentChatAgent(MultiChatAgent):
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7, data_dir: str = "./chat_data"):
        super().__init__(model, temperature)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing conversations
        self._load_conversations()
    
    def _get_conversation_file(self, conversation_id: str) -> str:
        return os.path.join(self.data_dir, f"conv_{conversation_id}.json")
    
    def _save_conversation(self, conversation_id: str):
        """Save conversation to disk"""
        if conversation_id not in self.conversations:
            return
        
        board = self.conversations[conversation_id]
        data = {
            "id": conversation_id,
            "history": board.get("history", []),
            "created_at": board.get("created_at"),
            "message_count": board.get("message_count", 0),
            "last_updated": time.time()
        }
        
        with open(self._get_conversation_file(conversation_id), 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_conversations(self):
        """Load conversations from disk"""
        for filename in os.listdir(self.data_dir):
            if filename.startswith("conv_") and filename.endswith(".json"):
                conv_id = filename[5:-5]  # Remove "conv_" and ".json"
                try:
                    with open(os.path.join(self.data_dir, filename), 'r') as f:
                        data = json.load(f)
                    
                    # Recreate conversation
                    board = Blackboard(scope=conv_id)
                    board.set("history", data.get("history", []))
                    board.set("created_at", data.get("created_at"))
                    board.set("message_count", data.get("message_count", 0))
                    board.set("status", "ready")
                    
                    self.conversations[conv_id] = board
                    
                except Exception as e:
                    print(f"Error loading conversation {conv_id}: {e}")
    
    def chat(self, conversation_id: str, user_message: str, timeout: float = 30.0) -> str:
        """Override to add persistence"""
        response = super().chat(conversation_id, user_message, timeout)
        self._save_conversation(conversation_id)
        return response
    
    def close_conversation(self, conversation_id: str):
        """Override to clean up files"""
        super().close_conversation(conversation_id)
        file_path = self._get_conversation_file(conversation_id)
        if os.path.exists(file_path):
            os.remove(file_path)
```

## Key Learning Points

1. **AsyncDispatcher Integration**: The dispatcher allows synchronous behavior trees to coordinate with asynchronous AI processing
2. **Blackboard State Management**: Scoped blackboards enable multi-tenant conversation isolation
3. **Error Handling**: Proper timeout and error recovery patterns ensure robust operation
4. **Scalability**: The multi-conversation pattern scales to handle many simultaneous users
5. **Extensibility**: The modular design allows easy addition of features like persistence, logging, etc.

## Next Steps

- Add streaming response support using `submit_stream()`
- Integrate with behavior trees for more complex conversation flows
- Add conversation summarization for long-running chats
- Implement user authentication and authorization
- Add conversation analytics and insights

This tutorial demonstrates the core patterns for building AI applications with Dachi's communication and async processing capabilities.