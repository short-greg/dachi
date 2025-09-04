# Tutorial: Multi-Agent Communication

This tutorial demonstrates how to build a multi-agent system using Dachi's Bulletin for message passing between agents. You'll learn how to create specialized agents that communicate through typed messages, coordinate work distribution, and build complex collaborative workflows.

## Overview

We'll build a document processing system with multiple specialized agents:

- **ManagerAgent**: Coordinates work and distributes tasks
- **AnalyzerAgent**: Analyzes document content and structure  
- **SummarizerAgent**: Creates summaries of analyzed content
- **ReporterAgent**: Generates final reports from summaries

The agents will communicate through typed messages using Bulletin, demonstrating patterns for:
- Work queue distribution
- Event-driven coordination
- Result aggregation
- Error handling and recovery

## Prerequisites

- Basic understanding of Dachi's Bulletin and Blackboard components
- Familiarity with Pydantic models for message typing
- Understanding of threading and async patterns

## Step 1: Define Message Types

First, let's define the typed messages that agents will exchange:

```python
# message_types.py
from __future__ import annotations

from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from enum import Enum
import time

class TaskType(str, Enum):
    ANALYZE_DOCUMENT = "analyze_document"
    SUMMARIZE_CONTENT = "summarize_content" 
    GENERATE_REPORT = "generate_report"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkRequest(BaseModel):
    """Request for work to be done by an agent"""
    task_id: str
    task_type: TaskType
    priority: Priority
    data: Dict[str, Any]
    requested_by: str
    created_at: float = None
    dependencies: List[str] = []  # Task IDs this depends on
    
    def __init__(self, **data):
        if data.get('created_at') is None:
            data['created_at'] = time.time()
        super().__init__(**data)

class WorkResult(BaseModel):
    """Result of completed work"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processed_by: str
    processing_time: float
    completed_at: float = None
    
    def __init__(self, **data):
        if data.get('completed_at') is None:
            data['completed_at'] = time.time()
        super().__init__(**data)

class AgentStatus(BaseModel):
    """Agent status update"""
    agent_id: str
    status: str  # "idle", "busy", "error", "offline"
    current_task: Optional[str] = None
    tasks_completed: int = 0
    last_activity: float = None
    capabilities: List[TaskType] = []
    
    def __init__(self, **data):
        if data.get('last_activity') is None:
            data['last_activity'] = time.time()
        super().__init__(**data)

class SystemEvent(BaseModel):
    """System-wide events and announcements"""
    event_type: str
    message: str
    data: Dict[str, Any] = {}
    timestamp: float = None
    sender: str = "system"
    
    def __init__(self, **data):
        if data.get('timestamp') is None:
            data['timestamp'] = time.time()
        super().__init__(**data)
```

## Step 2: Base Agent Class

Create a base class that all agents will inherit from:

```python
# base_agent.py
from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from dachi.comm import Bulletin, Blackboard, BulletinEventType
from message_types import (
    WorkRequest, WorkResult, AgentStatus, SystemEvent, 
    TaskType, TaskStatus, Priority
)

class BaseAgent(ABC):
    def __init__(self, agent_id: str, capabilities: List[TaskType]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        
        # Communication channels
        self.work_requests = Bulletin[WorkRequest]()
        self.work_results = Bulletin[WorkResult]()
        self.agent_status_board = Bulletin[AgentStatus]()
        self.system_events = Bulletin[SystemEvent]()
        
        # Shared state
        self.blackboard = Blackboard()
        
        # Agent state
        self.status = "initializing"
        self.current_task = None
        self.tasks_completed = 0
        self.running = False
        self.thread = None
        
        # Performance tracking
        self.start_time = time.time()
        self.task_history = []
        
        # Event handlers
        self._setup_event_handlers()
    
    def start(self):
        """Start the agent in a separate thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._agent_loop, daemon=True)
        self.thread.start()
        
        # Announce agent is online
        self._update_status("idle")
        self._publish_event("agent_started", f"Agent {self.agent_id} is now online")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self._update_status("offline")
        self._publish_event("agent_stopped", f"Agent {self.agent_id} is now offline")
    
    def _agent_loop(self):
        """Main agent processing loop"""
        while self.running:
            try:
                # Look for work that matches our capabilities
                work_request = self._find_suitable_work()
                
                if work_request:
                    self._process_work_request(work_request)
                else:
                    # No work available, brief sleep
                    time.sleep(1)
                    
            except Exception as e:
                self._handle_error(f"Agent loop error: {e}")
                time.sleep(5)  # Back off on errors
    
    def _find_suitable_work(self) -> Optional[WorkRequest]:
        """Find work request that matches agent capabilities"""
        # Get all pending work requests
        all_requests = self.work_requests.retrieve_all()
        
        for request in all_requests:
            # Check if we can handle this task type
            if request.task_type in self.capabilities:
                # Check dependencies
                if self._dependencies_satisfied(request):
                    # Try to lock this request
                    locked_request = self.work_requests.retrieve_first(
                        filters={"task_id": request.task_id},
                        lock_duration_seconds=300  # 5 minute lock
                    )
                    if locked_request:
                        return locked_request
        
        return None
    
    def _dependencies_satisfied(self, request: WorkRequest) -> bool:
        """Check if all dependencies for a work request are satisfied"""
        if not request.dependencies:
            return True
            
        for dep_task_id in request.dependencies:
            # Look for completed result
            result = self.work_results.retrieve_first(
                filters={"task_id": dep_task_id, "status": TaskStatus.COMPLETED}
            )
            if not result:
                return False
                
        return True
    
    def _process_work_request(self, request: WorkRequest):
        """Process a work request"""
        self.current_task = request.task_id
        self._update_status("busy")
        
        start_time = time.time()
        
        try:
            # Process the actual work
            result_data = self.process_task(request)
            
            # Create success result
            result = WorkResult(
                task_id=request.task_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                processed_by=self.agent_id,
                processing_time=time.time() - start_time
            )
            
            # Publish result
            self.work_results.publish(result)
            
            # Remove completed work from queue
            self.work_requests.remove(request.task_id)
            
            # Update statistics
            self.tasks_completed += 1
            self.task_history.append({
                "task_id": request.task_id,
                "task_type": request.task_type,
                "processing_time": result.processing_time,
                "completed_at": result.completed_at
            })
            
            print(f"[{self.agent_id}] Completed task {request.task_id} in {result.processing_time:.2f}s")
            
        except Exception as e:
            # Create failure result
            result = WorkResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                processed_by=self.agent_id,
                processing_time=time.time() - start_time
            )
            
            # Publish failure result
            self.work_results.publish(result)
            
            # Release the work request so another agent can try
            self.work_requests.release(request.task_id)
            
            self._handle_error(f"Task {request.task_id} failed: {e}")
        
        finally:
            self.current_task = None
            self._update_status("idle")
    
    @abstractmethod
    def process_task(self, request: WorkRequest) -> Dict[str, Any]:
        """Process a specific work request - implemented by subclasses"""
        pass
    
    def _update_status(self, status: str):
        """Update and broadcast agent status"""
        self.status = status
        
        status_update = AgentStatus(
            agent_id=self.agent_id,
            status=status,
            current_task=self.current_task,
            tasks_completed=self.tasks_completed,
            capabilities=self.capabilities
        )
        
        self.agent_status_board.publish(status_update)
    
    def _publish_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Publish a system event"""
        event = SystemEvent(
            event_type=event_type,
            message=message,
            data=data or {},
            sender=self.agent_id
        )
        self.system_events.publish(event)
    
    def _handle_error(self, error_message: str):
        """Handle and report errors"""
        self._update_status("error")
        self._publish_event("agent_error", error_message)
        print(f"[{self.agent_id}] ERROR: {error_message}")
    
    def _setup_event_handlers(self):
        """Setup event handlers for system events"""
        def on_system_event(event: SystemEvent, event_type):
            if event.sender != self.agent_id:  # Don't handle our own events
                self.handle_system_event(event)
        
        self.system_events.register_callback(
            on_system_event, 
            events={BulletinEventType.ON_PUBLISH}
        )
    
    def handle_system_event(self, event: SystemEvent):
        """Handle system events - can be overridden by subclasses"""
        if event.event_type == "shutdown":
            print(f"[{self.agent_id}] Received shutdown signal")
            self.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        uptime = time.time() - self.start_time
        avg_task_time = 0
        if self.task_history:
            avg_task_time = sum(t["processing_time"] for t in self.task_history) / len(self.task_history)
        
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "uptime": uptime,
            "tasks_completed": self.tasks_completed,
            "average_task_time": avg_task_time,
            "capabilities": self.capabilities,
            "current_task": self.current_task
        }
```

## Step 3: Specialized Agent Implementations

Now let's create the specialized agents:

```python
# specialized_agents.py
from __future__ import annotations

import time
import json
import random
from typing import Dict, Any

from base_agent import BaseAgent
from message_types import WorkRequest, TaskType

class AnalyzerAgent(BaseAgent):
    """Agent that analyzes document content and structure"""
    
    def __init__(self, agent_id: str = None):
        if agent_id is None:
            agent_id = f"analyzer_{random.randint(1000, 9999)}"
        super().__init__(agent_id, [TaskType.ANALYZE_DOCUMENT])
    
    def process_task(self, request: WorkRequest) -> Dict[str, Any]:
        """Analyze a document"""
        document_data = request.data
        
        # Simulate document analysis
        time.sleep(random.uniform(2, 5))  # Simulate processing time
        
        content = document_data.get("content", "")
        
        # Basic analysis (in real implementation, use NLP libraries)
        analysis = {
            "document_id": document_data.get("document_id"),
            "word_count": len(content.split()),
            "character_count": len(content),
            "paragraph_count": content.count('\n\n') + 1,
            "estimated_reading_time": len(content.split()) / 200,  # words per minute
            "topics": self._extract_topics(content),
            "sentiment": self._analyze_sentiment(content),
            "structure": self._analyze_structure(content)
        }
        
        return analysis
    
    def _extract_topics(self, content: str) -> list:
        """Extract topics from content (simplified)"""
        # In real implementation, use topic modeling
        words = content.lower().split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        significant_words = [w for w in words if len(w) > 4 and w not in common_words]
        
        # Return top 10 most frequent significant words
        from collections import Counter
        topics = Counter(significant_words).most_common(10)
        return [topic[0] for topic in topics]
    
    def _analyze_sentiment(self, content: str) -> str:
        """Analyze sentiment (simplified)"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        lines = content.split('\n')
        headings = [line for line in lines if line.strip().startswith('#') or line.isupper()]
        
        return {
            "has_headings": len(headings) > 0,
            "heading_count": len(headings),
            "line_count": len(lines),
            "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
        }

class SummarizerAgent(BaseAgent):
    """Agent that creates summaries from analyzed content"""
    
    def __init__(self, agent_id: str = None):
        if agent_id is None:
            agent_id = f"summarizer_{random.randint(1000, 9999)}"
        super().__init__(agent_id, [TaskType.SUMMARIZE_CONTENT])
    
    def process_task(self, request: WorkRequest) -> Dict[str, Any]:
        """Create summary from analysis results"""
        analysis_data = request.data
        
        # Simulate summarization processing
        time.sleep(random.uniform(3, 6))
        
        # Get analysis results from dependencies
        analysis_result = None
        for dep_task_id in request.dependencies:
            result = self.work_results.retrieve_first(
                filters={"task_id": dep_task_id, "status": "completed"}
            )
            if result and result.result.get("word_count"):  # This is analysis result
                analysis_result = result.result
                break
        
        if not analysis_result:
            raise Exception("Could not find analysis results for summarization")
        
        # Create summary based on analysis
        summary = {
            "document_id": analysis_result.get("document_id"),
            "executive_summary": self._create_executive_summary(analysis_result),
            "key_statistics": {
                "word_count": analysis_result.get("word_count"),
                "reading_time": f"{analysis_result.get('estimated_reading_time', 0):.1f} minutes",
                "sentiment": analysis_result.get("sentiment"),
                "topic_count": len(analysis_result.get("topics", []))
            },
            "main_topics": analysis_result.get("topics", [])[:5],  # Top 5 topics
            "summary_length": "brief" if analysis_result.get("word_count", 0) < 1000 else "detailed"
        }
        
        return summary
    
    def _create_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Create executive summary text"""
        word_count = analysis.get("word_count", 0)
        sentiment = analysis.get("sentiment", "neutral")
        topics = analysis.get("topics", [])[:3]  # Top 3 topics
        
        summary_parts = []
        
        # Document size assessment
        if word_count < 500:
            summary_parts.append("This is a brief document")
        elif word_count < 2000:
            summary_parts.append("This is a medium-length document")
        else:
            summary_parts.append("This is a comprehensive document")
        
        summary_parts.append(f"with {word_count} words")
        
        # Sentiment
        if sentiment != "neutral":
            summary_parts.append(f"and {sentiment} sentiment")
        
        # Topics
        if topics:
            topic_text = ", ".join(topics[:2])
            if len(topics) > 2:
                topic_text += f", and {topics[2]}"
            summary_parts.append(f"covering topics including {topic_text}")
        
        return " ".join(summary_parts) + "."

class ReporterAgent(BaseAgent):
    """Agent that generates final reports from summaries"""
    
    def __init__(self, agent_id: str = None):
        if agent_id is None:
            agent_id = f"reporter_{random.randint(1000, 9999)}"
        super().__init__(agent_id, [TaskType.GENERATE_REPORT])
    
    def process_task(self, request: WorkRequest) -> Dict[str, Any]:
        """Generate final report"""
        
        # Simulate report generation
        time.sleep(random.uniform(2, 4))
        
        # Gather all dependency results
        summary_result = None
        analysis_result = None
        
        for dep_task_id in request.dependencies:
            result = self.work_results.retrieve_first(
                filters={"task_id": dep_task_id, "status": "completed"}
            )
            if result:
                if result.result.get("executive_summary"):  # Summary result
                    summary_result = result.result
                elif result.result.get("word_count"):  # Analysis result
                    analysis_result = result.result
        
        if not summary_result:
            raise Exception("Could not find summary results for report generation")
        
        # Generate comprehensive report
        report = {
            "document_id": summary_result.get("document_id"),
            "report_generated_at": time.time(),
            "report_type": "comprehensive_analysis",
            "executive_summary": summary_result.get("executive_summary"),
            "detailed_analysis": self._create_detailed_analysis(analysis_result, summary_result),
            "recommendations": self._generate_recommendations(analysis_result, summary_result),
            "metadata": {
                "processing_pipeline": ["analysis", "summarization", "reporting"],
                "quality_score": self._calculate_quality_score(analysis_result),
                "confidence": "high" if analysis_result and analysis_result.get("word_count", 0) > 100 else "medium"
            }
        }
        
        return report
    
    def _create_detailed_analysis(self, analysis: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed analysis section"""
        if not analysis:
            return {"note": "Detailed analysis not available"}
        
        return {
            "content_metrics": {
                "word_count": analysis.get("word_count"),
                "paragraph_count": analysis.get("paragraph_count"),
                "estimated_reading_time": analysis.get("estimated_reading_time")
            },
            "content_analysis": {
                "sentiment": analysis.get("sentiment"),
                "main_topics": analysis.get("topics", [])[:5],
                "structure_quality": "good" if analysis.get("structure", {}).get("has_headings") else "basic"
            },
            "summary_insights": {
                "summary_type": summary.get("summary_length"),
                "key_topics": summary.get("main_topics", [])
            }
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any], summary: Dict[str, Any]) -> list:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if analysis:
            word_count = analysis.get("word_count", 0)
            if word_count < 300:
                recommendations.append("Consider expanding content for better depth")
            elif word_count > 5000:
                recommendations.append("Consider breaking into smaller sections for better readability")
            
            if not analysis.get("structure", {}).get("has_headings"):
                recommendations.append("Add headings to improve document structure")
            
            sentiment = analysis.get("sentiment")
            if sentiment == "negative":
                recommendations.append("Review content tone for potential improvements")
            
        if not recommendations:
            recommendations.append("Document meets quality standards")
        
        return recommendations
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate document quality score (0-1)"""
        if not analysis:
            return 0.5
        
        score = 0.5  # Base score
        
        # Word count factor
        word_count = analysis.get("word_count", 0)
        if 300 <= word_count <= 3000:
            score += 0.2
        
        # Structure factor
        if analysis.get("structure", {}).get("has_headings"):
            score += 0.2
        
        # Topic diversity factor
        topic_count = len(analysis.get("topics", []))
        if topic_count >= 3:
            score += 0.1
        
        return min(1.0, score)
```

## Step 4: Manager Agent for Coordination

Create a manager agent that coordinates the workflow:

```python
# manager_agent.py
from __future__ import annotations

import uuid
import time
from typing import Dict, Any, List
from base_agent import BaseAgent
from message_types import WorkRequest, TaskType, Priority

class ManagerAgent(BaseAgent):
    """Manager agent that coordinates multi-step document processing workflows"""
    
    def __init__(self, agent_id: str = "manager"):
        # Manager doesn't process work directly, just coordinates
        super().__init__(agent_id, [])
        
        # Track active workflows
        self.active_workflows = {}
    
    def process_document(self, document_id: str, content: str, priority: Priority = Priority.MEDIUM) -> str:
        """Start a complete document processing workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Create task IDs
        analyze_task_id = f"{workflow_id}_analyze"
        summarize_task_id = f"{workflow_id}_summarize"  
        report_task_id = f"{workflow_id}_report"
        
        # Track workflow
        self.active_workflows[workflow_id] = {
            "document_id": document_id,
            "tasks": [analyze_task_id, summarize_task_id, report_task_id],
            "status": "started",
            "started_at": time.time()
        }
        
        # Submit analysis task (no dependencies)
        analyze_request = WorkRequest(
            task_id=analyze_task_id,
            task_type=TaskType.ANALYZE_DOCUMENT,
            priority=priority,
            data={
                "document_id": document_id,
                "content": content,
                "workflow_id": workflow_id
            },
            requested_by=self.agent_id,
            dependencies=[]
        )
        self.work_requests.publish(analyze_request)
        
        # Submit summarization task (depends on analysis)
        summarize_request = WorkRequest(
            task_id=summarize_task_id,
            task_type=TaskType.SUMMARIZE_CONTENT,
            priority=priority,
            data={
                "document_id": document_id,
                "workflow_id": workflow_id
            },
            requested_by=self.agent_id,
            dependencies=[analyze_task_id]
        )
        self.work_requests.publish(summarize_request)
        
        # Submit report generation task (depends on both analysis and summary)
        report_request = WorkRequest(
            task_id=report_task_id,
            task_type=TaskType.GENERATE_REPORT,
            priority=priority,
            data={
                "document_id": document_id,
                "workflow_id": workflow_id
            },
            requested_by=self.agent_id,
            dependencies=[analyze_task_id, summarize_task_id]
        )
        self.work_requests.publish(report_request)
        
        print(f"[{self.agent_id}] Started workflow {workflow_id} for document {document_id}")
        self._publish_event("workflow_started", f"Started processing workflow {workflow_id}", {
            "workflow_id": workflow_id,
            "document_id": document_id,
            "task_count": 3
        })
        
        return workflow_id
    
    def process_task(self, request: WorkRequest) -> Dict[str, Any]:
        """Manager doesn't process individual tasks"""
        raise NotImplementedError("Manager agent coordinates but doesn't process tasks")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        task_statuses = {}
        
        # Check status of each task
        for task_id in workflow["tasks"]:
            # Check if task is completed
            result = self.work_results.retrieve_first(
                filters={"task_id": task_id}
            )
            
            if result:
                task_statuses[task_id] = {
                    "status": result.status,
                    "completed_at": result.completed_at,
                    "processed_by": result.processed_by
                }
            else:
                # Check if task is still in queue
                request = self.work_requests.retrieve_first(
                    filters={"task_id": task_id}
                )
                if request:
                    task_statuses[task_id] = {"status": "queued"}
                else:
                    task_statuses[task_id] = {"status": "unknown"}
        
        # Determine overall workflow status
        completed_tasks = [t for t in task_statuses.values() if t.get("status") == "completed"]
        failed_tasks = [t for t in task_statuses.values() if t.get("status") == "failed"]
        
        if len(completed_tasks) == len(workflow["tasks"]):
            workflow_status = "completed"
        elif len(failed_tasks) > 0:
            workflow_status = "failed"
        else:
            workflow_status = "in_progress"
        
        return {
            "workflow_id": workflow_id,
            "document_id": workflow["document_id"],
            "status": workflow_status,
            "started_at": workflow["started_at"],
            "task_statuses": task_statuses,
            "progress": f"{len(completed_tasks)}/{len(workflow['tasks'])} tasks completed"
        }
    
    def get_final_report(self, workflow_id: str) -> Dict[str, Any]:
        """Get the final report for a completed workflow"""
        workflow_status = self.get_workflow_status(workflow_id)
        
        if workflow_status.get("status") != "completed":
            return {"error": f"Workflow not completed. Status: {workflow_status.get('status')}"}
        
        # Find the report task result
        report_task_id = f"{workflow_id}_report"
        report_result = self.work_results.retrieve_first(
            filters={"task_id": report_task_id, "status": "completed"}
        )
        
        if not report_result:
            return {"error": "Report result not found"}
        
        return report_result.result
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows and their statuses"""
        workflows = []
        for workflow_id in self.active_workflows:
            status = self.get_workflow_status(workflow_id)
            workflows.append(status)
        return workflows
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        # Cancel any pending tasks
        cancelled_count = 0
        for task_id in workflow["tasks"]:
            if self.work_requests.remove(task_id):
                cancelled_count += 1
        
        # Mark workflow as cancelled
        workflow["status"] = "cancelled"
        workflow["cancelled_at"] = time.time()
        
        self._publish_event("workflow_cancelled", f"Cancelled workflow {workflow_id}", {
            "workflow_id": workflow_id,
            "cancelled_tasks": cancelled_count
        })
        
        return True
```

## Step 5: System Integration and Testing

Now let's create a complete system test:

```python
# test_multi_agent_system.py
from __future__ import annotations

import time
import threading
from typing import List

from manager_agent import ManagerAgent
from specialized_agents import AnalyzerAgent, SummarizerAgent, ReporterAgent
from message_types import Priority

class DocumentProcessingSystem:
    def __init__(self):
        # Create agents
        self.manager = ManagerAgent()
        self.analyzers = [AnalyzerAgent(f"analyzer_{i}") for i in range(2)]
        self.summarizers = [SummarizerAgent(f"summarizer_{i}") for i in range(2)]
        self.reporters = [ReporterAgent(f"reporter_{i}") for i in range(1)]
        
        self.all_agents = [self.manager] + self.analyzers + self.summarizers + self.reporters
        
    def start_system(self):
        """Start all agents"""
        print("Starting document processing system...")
        for agent in self.all_agents:
            agent.start()
        print(f"Started {len(self.all_agents)} agents")
        time.sleep(2)  # Let agents initialize
    
    def stop_system(self):
        """Stop all agents"""
        print("Stopping document processing system...")
        for agent in self.all_agents:
            agent.stop()
    
    def process_document(self, document_id: str, content: str, priority: Priority = Priority.MEDIUM) -> str:
        """Process a document through the system"""
        return self.manager.process_document(document_id, content, priority)
    
    def wait_for_completion(self, workflow_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for a workflow to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.manager.get_workflow_status(workflow_id)
            
            if status.get("status") == "completed":
                return self.manager.get_final_report(workflow_id)
            elif status.get("status") == "failed":
                return {"error": "Workflow failed", "details": status}
            
            time.sleep(2)
        
        return {"error": "Workflow timed out", "last_status": status}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        stats = {}
        for agent in self.all_agents:
            agent_stats = agent.get_stats()
            stats[agent.agent_id] = agent_stats
        
        # Calculate system totals
        total_tasks = sum(s["tasks_completed"] for s in stats.values())
        active_agents = sum(1 for s in stats.values() if s["status"] != "offline")
        
        return {
            "agents": stats,
            "system_totals": {
                "total_tasks_completed": total_tasks,
                "active_agents": active_agents,
                "total_agents": len(self.all_agents)
            },
            "workflows": self.manager.list_active_workflows()
        }

def test_single_document():
    """Test processing a single document"""
    system = DocumentProcessingSystem()
    system.start_system()
    
    try:
        # Process a test document
        test_content = """
        Artificial Intelligence and Machine Learning: A Comprehensive Overview
        
        Artificial intelligence (AI) has become one of the most transformative technologies 
        of the 21st century. Machine learning, a subset of AI, enables computers to learn 
        and improve from experience without being explicitly programmed.
        
        Applications of AI include natural language processing, computer vision, robotics,
        and autonomous systems. These technologies are revolutionizing industries from 
        healthcare to transportation.
        
        The future of AI holds great promise, with potential applications in climate change
        mitigation, drug discovery, and educational personalization. However, it also 
        raises important ethical considerations around privacy, bias, and job displacement.
        """
        
        print("Processing test document...")
        workflow_id = system.process_document("test_doc_001", test_content, Priority.HIGH)
        
        print(f"Started workflow: {workflow_id}")
        print("Waiting for completion...")
        
        # Wait for completion
        final_report = system.wait_for_completion(workflow_id, timeout=120)
        
        if "error" in final_report:
            print(f"Error: {final_report}")
        else:
            print("\n" + "="*60)
            print("FINAL REPORT")
            print("="*60)
            print(f"Document ID: {final_report.get('document_id')}")
            print(f"Executive Summary: {final_report.get('executive_summary')}")
            print(f"Quality Score: {final_report.get('metadata', {}).get('quality_score', 'N/A')}")
            print(f"Recommendations: {final_report.get('recommendations', [])}")
            
        # Show system stats
        print("\n" + "="*60)
        print("SYSTEM STATISTICS")
        print("="*60)
        stats = system.get_system_stats()
        for agent_id, agent_stats in stats["agents"].items():
            print(f"{agent_id}: {agent_stats['tasks_completed']} tasks, {agent_stats['status']}")
        
    finally:
        system.stop_system()

def test_concurrent_documents():
    """Test processing multiple documents concurrently"""
    system = DocumentProcessingSystem()
    system.start_system()
    
    try:
        documents = [
            ("tech_doc", "This document discusses various technology trends including AI, blockchain, and IoT. Technology is rapidly evolving..."),
            ("business_doc", "This business report analyzes market conditions and provides strategic recommendations for growth..."),
            ("science_doc", "This scientific paper explores recent discoveries in quantum physics and their potential applications...")
        ]
        
        workflow_ids = []
        
        # Start all workflows
        for doc_id, content in documents:
            workflow_id = system.process_document(doc_id, content, Priority.MEDIUM)
            workflow_ids.append(workflow_id)
            print(f"Started workflow {workflow_id} for {doc_id}")
        
        # Wait for all to complete
        print(f"\nWaiting for {len(workflow_ids)} workflows to complete...")
        
        completed_reports = {}
        start_time = time.time()
        timeout = 180  # 3 minutes
        
        while len(completed_reports) < len(workflow_ids) and time.time() - start_time < timeout:
            for workflow_id in workflow_ids:
                if workflow_id not in completed_reports:
                    status = system.manager.get_workflow_status(workflow_id)
                    if status.get("status") == "completed":
                        report = system.manager.get_final_report(workflow_id)
                        completed_reports[workflow_id] = report
                        print(f"✓ Workflow {workflow_id} completed")
                    elif status.get("status") == "failed":
                        completed_reports[workflow_id] = {"error": "failed"}
                        print(f"✗ Workflow {workflow_id} failed")
            
            time.sleep(3)
        
        # Show results
        print(f"\nCompleted {len(completed_reports)}/{len(workflow_ids)} workflows")
        
        for workflow_id, report in completed_reports.items():
            if "error" not in report:
                print(f"\n{workflow_id}:")
                print(f"  Document: {report.get('document_id')}")
                print(f"  Summary: {report.get('executive_summary', 'N/A')[:100]}...")
                print(f"  Quality: {report.get('metadata', {}).get('quality_score', 'N/A')}")
        
        # Final system stats
        print("\nFinal System Statistics:")
        stats = system.get_system_stats()
        print(f"Total tasks completed: {stats['system_totals']['total_tasks_completed']}")
        print(f"Active agents: {stats['system_totals']['active_agents']}")
        
    finally:
        system.stop_system()

if __name__ == "__main__":
    print("Multi-Agent Document Processing System Test")
    print("=" * 50)
    
    choice = input("Choose test: (1) Single document, (2) Concurrent documents: ")
    
    if choice == "1":
        test_single_document()
    elif choice == "2":
        test_concurrent_documents()
    else:
        print("Invalid choice")
```

## Key Learning Points

1. **Typed Message Communication**: Using Pydantic models ensures type safety and clear contracts between agents
2. **Dependency Management**: Tasks can depend on other tasks, enabling complex workflows
3. **Event-Driven Coordination**: Agents react to system events for coordination and monitoring
4. **Scalable Architecture**: Multiple agents of the same type can handle workload scaling
5. **Error Handling and Recovery**: Failed tasks can be retried by other agents
6. **Workflow Management**: Complex multi-step processes can be coordinated centrally

## Next Steps

- Add priority queue handling for urgent tasks
- Implement load balancing between agents of the same type
- Add persistent task storage for system restart recovery
- Create web dashboard for monitoring agent activities
- Add metrics collection and performance monitoring
- Implement agent discovery and dynamic scaling

This tutorial demonstrates the power of Dachi's Bulletin system for building sophisticated multi-agent architectures with clear communication patterns and robust error handling.