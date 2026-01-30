"""
Fixer ClawdBot Agent implementation.
Handles task execution, status tracking, and health monitoring.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import os

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRequest(BaseModel):
    """Request model for task execution."""
    task_type: str = Field(..., description="Type of task to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    async_execution: bool = Field(default=False, description="Whether to execute asynchronously")
    timeout: Optional[int] = Field(default=300, description="Task timeout in seconds")
    priority: int = Field(default=5, description="Task priority (1-10, 10 is highest)")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")


class TaskInfo(BaseModel):
    """Internal task information model."""
    request: TaskRequest
    response: TaskResponse
    start_time: Optional[float] = None
    task_handle: Optional[asyncio.Task] = None


class FixerAgent:
    """
    Main agent class for handling task execution and management.
    """
    
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.is_initialized = False
        self.start_time = time.time()
        self.task_counter = 0
        self.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
        self.task_retention_time = int(os.getenv("TASK_RETENTION_TIME", "3600"))  # 1 hour
        
        # Task execution handlers
        self.task_handlers = {
            "text_processing": self._handle_text_processing,
            "data_analysis": self._handle_data_analysis,
            "file_processing": self._handle_file_processing,
            "system_check": self._handle_system_check,
            "echo": self._handle_echo,
        }
    
    async def initialize(self) -> None:
        """Initialize the agent."""
        try:
            logger.info("Initializing Fixer ClawdBot Agent...")
            
            # Start background cleanup task
            asyncio.create_task(self._cleanup_old_tasks())
            
            self.is_initialized = True
            logger.info("Agent initialization completed")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        try:
            logger.info("Shutting down agent...")
            
            # Cancel all running tasks
            for task_id, task_info in self.tasks.items():
                if task_info.task_handle and not task_info.task_handle.done():
                    task_info.task_handle.cancel()
                    logger.info(f"Cancelled task {task_id}")
            
            # Wait for tasks to complete or timeout
            await asyncio.sleep(1)
            
            self.is_initialized = False
            logger.info("Agent shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def get_health(self) -> Dict[str, Any]:
        """Get agent health status."""
        try:
            running_tasks = sum(1 for t in self.tasks.values() 
                              if t.response.status == TaskStatus.RUNNING)
            
            memory_usage = self._get_memory_usage()
            
            healthy = (
                self.is_initialized and
                running_tasks < self.max_concurrent_tasks and
                memory_usage < 0.9  # Less than 90% memory usage
            )
            
            return {
                "healthy": healthy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime": time.time() - self.start_time,
                "running_tasks": running_tasks,
                "total_tasks": len(self.tasks),
                "memory_usage": memory_usage,
                "max_concurrent_tasks": self.max_concurrent_tasks
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status."""
        try:
            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = sum(
                    1 for t in self.tasks.values() 
                    if t.response.status == status
                )
            
            recent_tasks = [
                {
                    "task_id": task_id,
                    "task_type": task_info.request.task_type,
                    "status": task_info.response.status.value,
                    "created_at": task_info.response.created_at.isoformat(),
                    "execution_time": task_info.response.execution_time
                }
                for task_id, task_info in sorted(
                    self.tasks.items(),
                    key=lambda x: x[1].response.created_at,
                    reverse=True
                )[:10]  # Last 10 tasks
            ]
            
            return {
                "agent_id": "fixer.clawdbot",
                "status": "running" if self.is_initialized else "stopped",
                "uptime": time.time() - self.start_time,
                "task_counts": status_counts,
                "total_tasks_processed": self.task_counter,
                "available_task_types": list(self.task_handlers.keys()),
                "recent_tasks": recent_tasks,
                "configuration": {
                    "max_concurrent_tasks": self.max_concurrent_tasks,
                    "task_retention_time": self.task_retention_time
                }
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise
    
    async def execute_task(self, task_request: TaskRequest) -> TaskResponse:
        """Execute a task based on the request."""
        try:
            # Validate task type
            if task_request.task_type not in self.task_handlers:
                raise ValueError(f"Unsupported task type: {task_request.task_type}")
            
            # Check concurrent task limit
            running_tasks = sum(1 for t in self.tasks.values() 
                              if t.response.status == TaskStatus.RUNNING)
            if running_tasks >= self.max_concurrent_tasks:
                raise ValueError("Maximum concurrent tasks limit reached")
            
            # Create task
            task_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            response = TaskResponse(
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now
            )
            
            task_info = TaskInfo(request=task_request, response=response)
            self.tasks[task_id] = task_info
            self.task_counter += 1
            
            if task_request.async_execution:
                # Execute asynchronously
                task_handle = asyncio.create_task(
                    self._execute_task_async(task_id)
                )
                task_info.task_handle = task_handle
                return response
            else:
                # Execute synchronously
                return await self._execute_task_sync(task_id)
                
        except Exception as e:
            logger.error(f"Task execution setup failed: {e}")
            raise
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResponse]:
        """Get task result by ID."""
        task