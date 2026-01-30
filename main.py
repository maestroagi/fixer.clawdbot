"""
FastAPI application entry point for fixer.clawdbot agent.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import os

from agent import FixerAgent, TaskRequest, TaskResponse, TaskStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent_instance
    
    # Startup
    logger.info("Starting fixer.clawdbot agent...")
    agent_instance = FixerAgent()
    await agent_instance.initialize()
    logger.info("Agent initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down fixer.clawdbot agent...")
    if agent_instance:
        await agent_instance.shutdown()
    logger.info("Agent shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Fixer ClawdBot Agent",
    description="A production-ready FastAPI agent for automated task processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    Returns the current health status of the agent.
    """
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        health_status = await agent_instance.get_health()
        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "timestamp": health_status["timestamp"],
            "version": "1.0.0",
            "agent": "fixer.clawdbot",
            "details": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/status", tags=["Status"])
async def get_status() -> Dict[str, Any]:
    """
    Get agent status and metrics.
    Returns detailed information about agent performance and current state.
    """
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        status = await agent_instance.get_status()
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")


@app.post("/run", response_model=TaskResponse, tags=["Tasks"])
async def run_task(
    task_request: TaskRequest,
    background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    Execute a task using the fixer agent.
    
    Args:
        task_request: The task configuration and parameters
        background_tasks: FastAPI background tasks for cleanup
    
    Returns:
        TaskResponse with execution results or task ID for async tasks
    """
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        logger.info(f"Received task request: {task_request.task_type}")
        
        # Execute the task
        response = await agent_instance.execute_task(task_request)
        
        # Add cleanup task if needed
        if response.task_id:
            background_tasks.add_task(
                agent_instance.cleanup_task,
                response.task_id
            )
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid task request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Task execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Task execution failed")


@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task_result(task_id: str) -> TaskResponse:
    """
    Get the result of an asynchronous task.
    
    Args:
        task_id: The unique identifier of the task
    
    Returns:
        TaskResponse with current task status and results
    """
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        result = await agent_instance.get_task_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task result: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task result")


@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def cancel_task(task_id: str) -> Dict[str, str]:
    """
    Cancel a running task.
    
    Args:
        task_id: The unique identifier of the task to cancel
    
    Returns:
        Cancellation confirmation
    """
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        success = await agent_instance.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or already completed")
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel task")


if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = int(os.getenv("WORKERS", "1"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )