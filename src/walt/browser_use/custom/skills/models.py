"""
Generic Skills Models

Shared action models for skills that work across all platforms/benchmarks.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from walt.browser_use.agent.views import AgentHistoryList


class ExtractContentAction(BaseModel):
    """Parameters for extracting page content with multimodal support"""

    goal: str = Field(
        description="Specific information to extract from the page (e.g., 'all company names and logo descriptions')"
    )
    max_images: int = Field(
        default=10,
        description="Maximum number of images to process (to manage token usage)",
    )


class VerifyAction(BaseModel):
    """Parameters for verifying agent task completion using webjudge"""

    task: str = Field(
        description="The original task description that the agent is trying to complete"
    )
    task_image_paths: Optional[List[str]] = Field(
        default=None,
        description="File paths to task images if provided with the original task",
    )
    judge_model: str = Field(
        default="gpt-5-mini", description="Model to use for verification judgment"
    )
    score_threshold: int = Field(
        default=5, description="Score threshold for success evaluation"
    )
