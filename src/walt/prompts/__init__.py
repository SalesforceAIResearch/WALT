"""Centralized prompt management for WALT agents."""

from walt.prompts.agent import (
    build_extended_system_message,
    get_agent_system_prompt,
    get_tool_guidance,
)
from walt.prompts.discovery import (
    get_demonstration_prompt,
    get_exploration_prompt,
    get_tool_builder_prompt,
    get_workflow_creation_prompt,
)
from walt.prompts.memory import (
    get_narrative_memory_system,
    get_query_formulator_prompt,
)
from walt.prompts.planner import get_planner_prompt
from walt.prompts.tool_executor import (
    get_structured_output_prompt,
    get_tool_executor_step_prompt,
    get_tool_fallback_prompt,
)

__all__ = [
    # Agent prompts
    'build_extended_system_message',
    'get_agent_system_prompt',
    'get_tool_guidance',
    # Planner
    'get_planner_prompt',
    # Memory
    'get_query_formulator_prompt',
    'get_narrative_memory_system',
    # Tool Executor
    'get_tool_executor_step_prompt',
    'get_structured_output_prompt',
    'get_tool_fallback_prompt',
    # Discovery
    'get_tool_builder_prompt',
    'get_demonstration_prompt',
    'get_workflow_creation_prompt',
    'get_exploration_prompt',
]
