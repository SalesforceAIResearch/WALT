"""
WALT: Web Agents that Learn Tools

Automatic tool discovery from websites for LLM agents.
"""

__version__ = "0.1.0"

# Lazy imports to avoid slow startup for CLI help
# These will be imported on first access via __getattr__
def __getattr__(name):
    """Lazy import expensive modules only when accessed."""
    if name in ("Agent", "Browser", "BrowserConfig", "BrowserContextConfig", 
                "Controller", "DomService", "SystemPrompt", "ActionResult", 
                "ActionModel", "AgentHistoryList"):
        from walt.browser_use import (
            Agent,
            Browser,
            BrowserConfig,
            BrowserContextConfig,
            Controller,
            DomService,
            SystemPrompt,
            ActionResult,
            ActionModel,
            AgentHistoryList,
        )
        return locals()[name]
    elif name == "Tool":
        from walt.tools.tool import Tool
        return Tool
    elif name == "ToolDefinitionSchema":
        from walt.tools.schema import ToolDefinitionSchema
        return ToolDefinitionSchema
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Version
    "__version__",
    # Browser-use core
    "Agent",
    "Browser",
    "BrowserConfig",
    "BrowserContextConfig",
    "Controller",
    "DomService",
    "SystemPrompt",
    "ActionResult",
    "ActionModel",
    "AgentHistoryList",
    # Tool system
    "Tool",
    "ToolDefinitionSchema",
]

