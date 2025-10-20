"""
Standalone tool registration functions.
Used by both test.py and aeval_with_tools.py for consistent tool handling.
"""

from pathlib import Path
from typing import List, Optional
from walt.browser_use import Controller
from walt.browser_use.agent.views import ActionResult
from langchain_core.language_models import BaseChatModel


def _discover_tool_files(
    tool_dir: str, tool_type: str = "base"
) -> List[Path]:
    """Discover tool files in subdirectories of the given directory.

    If tool_type is "optimized", it will first look for .optimized.json files,
    and if none are found for a tool, it will fallback to .tool.json files.
    """
    tool_path = Path(tool_dir)
    if not tool_path.exists():
        return []

    tool_files = []

    # Track tools by name to implement fallback logic
    tools_found = {}  # tool_name -> {"optimized": Path|None, "base": Path|None}

    for subdir in tool_path.iterdir():
        if subdir.is_dir() and subdir.name not in ["logs", "exclude"]:
            tool_name = subdir.name
            tools_found[tool_name] = {"optimized": None, "base": None}

            # Look for optimized version
            for optimized_file in subdir.glob("*.optimized.json"):
                stem = optimized_file.stem
                # Skip versioned files (e.g., .v1.optimized.json)
                if ".v" in stem:
                    parts = stem.split(".v")
                    if len(parts) > 1:
                        version_part = parts[1].split(".")[0]
                        if version_part.isdigit():
                            continue
                tools_found[tool_name]["optimized"] = optimized_file
                break  # Take the first non-versioned optimized file

            # Look for base version
            for base_file in subdir.glob("*.tool.json"):
                stem = base_file.stem
                # Skip versioned files (e.g., .v1.tool.json)
                if ".v" in stem:
                    parts = stem.split(".v")
                    if len(parts) > 1:
                        version_part = parts[1].split(".")[0]
                        if version_part.isdigit():
                            continue
                tools_found[tool_name]["base"] = base_file
                break  # Take the first non-versioned base file

    # Select files based on tool_type and fallback logic
    for tool_name, files in tools_found.items():
        if tool_type == "optimized":
            # Prefer optimized, fallback to base
            if files["optimized"]:
                tool_files.append(files["optimized"])
            elif files["base"]:
                tool_files.append(files["base"])
        else:
            # Only use base tools
            if files["base"]:
                tool_files.append(files["base"])

    return tool_files


def _register_single_tool(
    controller: Controller,
    tool_file: Path,
    llm: BaseChatModel,
    page_extraction_llm: Optional[BaseChatModel] = None,
    logger=None,
    fallback_to_agent: bool = False,
) -> bool:
    """Register a single tool file as an action on the controller."""
    try:
        from walt.tools.schema.views import ToolDefinitionSchema
        from walt.tools.executor.service import Tool
        from walt.browser_use.controller.views import NoParamsAction

        schema = ToolDefinitionSchema.load_from_json(str(tool_file))
        if hasattr(schema, "tool_analysis"):
            del schema.tool_analysis
            
        tool = Tool(
            tool_schema=schema,  # Fixed: was 'tool_schema', should be 'tool_schema'
            browser=None,  # Will be set at runtime
            llm=llm,
            page_extraction_llm=page_extraction_llm,
            fallback_to_agent=fallback_to_agent,
        )

        # Determine action name from filename
        action_name = tool_file.stem.replace(".tool", "").replace(
            ".optimized", ""
        )

        # Create tool action function
        def create_tool_action(captured_tool, tool_name):
            async def tool_action(**kwargs) -> ActionResult:
                # Extract injected browser
                browser_context = kwargs.pop("browser", None)
                browser = browser_context.browser
                browser._original_context = browser_context

                # Create Pydantic model from remaining kwargs
                params = captured_tool._input_model(**kwargs)
                captured_tool.browser = browser

                # Ensure browser profile keeps alive
                if hasattr(browser, "browser_profile"):
                    browser.browser_profile.keep_alive = True

                # Verify browser context is still valid
                try:
                    await browser_context.get_current_page()
                except Exception as e:
                    return ActionResult(
                        extracted_content=f"Tool '{tool_name}' failed: browser context invalid",
                        include_in_memory=True,
                    )

                # Extract parameters
                if hasattr(params, "model_dump"):
                    inputs = params.model_dump()
                elif hasattr(params, "dict"):
                    inputs = params.dict()
                else:
                    inputs = {}

                if logger and callable(logger):
                    logger(
                        f"Executing tool '{tool_name}' with inputs: {inputs}"
                    )

                # Run the tool
                result = await captured_tool.run(
                    inputs=inputs, close_browser_at_end=False
                )

                success_msg = f"Tool '{tool_name}' completed: {len(result.step_results)} steps executed"
                if logger and callable(logger):
                    logger(success_msg)

                return ActionResult(
                    extracted_content=success_msg, include_in_memory=True
                )

            # Set unique function name for controller registration
            tool_action.__name__ = f"tool_{action_name}"
            return tool_action

        # Register action
        param_model = (
            NoParamsAction
            if not tool._input_model.model_fields
            else tool._input_model
        )

        controller.action(description=schema.description, param_model=param_model)(
            create_tool_action(tool, action_name)
        )

        if logger and callable(logger):
            logger(f"✅ Registered tool: {action_name} from {tool_file.name}")

        return True

    except Exception as e:
        if logger and hasattr(logger, "error"):
            logger.error(f"Failed to register tool {tool_file}: {e}")
        elif logger and callable(logger):
            logger(f"Failed to register tool {tool_file}: {e}")
        return False


def register_tools_from_directory(
    controller: Controller,
    tool_dir: str,
    llm: BaseChatModel,
    logger=None,
    tool_type: str = "optimized",
    page_extraction_llm: Optional[BaseChatModel] = None,
    fallback_to_agent: bool = False,
) -> int:
    """
    Standalone function to register tools from a directory.
    Used by aeval_with_tools.py for backward compatibility.
    """

    tool_path = Path(tool_dir)
    if not tool_path.exists():
        if logger and hasattr(logger, "warning"):
            logger.warning(f"tool directory not found: {tool_dir}")
        elif logger:
            logger(f"tool directory not found: {tool_dir}")
        return 0

    # Discover tool files
    tool_files = _discover_tool_files(tool_dir, tool_type)

    if not tool_files:
        if logger and hasattr(logger, "warning"):
            logger.warning(f"No tool files found in: {tool_dir}")
        elif logger:
            logger(f"No tool files found in: {tool_dir}")
        return 0

    # Register each tool
    registered_count = 0
    for tool_file in tool_files:
        if _register_single_tool(
            controller,
            tool_file,
            llm,
            page_extraction_llm,
            logger,
            fallback_to_agent,
        ):
            registered_count += 1

    return registered_count
