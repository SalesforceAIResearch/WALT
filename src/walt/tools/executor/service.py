from __future__ import annotations

import asyncio
import json
import json as _json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from typing import cast as _cast

from walt.browser_use import Agent, Browser
from walt.browser_use.agent.views import ActionResult, AgentHistoryList
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from walt.tools.registry.service import ToolController
from walt.tools.registry.utils import get_best_element_handle

# Import from discovery utils
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../discovery"))
from walt.tools.discovery.utils import evaluate_agent_result_with_tolerance
from walt.tools.schema.views import (
    AgenticToolStep,
    DeterministicToolStep,
    ToolDefinitionSchema,
    ToolInputSchemaDefinition,
    ToolStep,
)
from walt.prompts.tool_executor import (
    get_tool_executor_step_prompt,
    get_structured_output_prompt,
    get_tool_fallback_prompt,
)

# Load prompts from centralized location
AGENT_STEP_SYSTEM_PROMPT = get_tool_executor_step_prompt()
STRUCTURED_OUTPUT_PROMPT = get_structured_output_prompt()
from walt.tools.executor.step_agent.controller import toolStepAgentController
from walt.tools.executor.views import ToolRunOutput

logger = logging.getLogger(__name__)

WAIT_FOR_ELEMENT_TIMEOUT = 2500


@dataclass
class ToolExecutionConfig:
    """Configuration for tool execution timing and delays"""
    # Inter-step delay (between each step execution)
    inter_step_delay: float = 0.0  # Reduced from 0.1s
    
    # Post-navigation delays
    post_navigation_buffer: float = 0.3  # Reduced from 1.5-2.0s
    fallback_wait: float = 0.5  # Reduced from 2.0s
    
    # Agentic step delays
    agent_step_base_wait: float = 0.5  # Reduced from 3-8s
    agent_step_fallback: float = 0.5  # Reduced from 2.0s
    
    # Debug delays
    single_step_keep_alive: float = 0.0  # Removed 5s debug delay
    
    # Use network idle detection (more reliable than fixed delays)
    use_network_idle: bool = True
    network_idle_timeout: int = 3000  # ms
    
    # Network stability check before each step (usually redundant with post-action checks)
    check_network_before_step: bool = False  # Disabled for performance (post-action check is sufficient)


# Global config - can be overridden
DEFAULT_TOOL_CONFIG = ToolExecutionConfig()

T = TypeVar("T", bound=BaseModel)


class Tool:
    """Simple orchestrator that executes a list of tool *steps* defined in a ToolDefinitionSchema."""

    def __init__(
        self,
        tool_schema: ToolDefinitionSchema,
        llm: BaseChatModel,
        *,
        controller: ToolController | None = None,
        browser: Browser | None = None,
        page_extraction_llm: BaseChatModel | None = None,
        fallback_to_agent: bool = False,
        config: ToolExecutionConfig | None = None,
    ) -> None:
        """Initialize a new tool instance from a schema object.

        Args:
                tool_schema: The parsed tool definition schema.
                controller: Optional ToolController instance to handle action execution
                browser: Optional Browser instance to use for browser automation
                llm: Optional language model for fallback agent functionality
                fallback_to_agent: Whether to fall back to agent-based execution on step failure
                config: Optional ToolExecutionConfig for timing/delays (uses default if None)

        Raises:
                ValueError: If the tool schema is invalid (though Pydantic handles most).
        """
        self.schema = tool_schema  # Store the schema object
        self.config = config or DEFAULT_TOOL_CONFIG

        self.controller = controller or ToolController()

        self._browser = None
        self.browser = browser or Browser()  # This will trigger the setter

        self.llm = llm
        self.page_extraction_llm = page_extraction_llm

        self.fallback_to_agent = fallback_to_agent

        self.context: dict[str, Any] = {}

        self.inputs_def: List[ToolInputSchemaDefinition] = self.schema.input_schema
        self._input_model: type[BaseModel] = self._build_input_model()

    @property
    def description(self):
        """Get the tool description from schema."""
        return self.schema.description

    @property
    def steps(self):
        """Get the tool steps from schema."""
        return self.schema.steps

    @property
    def browser(self):
        """Get the browser instance."""
        return self._browser

    @browser.setter
    def browser(self, value):
        """Set the browser instance and apply compatibility patches."""
        self._browser = value
        if self._browser:
            self._add_browser_compatibility()

            # Apply browser keep-alive settings
            if hasattr(self._browser, "config") and hasattr(
                self._browser.config, "_force_keep_browser_alive"
            ):
                self._browser.config._force_keep_browser_alive = True
            elif hasattr(self._browser, "browser_profile"):
                self._browser.browser_profile.keep_alive = True

    def _add_browser_compatibility(self):
        """Add missing methods to browser for compatibility with older browser-use versions."""
        # Add start method if missing
        if not hasattr(self.browser, "start"):

            async def start():
                pass

            self.browser.start = start

        # Add _wait_for_stable_network method if missing
        if not hasattr(self.browser, "_wait_for_stable_network"):

            async def _wait_for_stable_network():
                pass

            self.browser._wait_for_stable_network = _wait_for_stable_network

        # Add async context manager support if missing
        if not hasattr(self.browser, "__aenter__"):

            async def __aenter__():
                return self.browser

            async def __aexit__(exc_type, exc_val, exc_tb):
                pass

            self.browser.__aenter__ = __aenter__
            self.browser.__aexit__ = __aexit__

        # Add get_current_page method if missing (this requires special handling)
        if not hasattr(self.browser, "get_current_page"):
            # If this browser was extracted from a context, we should have stored the context
            async def get_current_page():
                # If we have a stored context, use it
                if hasattr(self.browser, "_original_context"):
                    return await self.browser._original_context.get_current_page()
                else:
                    # Fallback: create a minimal context and get page
                    from walt.browser_use.browser.context import (
                        BrowserContext,
                        BrowserContextConfig,
                    )

                    context = BrowserContext(
                        browser=self.browser, config=BrowserContextConfig()
                    )
                    return await context.get_current_page()

            self.browser.get_current_page = get_current_page

    # --- Loaders ---
    @classmethod
    def load_from_file(
        cls,
        file_path: str | Path,
        llm: BaseChatModel,
        *,
        controller: ToolController | None = None,
        browser: Browser | None = None,
        page_extraction_llm: BaseChatModel | None = None,
        fallback_to_agent: bool = False,
    ) -> tool:
        """Load a tool from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        tool_schema = ToolDefinitionSchema(**data)
        return tool(
            tool_schema=tool_schema,
            controller=controller,
            browser=browser,
            llm=llm,
            page_extraction_llm=page_extraction_llm,
            fallback_to_agent=fallback_to_agent,
        )

    # --- Runners ---
    async def _run_deterministic_step(
        self, step: DeterministicToolStep, step_index: int
    ) -> ActionResult:
        """Execute a deterministic (controller) action based on step dictionary."""
        # Assumes ToolStep for deterministic type has 'action' and 'params' keys
        action_name: str = step.type  # Expect 'action' key for deterministic steps
        params: Dict[str, Any] = step.model_dump()  # Use params if present

        # Special handling for navigation with url_operation
        if (
            action_name == "navigation"
            and "url_operation" in params
            and params["url_operation"]
        ):
            url_operation = params["url_operation"]

            # Use controller's URL operation logic for proper system variable resolution
            if isinstance(url_operation, dict):
                try:
                    # Get browser context for URL operations
                    browser_context = getattr(self.browser, "_original_context", None)
                    if browser_context:
                        # Let controller handle URL operation with system variables
                        resolved_url = await self.controller._apply_url_operation(
                            url_operation, browser_context, self.context
                        )
                        # Replace url_operation with resolved URL
                        params["url"] = resolved_url
                        params.pop("url_operation", None)
                    else:
                        logger.warning("No browser context available for URL operation")
                        # Fall back to basic placeholder resolution for user variables only
                        safe_context = {
                            k: v for k, v in self.context.items() if v is not None
                        }

                        # Resolve user placeholders in replace operations
                        if "replace" in url_operation and isinstance(
                            url_operation["replace"], dict
                        ):
                            resolved_replace = {}
                            for param_name, value in url_operation["replace"].items():
                                if (
                                    isinstance(value, str)
                                    and "{" in value
                                    and value not in ["{current_url}", "{current_page}"]
                                ):
                                    try:
                                        resolved_replace[param_name] = value.format(
                                            **safe_context
                                        )
                                    except KeyError:
                                        resolved_replace[param_name] = value
                                else:
                                    resolved_replace[param_name] = value
                            url_operation["replace"] = resolved_replace

                        params["url_operation"] = url_operation
                except Exception as e:
                    logger.warning(f"Failed to apply URL operation: {e}")
                    params["url_operation"] = url_operation

        ActionModel = self.controller.registry.create_action_model(
            include_actions=[action_name]
        )
        # Pass the params dictionary directly
        action_model = ActionModel(**{action_name: params})

        try:
            # Pass the original context if available, otherwise create one
            browser_context = getattr(self.browser, "_original_context", None)
            if not browser_context:
                from walt.browser_use.browser.context import (
                    BrowserContext,
                    BrowserContextConfig,
                )

                browser_context = BrowserContext(
                    browser=self.browser, config=BrowserContextConfig()
                )

            result = await self.controller.act(
                action_model,
                browser_context,
                page_extraction_llm=self.page_extraction_llm,
            )
        except Exception as e:
            raise RuntimeError(f"Deterministic action '{action_name}' failed: {str(e)}")

        # Wait for network stability after completing the action
        await self.browser._wait_for_stable_network()

        # Helper function to truncate long selectors in logs
        def truncate_selector(selector: str) -> str:
            return selector if len(selector) <= 45 else f"{selector[:45]}..."

        # Determine if this is not the last step, and extract next step's cssSelector if available
        current_index = step_index
        if current_index < len(self.schema.steps) - 1:
            next_step = self.schema.steps[current_index + 1]
            next_step_resolved = self._resolve_placeholders(next_step)
            css_selector = getattr(next_step_resolved, "cssSelector", None)
            if css_selector:
                try:
                    await self.browser._wait_for_stable_network()
                    page = await self.browser.get_current_page()

                    logger.info(
                        f"Waiting for element with selector: {truncate_selector(css_selector)}"
                    )
                    locator, selector_used = await get_best_element_handle(
                        page,
                        css_selector,
                        next_step_resolved,
                        timeout_ms=WAIT_FOR_ELEMENT_TIMEOUT,
                    )
                    logger.info(
                        f"Element with selector found: {truncate_selector(selector_used)}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to wait for element with selector: {truncate_selector(css_selector)}. Error: {e}"
                    )
                    raise Exception(
                        f"Failed to wait for element. Selector: {css_selector}"
                    ) from e

        return result

    def _format_agent_step_context(
        self, current_step: AgenticToolStep, step_index: int
    ) -> str:
        """Format the tool step context for the agent with extended context (last 2, current, next 2 steps)."""

        def format_step_info(step: ToolStep, step_num: int) -> str:
            """Format step information consistently."""
            info = [f"Step {step_num}: Type: {step.type}"]
            if step.description:
                info.append(f"Description: {step.description}")
            # For agent steps, show the task
            if isinstance(step, AgenticToolStep):
                info.append(f"Task: {step.task}")
            return "\n".join(info)

        sections = []
        total_steps = len(self.schema.steps)

        # Add previous steps context (last 2 steps)
        prev_steps = []
        for i in range(max(0, step_index - 2), step_index):
            prev_step = self.schema.steps[i]
            prev_steps.append(format_step_info(prev_step, i + 1))

        if prev_steps:
            sections.extend(
                [
                    "=== PREVIOUS STEPS (FOR CONTEXT ONLY) ===",
                    "\n\n".join(prev_steps),
                    "",
                ]
            )

        # Add current step context
        sections.extend(
            [
                "=== CURRENT STEP (YOUR TASK) ===",
                format_step_info(current_step, step_index + 1),
                "",
            ]
        )

        # Add next steps context (next 2 steps)
        next_steps = []
        for i in range(step_index + 1, min(total_steps, step_index + 3)):
            next_step = self.schema.steps[i]
            next_steps.append(format_step_info(next_step, i + 1))

        if next_steps:
            sections.extend(
                [
                    "=== NEXT STEPS (FOR CONTEXT ONLY) ===",
                    "\n\n".join(next_steps),
                ]
            )

        return "\n".join(sections)

    async def _run_agent_step(
        self, step: AgenticToolStep, step_index: int
    ) -> AgentHistoryList:
        """Spin-up an Agent based on step dictionary."""
        # Ensure page is stable before agent step execution
        logger.info(f"Waiting for page stability before agent step {step_index + 1}")

        try:
            # Wait for network to stabilize and page to be fully loaded
            await self.browser._wait_for_stable_network()

            # Intelligent wait times based on task type (using config)
            task_lower = step.task.lower()
            base_wait = self.config.agent_step_base_wait

            if any(
                keyword in task_lower for keyword in ["dropdown", "select", "mark as"]
            ):
                # Longer wait for dropdown interactions
                base_wait = self.config.agent_step_base_wait * 1.5
                logger.info(
                    f"Extended wait for dropdown interaction in step {step_index + 1}"
                )
            elif any(
                keyword in task_lower for keyword in ["checkbox", "verify", "ensure"]
            ):
                # Extra wait for state verification tasks
                base_wait = self.config.agent_step_base_wait * 1.2
                logger.info(
                    f"Extended wait for state verification in step {step_index + 1}"
                )
            elif any(keyword in task_lower for keyword in ["form", "input", "fill"]):
                # Wait for form elements to be ready
                base_wait = self.config.agent_step_base_wait
                logger.info(
                    f"Extended wait for form interaction in step {step_index + 1}"
                )

            await asyncio.sleep(base_wait)

            # Ensure page is in a good state for interaction
            page = await self.browser.get_current_page()
            await page.wait_for_load_state("domcontentloaded")

            # Additional wait for network idle specifically for dynamic content
            if self.config.use_network_idle:
                try:
                    await page.wait_for_load_state("networkidle", timeout=self.config.network_idle_timeout)
                except Exception:
                    logger.debug(
                        f"Network idle timeout for step {step_index + 1}, continuing..."
                    )

            logger.info(f"Page stability confirmed for agent step {step_index + 1}")

        except Exception as e:
            logger.warning(
                f"Page stability wait failed for agent step {step_index + 1}: {e}"
            )
            # Continue anyway but with a fallback wait
            await asyncio.sleep(self.config.agent_step_fallback)

        # Create contextual task with extended context (last 2, current, next 2 steps)
        contextual_task = self._format_agent_step_context(step, step_index)

        # logger.info(f'Contextual task: {contextual_task}')

        # 		task = """
        # {step.task}

        # Please do not make up any fake data.
        # """

        # Get the existing browser context to ensure agent uses the same context as tool
        browser_context = getattr(self.browser, "_original_context", None)

        agent = Agent(
            task=step.task,  # Only the current step task goes into ultimate task
            message_context=contextual_task,  # Extended context with surrounding steps
            llm=self.llm,
            browser=self.browser,
            browser_context=browser_context,  # Pass existing context to prevent new context creation
            controller=toolStepAgentController(),
            # use_vision=True,  # Consider making this configurable via ToolStep schema
            override_system_message=AGENT_STEP_SYSTEM_PROMPT,
        )

        return await agent.run()

    async def _fallback_to_agent(
        self,
        step_resolved: ToolStep,
        step_index: int,
        error: Exception | str | None = None,
    ) -> AgentHistoryList:
        """Handle step failure by delegating to an agent."""
        from walt.tools.schema.views import AgenticToolStep

        # Extract details from the failed step dictionary
        failed_action_name = step_resolved.type
        failed_params = step_resolved.model_dump()
        step_description = step_resolved.description or "No description provided"
        error_msg = str(error) if error else "Unknown error"
        total_steps = len(self.steps)
        fail_details = (
            f"step={step_index + 1}/{total_steps}, action='{failed_action_name}', "
            f"description='{step_description}', params={str(failed_params)}, error='{error_msg}'"
        )

        # Determine the failed_value based on step type and attributes
        failed_value = None
        description_prefix = (
            f"Purpose: {step_description}. " if step_description else ""
        )

        # Create a generic fallback description since we don't have specific step types imported
        failed_value = f"{description_prefix}Execute action '{failed_action_name}' with parameters: {failed_params}"

        # Build the fallback task with the failed_value
        fallback_task = get_tool_fallback_prompt(
            step_index=step_index + 1,
            total_steps=len(self.steps),
            action_type=failed_action_name,
            fail_details=fail_details,
            failed_value=failed_value,
            step_description=step_description,
        )
        logger.info(f"Agent fallback task: {fallback_task}")

        # Prepare agent step config based on the failed step, adding task
        agent_step_config = AgenticToolStep(
            type="agent",
            task=fallback_task,
            max_steps=5,
            output=None,
            description="Fallback agent to handle step failure",
        )

        return await self._run_agent_step(agent_step_config, step_index)

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate provided inputs against the tool's input schema definition."""
        # If no inputs are defined in the schema, no validation needed
        if not self.inputs_def:
            return

        try:
            # Let Pydantic perform the heavy lifting – this covers both presence and
            # type validation based on the JSON schema model.
            self._input_model(**inputs)
        except Exception as e:
            raise ValueError(f"Invalid tool inputs: {e}") from e

    def _resolve_placeholders(self, data: Any) -> Any:
        """Recursively replace placeholders in *data* using current context variables.

        String placeholders are written using Python format syntax, e.g. "{index}".
        """
        if isinstance(data, str):
            try:
                # Only attempt to format if placeholder syntax is likely present
                if "{" in data and "}" in data:
                    # Create safe context that converts None values to empty strings for URLs
                    safe_context = {}
                    for k, v in self.context.items():
                        if v is None:
                            safe_context[k] = ""
                        else:
                            safe_context[k] = v
                    
                    formatted_data = data.format(**safe_context)
                    
                    # Clean up URLs by removing empty query parameters and fixing common issues
                    if "://" in formatted_data or formatted_data.startswith("/"):
                        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
                        try:
                            # Note: URL path fixing was removed here due to risk of breaking other tools
                            # Root cause should be fixed in tool parameter construction instead
                            
                            # Clean query parameters if present
                            if "?" in formatted_data and ("=" in formatted_data):
                                parsed = urlparse(formatted_data)
                                if parsed.query:
                                    # Parse query parameters and remove empty ones
                                    query_params = parse_qs(parsed.query, keep_blank_values=False)
                                    # Remove empty values and rebuild query string
                                    clean_params = {k: v for k, v in query_params.items() if v and v[0]}
                                    clean_query = urlencode(clean_params, doseq=True)
                                    # Rebuild the URL with clean query string
                                    formatted_data = urlunparse(parsed._replace(query=clean_query))
                        except Exception:
                            # If URL parsing fails, just return the formatted string
                            pass
                    
                    return formatted_data
                return data  # No placeholders, return as is
            except KeyError:
                # A key in the placeholder was not found in the context.
                # Return the original string as per previous behavior.
                return data

        # TODO: This next things are not really supported atm, we'll need to to do it in the future.
        elif isinstance(data, list):
            new_list = []
            changed = False
            for item in data:
                resolved_item = self._resolve_placeholders(item)
                if resolved_item is not item:
                    changed = True
                new_list.append(resolved_item)
            return new_list if changed else data
        elif isinstance(data, dict):
            new_dict = {}
            changed = False
            for key, value in data.items():
                resolved_value = self._resolve_placeholders(value)
                if resolved_value is not value:
                    changed = True
                new_dict[key] = resolved_value
            return new_dict if changed else data
        elif isinstance(data, BaseModel):  # Handle Pydantic models
            update_dict = {}
            model_changed = False
            for field_name in data.model_fields:  # Iterate using model_fields keys
                original_value = getattr(data, field_name)
                resolved_value = self._resolve_placeholders(original_value)
                if resolved_value is not original_value:
                    model_changed = True
                update_dict[field_name] = resolved_value

            if model_changed:
                return data.model_copy(update=update_dict)
            else:
                return data  # Return original instance if no field's value changed
        else:
            # For any other types (int, float, bool, None, etc.), return as is
            return data

    def _should_skip_step(self, step_dict: dict) -> bool:
        """Check if step should be skipped due to optional fields with None/null values."""
        import re

        # Convert step dict to string to find all placeholders
        step_str = str(step_dict)

        # Find all placeholders like {fieldName}
        placeholders = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", step_str)

        if not placeholders:
            return False  # No placeholders, don't skip

        # Check each placeholder against the input schema
        has_required_fields = False
        num_optional_fields = 0
        num_optional_fields_with_non_null_values = 0
        
        for field_name in placeholders:
            # Find the field in the input schema
            field_schema = None
            for input_def in self.inputs_def:
                if input_def.name == field_name:
                    field_schema = input_def
                    break

            if field_schema is None:
                continue  # Field not in schema, continue with execution

            if field_schema.required:
                has_required_fields = True
            else:
                # Count optional fields and their values
                field_value = self.context.get(field_name)
                if not (
                    field_value is None or field_value == "None" or field_value == ""
                ):
                    num_optional_fields_with_non_null_values += 1
                num_optional_fields += 1
        
        # Only skip if:
        # 1. There are NO required fields in this step, AND
        # 2. There are optional fields, AND 
        # 3. ALL optional fields are null
        condition = (
            not has_required_fields and  # No required fields in this step
            num_optional_fields > 0 and  # At least one optional field exists
            num_optional_fields_with_non_null_values == 0  # All optional fields are null
        )
        if condition:
            logger.info(f"Skipping step - no required fields and all optional fields are null")
        return condition

    def _store_output(self, step_cfg: ToolStep, result: Any) -> None:
        """Store output into context based on 'output' key in step dictionary."""
        # Assumes ToolStep schema includes an optional 'output' field (string)
        output_key = step_cfg.output
        if not output_key:
            return

        # Helper to extract raw content we want to store

        value: Any = None

        if isinstance(result, ActionResult):
            # Prefer JSON in extracted_content if available
            content = result.extracted_content
            if content is None:
                value = {
                    "success": result.success,
                    "is_done": result.is_done,
                }
            else:
                try:
                    value = json.loads(content)
                except Exception:
                    value = content
        elif isinstance(result, AgentHistoryList):
            # Try to pull last ActionResult with extracted_content
            try:
                last_item = result.history[-1]
                last_action_result = next(
                    (
                        r
                        for r in reversed(last_item.result)
                        if r.extracted_content is not None
                    ),
                    None,
                )
                if last_action_result and last_action_result.extracted_content:
                    try:
                        value = json.loads(last_action_result.extracted_content)
                    except Exception:
                        value = last_action_result.extracted_content
            except Exception:
                value = None
        else:
            value = str(result)

        self.context[output_key] = value

    async def _execute_step(
        self, step_index: int, step_resolved: ToolStep
    ) -> ActionResult | AgentHistoryList:
        """Execute the resolved step dictionary, handling type branching and fallback."""
        # Use 'type' field from the ToolStep dictionary
        result: ActionResult | AgentHistoryList

        if isinstance(step_resolved, DeterministicToolStep):
            from walt.browser_use.agent.views import ActionResult  # Local import ok

            try:
                # Use action key from step dictionary
                action_name = step_resolved.type or "[No action specified]"
                logger.info(f"Attempting deterministic action: {action_name}")
                result = await self._run_deterministic_step(step_resolved, step_index)

                # Add enhanced wait conditions after navigation steps
                if action_name == "navigation":
                    logger.info(
                        f"Adding enhanced wait after navigation step {step_index + 1}"
                    )
                    try:
                        # Extra wait for navigation to ensure page is fully ready for subsequent steps
                        await self.browser._wait_for_stable_network()

                        # Wait for any dynamic content to load after navigation
                        page = await self.browser.get_current_page()
                        if self.config.use_network_idle:
                            await page.wait_for_load_state("networkidle", timeout=5000)

                        # Additional buffer for complex pages with AJAX content (optimized)
                        await asyncio.sleep(self.config.post_navigation_buffer)

                        logger.info(
                            f"Enhanced navigation wait completed for step {step_index + 1}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Enhanced navigation wait failed for step {step_index + 1}: {e}"
                        )
                        # Fallback to basic wait
                        await asyncio.sleep(self.config.fallback_wait)

                if isinstance(result, ActionResult) and result.error:
                    logger.warning(
                        f"Deterministic action reported error: {result.error}"
                    )
                    raise ValueError(
                        f"Deterministic action {action_name} failed: {result.error}"
                    )

            except Exception as e:
                action_name = step_resolved.type or "[Unknown Action]"
                logger.warning(
                    f"Deterministic step {step_index + 1} ({action_name}) failed: {e}"
                )

                # Use fallback if enabled, otherwise fail fast for testing
                if self.fallback_to_agent:
                    logger.info(
                        f"Attempting agent fallback for failed deterministic step {step_index + 1}"
                    )
                    try:
                        result = await self._fallback_to_agent(
                            step_resolved, step_index, e
                        )
                    except Exception as fallback_error:
                        logger.error(f"Agent fallback also failed: {fallback_error}")
                        raise ValueError(
                            f"Both deterministic step and agent fallback failed for step {step_index + 1}: {e}"
                        )
                else:
                    # Fallbacks disabled for deterministic testing
                    raise ValueError(
                        f"Deterministic step {step_index + 1} ({action_name}) failed: {e}"
                    )

        elif isinstance(step_resolved, AgenticToolStep):
            # Use task key from step dictionary
            task_description = step_resolved.task
            logger.info(f"Running agent task: {task_description}")
            try:
                result = await self._run_agent_step(step_resolved, step_index)

                # Enhanced agent evaluation using shared logic from discovery utils
                success, reason = evaluate_agent_result_with_tolerance(result)

                if not success:
                    logger.warning(
                        f"Agent step {step_index + 1} failed evaluation: {reason}"
                    )
                    raise ValueError(f"Agent step {step_index + 1} failed evaluation.")
                elif (
                    "marked itself as failed but produced meaningful results" in reason
                ):
                    logger.info(
                        f"Agent step {step_index + 1}: {reason}. Treating as success."
                    )

            except Exception as e:
                # For explicit agent steps, just re-raise the original exception
                # (don't mention fallback since this IS the intended agent step)
                raise ValueError(f"Agent step {step_index + 1} failed: {e}")

        return result

    # --- Convert all extracted stuff to final output model ---
    async def _convert_results_to_output_model(
        self,
        results: List[ActionResult | AgentHistoryList],
        output_model: type[T],
    ) -> T:
        """Convert tool results to a specified output model.

        Filters ActionResults with extracted_content, then uses LangChain to parse
        all extracted texts into the structured output model.

        Args:
                results: List of tool step results
                output_model: Target Pydantic model class to convert to

        Returns:
                An instance of the specified output model
        """
        if not results:
            raise ValueError("No results to convert")

        # Extract all content from ActionResults
        extracted_contents = []

        for result in results:
            if isinstance(result, ActionResult) and result.extracted_content:
                extracted_contents.append(result.extracted_content)
            # TODO: this might be incorrect; but it helps A LOT if extract fucks up and only the agent is able to solve it
            elif isinstance(result, AgentHistoryList):
                # Check the agent history for any extracted content
                for item in result.history:
                    for action_result in item.result:
                        if action_result.extracted_content:
                            extracted_contents.append(action_result.extracted_content)

        if not extracted_contents:
            raise ValueError("No extracted content found in tool results")

        # Combine all extracted contents
        combined_text = "\n\n".join(extracted_contents)

        messages: list[BaseMessage] = [
            AIMessage(content=STRUCTURED_OUTPUT_PROMPT),
            HumanMessage(content=combined_text),
        ]

        chain = self.llm.with_structured_output(output_model)
        chain_result: T = await chain.ainvoke(messages)  # type: ignore

        return chain_result

    async def run_step(self, step_index: int, inputs: dict[str, Any] | None = None):
        """Run a *single* tool step asynchronously and return its result.

        Parameters
        ----------
        step_index:
                        Zero-based index of the step to execute.
        inputs:
                        Optional tool-level inputs.  If provided on the first call they
                        are validated and injected into :pyattr:`context`.  Subsequent
                        calls can omit *inputs* as :pyattr:`context` is already populated.
        """
        if not (0 <= step_index < len(self.schema.steps)):
            raise IndexError(
                f"step_index {step_index} is out of range for tool with {len(self.schema.steps)} steps"
            )

        # Initialise/augment context once with the provided inputs
        if inputs is not None or not self.context:
            runtime_inputs = inputs or {}
            self._validate_inputs(runtime_inputs)
            # If context is empty we assume this is the first invocation – start fresh;
            # otherwise merge new inputs on top (explicitly overriding duplicates)
            if not self.context:
                self.context = runtime_inputs.copy()
            else:
                self.context.update(runtime_inputs)

        async with self.browser:
            raw_step_cfg = self.schema.steps[step_index]
            step_resolved = self._resolve_placeholders(raw_step_cfg)
            result = await self._execute_step(step_index, step_resolved)
            # Persist outputs (if declared) for future steps
            self._store_output(step_resolved, result)
            # Optional keep-alive delay (disabled by default for performance)
            if self.config.single_step_keep_alive > 0:
                await asyncio.sleep(self.config.single_step_keep_alive)
        # Each invocation opens a new browser context – we close the browser to
        # release resources right away.  This keeps the single-step API
        # self-contained.
        try:
            # Close browser context first if available
            if hasattr(self.browser, "_original_context"):
                await self.browser._original_context.close()
            # Then close browser
            await self.browser.close()
        except Exception as e:
            print(f"Warning: Failed to close browser resources: {e}")
        return result

    async def run(
        self,
        inputs: dict[str, Any] | None = None,
        close_browser_at_end: bool = True,
        cancel_event: asyncio.Event | None = None,
        output_model: type[T] | None = None,
    ) -> ToolRunOutput[T]:
        """Execute the tool asynchronously using step dictionaries.

        @dev This is the main entry point for the tool.

        Args:
                inputs: Optional dictionary of tool inputs
                close_browser_at_end: Whether to close the browser when done
                cancel_event: Optional event to signal cancellation
                output_model: Optional Pydantic model class to convert results to

        Returns:
                Either ToolRunOutput containing all step results or an instance of output_model if provided
        """
        runtime_inputs = inputs or {}
        # 1. Validate inputs against definition
        self._validate_inputs(runtime_inputs)
        # 2. Initialize context with validated inputs
        self.context = runtime_inputs.copy()  # Start with a fresh context

        results: List[ActionResult | AgentHistoryList] = []

        await self.browser.start()
        try:
            for step_index, step_dict in enumerate(
                self.schema.steps
            ):  # self.steps now holds dictionaries
                # Optional inter-step delay (minimized for performance)
                if self.config.inter_step_delay > 0:
                    await asyncio.sleep(self.config.inter_step_delay)
                
                # Optional network stability check (disabled by default as post-action checks are sufficient)
                if self.config.check_network_before_step:
                    await self.browser._wait_for_stable_network()

                # Check if cancellation was requested
                if cancel_event and cancel_event.is_set():
                    logger.info("Cancellation requested - stopping tool execution")
                    break

                # Use description from the step dictionary
                step_description = step_dict.description or "No description provided"
                logger.info(
                    f"--- Running Step {step_index + 1}/{len(self.steps)} -- {step_description} ---"
                )

                # Check if step should be skipped due to optional fields with no values
                if self._should_skip_step(step_dict):
                    # Create a dummy successful result for skipped steps
                    from walt.browser_use.agent.views import ActionResult

                    result = ActionResult(
                        is_done=False,
                        extracted_content="Step skipped - optional field has no value",
                    )
                    results.append(result)
                    logger.info(f"--- Finished Step {step_index + 1} (Skipped) ---\n")
                    continue

                # Resolve placeholders using the current context (works on the dictionary)
                step_resolved = self._resolve_placeholders(step_dict)

                # Execute step using the unified _execute_step method
                result = await self._execute_step(step_index, step_resolved)

                results.append(result)
                # Persist outputs using the resolved step dictionary
                self._store_output(step_resolved, result)
                logger.info(f"--- Finished Step {step_index + 1} ---\n")

            # Convert results to output model if requested
            output_model_result: T | None = None
            if output_model:
                output_model_result = await self._convert_results_to_output_model(
                    results, output_model
                )

        finally:
            # Clean-up browser after finishing tool
            if close_browser_at_end:
                self.browser.browser_profile.keep_alive = False
                await self.browser.close()

        return ToolRunOutput(step_results=results, output_model=output_model_result)

    # ------------------------------------------------------------------
    # LangChain tool wrapper
    # ------------------------------------------------------------------

    def _build_input_model(self) -> type[BaseModel]:
        """Return a *pydantic* model matching the tool's ``input_schema`` section.

        This creates a dynamic Pydantic model that includes format information in field
        descriptions, making format requirements visible to LLMs when tools are used as tools.
        """

        if not self.inputs_def:
            # No declared inputs -> generate an empty model
            # Use schema name for uniqueness, fallback if needed
            model_name = (
                f'{(self.schema.name or "tool").replace(" ", "_")}_NoInputs'
            )
            return create_model(model_name)

        # Map tool input types to Python types
        # For string type, use a custom type that allows coercion from numbers
        from typing import Annotated
        from pydantic import BeforeValidator

        def coerce_to_string(v):
            """Convert numbers to strings, leave strings as-is"""
            if isinstance(v, (int, float)):
                return str(v)
            return v

        FlexibleString = Annotated[str, BeforeValidator(coerce_to_string)]

        type_mapping = {
            "string": FlexibleString,
            "number": float,
            "bool": bool,
        }

        # Build fields dictionary for create_model()
        fields: Dict[str, tuple[type, Any]] = {}

        for input_def in self.inputs_def:
            name = input_def.name
            type_str = input_def.type

            # Check for enum constraints
            if hasattr(input_def, "enum") and input_def.enum:
                from typing import Literal

                py_type = Literal[tuple(input_def.enum)]
            else:
                py_type = type_mapping.get(type_str)
                if py_type is None:
                    raise ValueError(
                        f"Unsupported input type: {type_str!r} for field {name!r}"
                    )

            if input_def.required:
                # Required fields use ... (Ellipsis) and the base type
                default = ...
                field_type = py_type
            else:
                # Optional fields use None as default and Optional[Type]
                default = None
                field_type = Optional[py_type]

            fields[name] = (field_type, default)

            # Create field description with format information if available
            # This helps LLMs understand expected input formats when tool is used as a tool
            field_description = None
            description_parts = []
            if hasattr(input_def, "description") and input_def.description:
                description_parts.append(input_def.description)
            if hasattr(input_def, "format") and input_def.format:
                description_parts.append(f"Format: {input_def.format}")
            if description_parts:
                field_description = " ".join(description_parts)

            # Build field tuple: (type, default_or_field_info)
            # Pydantic's create_model uses ... (Ellipsis) to mark required fields
            if input_def.required:
                if field_description:
                    # Required field with format description
                    fields[name] = (py_type, Field(..., description=field_description))
                else:
                    # Required field without format description
                    fields[name] = (py_type, ...)
            else:
                if field_description:
                    # Optional field with format description
                    fields[name] = (
                        field_type,
                        Field(None, description=field_description),
                    )
                else:
                    # Optional field without format description
                    fields[name] = (field_type, None)

        # The raw ``create_model`` helper from Pydantic deliberately uses *dynamic*
        # signatures, which the static type checker cannot easily verify.  We cast
        # the **fields** mapping to **Any** to silence these warnings.
        return create_model(  # type: ignore[arg-type]
            f'{(self.schema.name or "tool").replace(" ", "_")}_Inputs',
            **_cast(Dict[str, Any], fields),
        )

    def as_tool(
        self, *, name: str | None = None, description: str | None = None
    ):  # noqa: D401
        """Expose the entire tool as a LangChain *StructuredTool* instance.

        The generated tool validates its arguments against the tool's input
        schema (if present) and then returns the JSON-serialised output of
        :py:meth:`run`.
        """

        InputModel = self._build_input_model()
        # Use schema name as default, sanitize for tool name requirements
        default_name = "".join(c if c.isalnum() else "_" for c in self.schema.name)
        tool_name = name or default_name[:50]
        doc = description or self.schema.description  # Use schema description

        # `self` is closed over via the inner function so we can keep state.
        async def _invoke(**kwargs):  # type: ignore[override]
            logger.info(f"Running tool as tool with inputs: {kwargs}")
            augmented_inputs = kwargs.copy() if kwargs else {}
            for input_def in self.inputs_def:
                if not input_def.required and input_def.name not in augmented_inputs:
                    augmented_inputs[input_def.name] = ""
            result = await self.run(inputs=augmented_inputs)
            # Serialise non-string output so models that expect a string tool
            # response still work.
            try:
                return _json.dumps(result, default=str)
            except Exception:
                return str(result)

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=tool_name,
            description=doc,
            args_schema=InputModel,
        )

    async def run_as_tool(self, prompt: str) -> str:
        """
        Run the tool with a prompt and automatically parse the required variables.

        @dev Uses AgentExecutor to properly handle the tool invocation loop.
        """

        # For now I kept it simple but one could think of using a react agent here.

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create the tool tool
        tool_tool = self.as_tool()
        agent = create_tool_calling_agent(self.llm, [tool_tool], prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=[tool_tool])
        result = await agent_executor.ainvoke({"input": prompt})
        return result["output"]
