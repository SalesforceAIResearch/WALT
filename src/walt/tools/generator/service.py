import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from walt.prompts.discovery import get_tool_builder_prompt

# Load tool builder prompt from centralized location
tool_BUILDER_PROMPT_TEMPLATE = get_tool_builder_prompt()
from walt.tools.registry.service import ToolController
from walt.tools.schema.views import ToolDefinitionSchema

logger = logging.getLogger(__name__)


class BuilderService:
    """
    Service responsible for building executable tool JSON definitions
    from recorded browser session events using an LLM.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initializes the BuilderService.

        Args:
            llm: A LangChain BaseChatModel instance configured for use.
                 It should ideally support vision capabilities if screenshots are used.
        """
        if llm is None:
            raise ValueError("A BaseChatModel instance must be provided.")

        # Configure the LLM to return structured output based on the Pydantic model
        try:
            # Specify method="function_calling" for better compatibility
            self.llm_structured = llm.with_structured_output(
                ToolDefinitionSchema, method="function_calling"
            )
        except NotImplementedError:
            logger.warning(
                "LLM does not support structured output natively. Falling back."
            )
            # Basic LLM call if structured output is not supported
            # Output parsing will be handled manually later
            self.llm_structured = llm  # Store the original llm

        self.prompt_template = PromptTemplate.from_template(
            tool_BUILDER_PROMPT_TEMPLATE
        )
        self.actions_markdown = self._get_available_actions_markdown()
        logger.info("BuilderService initialized.")

    @staticmethod
    def _get_available_actions_markdown() -> str:
        """Return a markdown list of available actions and their schema."""
        controller = ToolController()
        lines: List[str] = []
        for action in controller.registry.registry.actions.values():
            # Only include deterministic actions relevant for building from recordings
            # Exclude agent-specific or meta-actions if necessary
            # Based on schema/views.py, the recorder types seem to map directly
            # to controller action *names*, but the prompt uses the event `type` field.
            # Let's assume the prompt template correctly lists the *event types* expected.
            # This function provides the detailed schema for the LLM.
            schema_info = action.param_model.model_json_schema()
            # Simplify schema representation for the prompt if too verbose
            param_details = []
            props = schema_info.get("properties", {})
            required = schema_info.get("required", [])
            for name, details in props.items():
                req_star = "*" if name in required else ""
                param_details.append(
                    f'`{name}`{req_star} ({details.get("type", "any")})'
                )

            lines.append(
                f"- **`{action.name}`**: {action.description}"
            )  # Using action name from controller
            if param_details:
                lines.append(f'  - Parameters: {", ".join(param_details)}')

        # Add descriptions for agent/extract_content types manually if not in controller
        if "agent" not in [
            a.name for a in controller.registry.registry.actions.values()
        ]:
            lines.append("- **`agent`**: Executes a task using an autonomous agent.")
            lines.append(
                "  - Parameters: `task`* (string), `description` (string), `max_steps` (integer)"
            )
        # if "extract_content" not in [
        #     a.name for a in controller.registry.registry.actions.values()
        # ]:
        #     lines.append(
        #         "- **`extract_content`**: Uses an LLM to extract specific information from the current page."
        #     )
        #     lines.append(
        #         "  - Parameters: `goal`* (string), `description` (string), `should_strip_link_urls` (boolean)"
        #     )

        logger.debug(f"Generated actions markdown:\n{lines}")
        return "\n".join(lines)

    @staticmethod
    def _find_first_user_interaction_url(events: List[Dict[str, Any]]) -> Optional[str]:
        """Finds the URL of the first recorded user interaction."""
        return next(
            (
                evt.get("frameUrl")
                for evt in events
                if evt.get("type")
                in [
                    "input",
                    "click",
                    "scroll",
                    "select_change",
                    "key_press",
                ]  # Added more types
            ),
            None,
        )

    def _parse_llm_output_to_tool(
        self, llm_content: str
    ) -> ToolDefinitionSchema:
        """Attempts to parse the LLM string output into a ToolDefinitionSchema."""
        logger.debug(f"Raw LLM Output:\n{llm_content}")
        content_to_parse = llm_content

        # Heuristic cleanup: Extract JSON from markdown code blocks
        if "```json" in content_to_parse:
            match = re.search(
                r"```json\s*([\s\S]*?)\s*```", content_to_parse, re.DOTALL
            )
            if match:
                content_to_parse = match.group(1).strip()
                logger.debug("Extracted JSON from ```json block.")
        elif content_to_parse.strip().startswith(
            "{"
        ) and content_to_parse.strip().endswith("}"):
            # Assume it's already JSON if it looks like it
            content_to_parse = content_to_parse.strip()
            logger.debug("Assuming raw output is JSON.")
        else:
            logger.warning(
                "Could not reliably extract JSON from LLM output, attempting parse anyway."
            )

        try:
            # Try parsing directly first (might work with structured output)
            tool_data = ToolDefinitionSchema.model_validate_json(
                content_to_parse
            )
            logger.info("Successfully parsed LLM output into ToolDefinitionSchema.")
            return tool_data
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"Failed to parse LLM output into ToolDefinitionSchema: {e}"
            )
            logger.debug(f"Content attempted parsing:\n{content_to_parse}")
            raise ValueError(
                f"LLM output could not be parsed into a valid tool schema. Error: {e}"
            ) from e

    async def build_tool(
        self,
        input_tool: ToolDefinitionSchema,
        user_goal: str,
        use_screenshots: bool = False,
        max_images: int = 20,
    ) -> ToolDefinitionSchema:
        """
        Generates an enhanced tool definition from an input tool object using an LLM.

        Args:
            input_tool: The initial ToolDefinitionSchema object containing steps to process.
            user_goal: Optional high-level description of the tool's purpose.
                       If None, the user might be prompted interactively.
            use_screenshots: Whether to include screenshots as visual context for the LLM (if available in steps).
            max_images: Maximum number of screenshots to include (to manage cost/tokens).

        Returns:
            A new ToolDefinitionSchema object generated by the LLM.

        Raises:
            ValueError: If the input tool is invalid or the LLM output cannot be parsed.
            Exception: For other LLM or processing errors.
        """
        # Validate input slightly
        if not input_tool or not isinstance(input_tool.steps, list):
            raise ValueError("Invalid input_tool object provided.")

        # Handle user goal
        goal = user_goal
        if goal is None:
            try:
                goal = input(
                    "Please describe the high-level task for the tool (optional, press Enter to skip): "
                ).strip()
            except EOFError:  # Handle non-interactive environments
                goal = ""
        goal = goal or "Automate the recorded browser actions."  # Default goal if empty

        # Format the main instruction prompt
        prompt_str = self.prompt_template.format(
            actions=self.actions_markdown,
            goal=goal,
        )

        # Prepare the vision messages list
        vision_messages: List[Dict[str, Any]] = [{"type": "text", "text": prompt_str}]

        # Integrate message preparation logic here
        images_used = 0
        for step in input_tool.steps:
            step_messages: List[Dict[str, Any]] = []  # Messages for this specific step

            # 1. Text representation (JSON dump)
            step_dict = step.model_dump(mode="json", exclude_none=True)
            screenshot_data = step_dict.pop(
                "screenshot", None
            )  # Pop potential screenshot
            step_messages.append(
                {"type": "text", "text": json.dumps(step_dict, indent=2)}
            )

            # 2. Optional screenshot
            attach_image = use_screenshots and images_used < max_images
            step_type = getattr(step, "type", step_dict.get("type"))

            if attach_image and step_type != "input":  # Don't attach for inputs
                # Re-retrieve screenshot data if it wasn't popped (e.g., nested under 'data')
                # This assumes screenshot might still be in the original step model or dict
                # A bit redundant, ideally screenshot handling is consistent
                screenshot = (
                    screenshot_data
                    or getattr(step, "screenshot", None)
                    or step_dict.get("data", {}).get("screenshot")
                )

                if screenshot:
                    if isinstance(screenshot, str) and screenshot.startswith("data:"):
                        screenshot = screenshot.split(",", 1)[-1]

                    # Validate base64 payload
                    try:
                        base64.b64decode(cast(str, screenshot), validate=True)
                        meta = f"<Screenshot for event type '{step_type}'>"
                        step_messages.append({"type": "text", "text": meta})
                        step_messages.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot}"
                                },
                            }
                        )
                        images_used += (
                            1  # Increment image count *only* if successfully added
                        )
                    except (TypeError, ValueError, Exception) as e:
                        logger.warning(
                            f"Invalid or missing screenshot for step type '{step_type}' "
                            f'@ {step_dict.get("timestamp", "")}. Error: {e}'
                        )
                        # Don't add image messages if invalid

            # Add the messages for this step to the main list
            vision_messages.extend(step_messages)

        logger.info(
            f"Prepared {len(vision_messages)} total message parts, including {images_used} images."
        )

        # Invoke the LLM (structured output preferred)
        try:
            # Invoke the LLM (structured output preferred)
            # Need to handle cases where structured output isn't truly supported
            if hasattr(
                self.llm_structured, "output_schema"
            ):  # Check if it seems like structured output model
                llm_response = await self.llm_structured.ainvoke(
                    [HumanMessage(content=cast(Any, vision_messages))]
                )
                # If structured output worked, llm_response is the Pydantic object
                if isinstance(llm_response, ToolDefinitionSchema):
                    tool_data = llm_response
                else:
                    # It might have returned a message or dict, try parsing its content
                    content = getattr(llm_response, "content", str(llm_response))
                    tool_data = self._parse_llm_output_to_tool(str(content))
            else:
                # Fallback to basic LLM call and manual parsing
                llm_response = await self.llm_structured.ainvoke(
                    [HumanMessage(content=cast(Any, vision_messages))]
                )
                llm_content = str(
                    getattr(llm_response, "content", llm_response)
                )  # Get string content
                tool_data = self._parse_llm_output_to_tool(llm_content)

        except OutputParserException as ope:
            logger.error(f"LLM output parsing failed (OutputParserException): {ope}")
            # Try to parse the raw output as a fallback
            raw_output = getattr(ope, "llm_output", str(ope))
            logger.info("Attempting to parse raw output as fallback...")
            try:
                tool_data = self._parse_llm_output_to_tool(raw_output)
            except ValueError as ve_fallback:
                raise ValueError(
                    f"LLM structured output failed, and fallback parsing also failed. Error: {ve_fallback}"
                ) from ve_fallback
        except Exception as e:
            logger.exception(
                f"An error occurred during LLM invocation or processing: {e}"
            )
            raise  # Re-raise other unexpected errors

        # Return the tool data object directly
        return tool_data

    # path handlers
    async def build_tool_from_path(
        self, path: Path, user_goal: str
    ) -> ToolDefinitionSchema:
        """Build a tool from a JSON file path."""
        with open(path, "r") as f:
            tool_data = json.load(f)

        tool_data_schema = ToolDefinitionSchema.model_validate(tool_data)
        return await self.build_tool(tool_data_schema, user_goal)

    async def save_tool_to_path(
        self, tool: ToolDefinitionSchema, path: Path
    ):
        """Save a tool to a JSON file path."""
        with open(path, "w") as f:
            json.dump(tool.model_dump(mode="json"), f, indent=2)
