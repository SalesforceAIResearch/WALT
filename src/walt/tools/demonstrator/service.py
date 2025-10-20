import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Sequence, Union, Optional

import aiofiles
from walt.browser_use import Agent, AgentHistoryList, Browser
from walt.browser_use.agent.views import DOMHistoryElement
from walt.browser_use.browser.browser import BrowserConfig
from walt.browser_use.custom.eval_envs.VWA import (
    VWABrowser,
    VWABrowserContext,
    VWABrowserContextConfig,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from patchright.async_api import async_playwright

from walt.tools.generator.service import BuilderService

from walt.prompts.discovery import get_demonstration_prompt
from walt.tools.demonstrator.views import ParsedAgentStep, SimpleDomElement, SimpleResult
from walt.tools.schema.views import SelectorToolSteps, ToolDefinitionSchema


class DemonstratorService:
    def __init__(
        self,
        llm: BaseChatModel,
    ):
        self.llm = llm

        self.interacted_elements_hash_map: dict[str, DOMHistoryElement] = {}

    def _remove_none_fields_from_dict(self, d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    def _generate_stable_selector(self, dom_element: DOMHistoryElement) -> str:
        """
        Generate a stable CSS selector from DOM element attributes.
        Prioritizes stable attributes over positional selectors.

        Args:
            dom_element: The DOM element with attributes and tag information

        Returns:
            A stable CSS selector string
        """
        tag_name = dom_element.tag_name.lower()
        attributes = dom_element.attributes or {}

        # Priority 1: Unique ID (if it looks stable, not auto-generated)
        element_id = attributes.get("id", "").strip()
        if element_id and self._is_stable_id(element_id):
            return f"#{element_id}"

        # Priority 2: Name attribute (very stable for form elements)
        name_attr = attributes.get("name", "").strip()
        if name_attr:
            # For form elements, name is usually very stable
            if tag_name in ["input", "select", "textarea", "button"]:
                type_attr = attributes.get("type", "").strip()
                if type_attr:
                    return f'{tag_name}[name="{name_attr}"][type="{type_attr}"]'
                else:
                    return f'{tag_name}[name="{name_attr}"]'

        # Priority 3: Stable attribute combinations
        stable_selector = self._build_attribute_selector(tag_name, attributes)
        if stable_selector:
            return stable_selector

        # Priority 4: Class-based selector (if classes look stable)
        class_attr = attributes.get("class", "").strip()
        if class_attr:
            stable_classes = self._extract_stable_classes(class_attr)
            if stable_classes:
                class_selector = "." + ".".join(stable_classes)
                return f"{tag_name}{class_selector}"

        # Priority 5: Fallback to simplified positional selector
        # Use the original selector but try to simplify it
        original_selector = dom_element.css_selector or ""
        simplified = self._simplify_positional_selector(
            original_selector, tag_name, attributes
        )
        if simplified:
            return simplified

        # Last resort: return original selector or basic tag
        return original_selector or tag_name

    def _is_stable_id(self, element_id: str) -> bool:
        """Check if an ID looks stable (not auto-generated)."""
        # Skip IDs that look auto-generated
        unstable_patterns = [
            r"^[a-f0-9]{8,}$",  # Long hex strings
            r"^\d+$",  # Pure numbers
            r"^id\d+$",  # id123, id456
            r"^_\w+\d+$",  # _element123
            r"react-\w+",  # React generated IDs
            r"mui-\d+",  # Material-UI generated IDs
            r"^ui-id-\d+$",  # jQuery UI generated IDs like ui-id-123
            r"^[A-Z0-9]{6,}$",  # Random uppercase sequences like B3R4DD
            r"^\w*\d{3,}$",  # IDs ending with 3+ digits (likely generated)
            r"^gen-\w+",  # Generated IDs with gen- prefix
            r"^auto-\w+",  # Auto-generated IDs with auto- prefix
        ]

        for pattern in unstable_patterns:
            if re.match(pattern, element_id, re.IGNORECASE):
                return False

        return True

    def _is_stable_attribute_value(self, value: str) -> bool:
        """Check if an attribute value looks stable (not auto-generated)."""
        # Skip values that look auto-generated
        unstable_value_patterns = [
            r"^[a-f0-9]{8,}$",  # Long hex strings
            r"^\d+$",  # Pure numbers (likely IDs)
            r"^[A-Z0-9]{6,}$",  # Random uppercase sequences
            r"^ui-id-\d+$",  # jQuery UI generated values
            r"^\w*\d{4,}$",  # Values ending with 4+ digits
            r"^tmp-\w+",  # Temporary values
            r"^gen-\w+",  # Generated values
        ]

        for pattern in unstable_value_patterns:
            if re.match(pattern, value, re.IGNORECASE):
                return False

        return True

    def _build_attribute_selector(
        self, tag_name: str, attributes: Dict[str, str]
    ) -> Optional[str]:
        """Build selector using stable attributes."""
        # Priority attributes for different element types
        stable_attrs = []

        # For form elements
        if tag_name in ["input", "select", "textarea", "button"]:
            for attr in ["placeholder", "aria-label", "title", "role"]:
                value = attributes.get(attr, "").strip()
                if value:
                    stable_attrs.append(f'{attr}="{value}"')

        # For interactive elements
        if tag_name in ["button", "a", "div"]:
            for attr in ["aria-label", "role", "data-testid", "title"]:
                value = attributes.get(attr, "").strip()
                if value:
                    stable_attrs.append(f'{attr}="{value}"')

        # For rating/icon elements (i, span, etc.)
        if tag_name in ["i", "span"]:
            for attr in [
                "data-value",
                "data-rating",
                "aria-label",
                "title",
                "data-testid",
            ]:
                value = attributes.get(attr, "").strip()
                if value:
                    stable_attrs.append(f'{attr}="{value}"')

        # Generic data attributes for any element type (fallback)
        if not stable_attrs:
            for attr_name, value in attributes.items():
                if attr_name.startswith("data-") and attr_name in [
                    "data-value",
                    "data-rating",
                    "data-testid",
                    "data-qa",
                    "data-cy",
                    "data-id",
                    "data-role",
                    "data-action",
                ]:
                    value = value.strip()
                    # Skip dynamic-looking data attribute values
                    if value and self._is_stable_attribute_value(value):
                        stable_attrs.append(f'{attr_name}="{value}"')

        # Build selector with most stable attributes
        if stable_attrs:
            # Use up to 2 most specific attributes to avoid over-specification
            selected_attrs = stable_attrs[:2]
            attr_selector = "[" + "][".join(selected_attrs) + "]"
            return f"{tag_name}{attr_selector}"

        return None

    def _extract_stable_classes(self, class_attr: str) -> List[str]:
        """Extract classes that appear stable (not state-related)."""
        classes = class_attr.split()

        # Filter out likely unstable classes
        stable_classes = []
        unstable_patterns = [
            r"focus",
            r"hover",
            r"active",
            r"selected",
            r"checked",
            r"disabled",
            r"loading",
            r"error",
            r"success",
            r"^\d+$",  # Pure numbers
            r"^[a-f0-9]{6,}$",  # Hex codes
            r"css-\w+",  # CSS-in-JS generated classes
            r"^[A-Z0-9]{6,}$",  # Random uppercase sequences like B3R4DD
            r"ui-id-\d+",  # jQuery UI generated classes
            r"^x-\w+\d+",  # ExtJS style generated classes
            r"^\w*\d{3,}$",  # Classes ending with 3+ digits
            r"^gen-\w+",  # Generated classes with gen- prefix
            r"^auto-\w+",  # Auto-generated classes
            r"^tmp-\w+",  # Temporary classes
            r"^dyn-\w+",  # Dynamic classes
        ]

        for cls in classes:
            is_stable = True
            for pattern in unstable_patterns:
                if re.search(pattern, cls, re.IGNORECASE):
                    is_stable = False
                    break

            if is_stable and len(cls) > 1:  # Skip single character classes
                stable_classes.append(cls)

        # Return up to 2 most stable classes to avoid over-specification
        return stable_classes[:2]

    def _simplify_positional_selector(
        self, original_selector: str, tag_name: str, attributes: Dict[str, str]
    ) -> Optional[str]:
        """Simplify a complex positional selector by removing deep nesting."""
        if not original_selector:
            return None

        # Try to extract the meaningful part of the selector
        # Look for the last part that has the tag and attributes
        parts = original_selector.split(">")

        # Find the part with our target element
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i].strip()
            if tag_name in part:
                # Try to build a simpler selector from this part and maybe 1-2 parents
                simplified_parts = []

                # Add up to 2 parent levels for context
                start_idx = max(0, i - 2)
                for j in range(start_idx, len(parts)):
                    part_clean = parts[j].strip()
                    # Remove nth-of-type selectors that are too specific
                    part_clean = re.sub(r":nth-of-type\(\d+\)", "", part_clean)
                    part_clean = re.sub(r":nth-child\(\d+\)", "", part_clean)
                    if part_clean:
                        simplified_parts.append(part_clean)

                if simplified_parts:
                    return " > ".join(simplified_parts)

        return None

    def _history_to_tool_definition(
        self, history_list: AgentHistoryList
    ) -> list[HumanMessage]:
        # history

        messages: list[HumanMessage] = []

        for history in history_list.history:
            if history.model_output is None:
                continue

            interacted_elements: list[SimpleDomElement] = []
            for element in history.state.interacted_element:
                if element is None:
                    continue

                # hash element using stable selector instead of brittle positional selector
                try:
                    stable_selector = self._generate_stable_selector(element)
                    # Use stable selector for hash generation to ensure consistency
                    element_hash = hashlib.sha256(
                        f"{element.tag_name}_{stable_selector}".encode()
                    ).hexdigest()[:10]
                    logging.info(
                        f"Generated stable hash {element_hash} from selector: {stable_selector}"
                    )
                except Exception as e:
                    # Fallback to original method if stable selector generation fails
                    logging.warning(
                        f"Failed to generate stable selector for hash, using original method: {e}"
                    )
                    element_hash = hashlib.sha256(
                        f"{element.tag_name}_{element.css_selector}_{element.highlight_index}".encode()
                    ).hexdigest()[:10]
                    logging.info(
                        f"Generated fallback hash {element_hash} from: {element.css_selector}[{element.highlight_index}]"
                    )

                if element_hash not in self.interacted_elements_hash_map:
                    self.interacted_elements_hash_map[element_hash] = element

                interacted_elements.append(
                    SimpleDomElement(
                        tag_name=element.tag_name,
                        highlight_index=element.highlight_index,
                        shadow_root=element.shadow_root,
                        element_hash=element_hash,
                    )
                )

            screenshot = history.state.screenshot
            parsed_step = ParsedAgentStep(
                url=history.state.url,
                title=history.state.title,
                agent_brain=history.model_output.current_state,
                actions=[
                    self._remove_none_fields_from_dict(action.model_dump())
                    for action in history.model_output.action
                ],
                results=[
                    SimpleResult(
                        success=result.success or False,
                        extracted_content=result.extracted_content,
                    )
                    for result in history.result
                ],
                interacted_elements=interacted_elements,
            )

            parsed_step_json = json.dumps(parsed_step.model_dump(exclude_none=True))
            content_blocks: List[Union[str, Dict[str, Any]]] = []

            text_block: Dict[str, Any] = {"type": "text", "text": parsed_step_json}
            content_blocks.append(text_block)

            if screenshot:
                # Assuming screenshot is a base64 encoded string.
                # Adjust mime type if necessary (e.g., image/png)
                image_block: Dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
                }
                content_blocks.append(image_block)

            messages.append(HumanMessage(content=content_blocks))

        return messages

    def _populate_selector_fields(
        self, tool_definition: ToolDefinitionSchema
    ) -> ToolDefinitionSchema:
        """Populate cssSelector, xpath, and elementTag fields from interacted_elements_hash_map"""
        # Process each step to add back the selector fields
        for step in tool_definition.steps:
            if isinstance(step, SelectorToolSteps):
                if step.elementHash in self.interacted_elements_hash_map:
                    dom_element = self.interacted_elements_hash_map[step.elementHash]

                    # Generate a stable selector instead of using the original brittle one
                    try:
                        stable_selector = self._generate_stable_selector(dom_element)
                        step.cssSelector = stable_selector
                        logging.info(
                            f"Generated stable selector: {stable_selector} (original: {dom_element.css_selector})"
                        )
                    except Exception as e:
                        # Fallback to original selector if generation fails
                        logging.warning(
                            f"Failed to generate stable selector, using original: {e}"
                        )
                        step.cssSelector = dom_element.css_selector or ""

                    # Keep xpath and tag for potential fallback use
                    step.xpath = dom_element.xpath
                    step.elementTag = dom_element.tag_name

        # Create the full ToolDefinitionSchema with populated fields
        return tool_definition

    async def create_tool_definition(
        self, task: str, history_list: AgentHistoryList, **kwargs
    ) -> ToolDefinitionSchema:
        # Load workflow creation prompt from configs
        from walt.prompts.discovery import get_tool_builder_prompt
        prompt_content = get_tool_builder_prompt()

        demonstration_actions_markdown = BuilderService._get_available_actions_markdown()

        prompt_content = prompt_content.format(
            goal=task, actions=demonstration_actions_markdown
        )
        if kwargs.get("extend_system_message"):
            prompt_content = (
                f"""{prompt_content}\n\n{kwargs.get('extend_system_message')}"""
            )

        system_message = SystemMessage(content=prompt_content)
        human_messages = self._history_to_tool_definition(history_list)

        all_messages: Sequence[BaseMessage] = [system_message] + human_messages

        # Use structured output with the schema (original clean approach)
        structured_llm = self.llm.with_structured_output(
            ToolDefinitionSchema, method="function_calling"
        )

        print("🤖 LLM Call - Generating tool definition...")
        tool_definition: ToolDefinitionSchema = await structured_llm.ainvoke(all_messages)  # type: ignore
        print("💰 LLM Call Complete")

        tool_definition = self._populate_selector_fields(tool_definition)

        # Note: Optimization is now handled by the discover.py pipeline

        return tool_definition

    # Generate tool from prompt
    async def generate_tool_from_prompt(
        self,
        prompt: str,
        agent_llm: BaseChatModel,
        extraction_llm: BaseChatModel,
        headless: bool = True,
        storage_state: str = None,
        max_steps: int = 25,
    ) -> ToolDefinitionSchema:
        """
        Generate a tool definition from a prompt by:
        1. Running a browser agent to explore and complete the task
        2. Converting the agent history into a tool definition
        """

        browser_config = BrowserConfig(headless=headless)
        browser = VWABrowser(config=browser_config)

        # Build element hash map for later use
        self.interacted_elements_hash_map: Dict[str, DOMHistoryElement] = {}

        print("🔧 Creating browser context...")
        print(f"🔑 Storage state provided: {storage_state}")
        try:
            # Create browser context for the Agent (using VWA classes for auth support)
            if storage_state:
                print(f"✅ Using auth state from: {storage_state}")
                context_config = VWABrowserContextConfig(storage_state=storage_state)
            else:
                print("❌ No storage state - agent will start unauthenticated")
                context_config = VWABrowserContextConfig()

            # Create VWABrowserContext manually (like discovery phase)
            browser_context = VWABrowserContext(browser=browser, config=context_config)
            print("✅ Browser context created")

            agent = Agent(
                task=prompt,
                llm=agent_llm,
                browser_context=browser_context,
                use_vision=True,
                save_conversation_path=None,
                max_failures=3,
                override_system_message=get_demonstration_prompt(),
            )

            # Run the agent to get history (limit steps to prevent endless exploration)
            print("🤖 Running agent...")
            history = await agent.run(max_steps=max_steps)
            print(f"✅ Agent completed with {len(history.history)} steps")

            print(f"🔍 Agent execution completed with {len(history.history)} steps")

            # Create tool definition from the history
            print("🔨 Creating tool definition...")
            try:
                tool_definition = await self.create_tool_definition(
                    prompt, history
                )
                print("✅ tool definition created")

                # Extract test inputs using LLM
                print("🧪 Extracting test inputs from execution...")
                test_inputs = await self._extract_test_inputs_with_llm(
                    history, tool_definition, agent_llm
                )

                # Return both tool definition and test inputs
                return tool_definition, test_inputs
            except Exception as e:
                raise

        finally:
            # Clean up browser resources
            try:
                await browser.close()
            except Exception:
                pass  # Ignore cleanup errors

    async def regenerate_tool_with_feedback(
        self,
        prompt: str,
        previous_tool: ToolDefinitionSchema,
        failure_logs: str,
        attempt_number: int,
        agent_llm: BaseChatModel,
        extraction_llm: BaseChatModel,
        headless: bool = True,
        storage_state: str = None,
        max_steps: int = 25,
    ) -> ToolDefinitionSchema:
        """
        Regenerate a tool definition with feedback from previous failures.

        Args:
            prompt: Original task prompt
            previous_tool: The tool that failed testing
            failure_logs: Text summary of test failure logs
            attempt_number: Current attempt number (for context)
            agent_llm: LLM for browser agent
            extraction_llm: LLM for extraction
            headless: Browser headless mode
            storage_state: Authentication state file

        Returns:
            New tool definition incorporating feedback
        """
        # Generate specific selector fixes from error logs
        selector_fixes = self._generate_selector_fixes(failure_logs)

        # Enhance the original prompt with feedback context
        enhanced_prompt = f"""ORIGINAL TASK: {prompt}

PREVIOUS ATTEMPT ANALYSIS:
- This is attempt #{attempt_number} to create a working tool
- The previous tool failed during testing with these issues:

{failure_logs}

CRITICAL SELECTOR IMPROVEMENT RULES:
{selector_fixes}

IMPLEMENTATION REQUIREMENTS:
- Generate working selectors that resolve to exactly ONE element
- When logs show "strict mode violation: locator resolved to X elements", replace with specific selectors
- For ambiguous selectors, prefer attribute-based targeting over class-only
- Use elementHash as backup but prioritize semantic selectors
- Test mental model: "Will this selector find exactly one element on the target page?"

FEEDBACK INSTRUCTIONS:
- Analyze the failure logs to understand what went wrong
- Pay special attention to element selection, timing, and navigation issues
- Consider if agentic steps might be more appropriate for dynamic content
- Ensure selectors are more robust or use agent steps for unpredictable elements
- Double-check URL navigation and form interaction patterns
- MOST IMPORTANT: Actually implement the specific selector fixes listed above

Please create an improved tool that addresses these specific failures."""

        # Use existing generate_tool_from_prompt with enhanced prompt
        return await self.generate_tool_from_prompt(
            prompt=enhanced_prompt,
            agent_llm=agent_llm,
            extraction_llm=extraction_llm,
            headless=headless,
            storage_state=storage_state,
            max_steps=max_steps,
        )

    def _generate_selector_fixes(self, failure_logs: str) -> str:
        """
        Generate specific selector fix recommendations based on error patterns in logs.

        Args:
            failure_logs: Raw failure log content

        Returns:
            Formatted string with specific selector fixes
        """
        fixes = []

        # Check for strict mode violations (selector ambiguity)
        if "strict mode violation" in failure_logs and "locator" in failure_logs:
            if "a.tab" in failure_logs:
                fixes.append(
                    "- Replace 'a.tab' with 'a[href=\"/comments\"]' for Comments tab or 'a[href=\"/submissions\"]' for Submissions tab"
                )
                fixes.append(
                    "- Alternative: Use xpath with position: '//nav//a[contains(@class, \"tab\")][2]' for Comments"
                )

            if "button" in failure_logs and "resolved to" in failure_logs:
                fixes.append(
                    "- Replace generic 'button' with specific attributes: 'button[type=\"submit\"]', 'button.primary', or 'button[aria-label=\"...\"]'"
                )

            if "input" in failure_logs:
                fixes.append(
                    "- Replace generic 'input' with 'input[name=\"...\"]', 'input[type=\"...\"]', or 'input#id'"
                )

            # Generic ambiguity fix
            if not fixes:  # No specific pattern matched
                fixes.append(
                    "- SELECTOR AMBIGUITY DETECTED: Add specific attributes, use nth-child(), or convert to agent step"
                )

        # Check for element not found errors
        if "Failed to wait for element" in failure_logs or "Selector:" in failure_logs:
            # Extract the failing selector
            import re

            selector_match = re.search(r"Selector:\s*([^\s\n]+)", failure_logs)
            if selector_match:
                failing_selector = selector_match.group(1)
                fixes.append(
                    f"- Selector '{failing_selector}' not found - verify element exists or use more robust selector"
                )
                fixes.append(
                    f"- Consider using agent step: 'Click the [element description] button/link'"
                )

        # Check for navigation issues
        if "404" in failure_logs or "Not Found" in failure_logs:
            fixes.append(
                "- URL navigation failed - check for double slashes, verify path construction"
            )
            fixes.append(
                "- Use base_url without concatenation: navigate to {{base_url}} then click relative links"
            )

        # Check for timing issues
        if "timeout" in failure_logs.lower() or "wait" in failure_logs.lower():
            fixes.append(
                "- Element timing issue - add explicit wait or convert to agent step for dynamic content"
            )

        # Check for Pydantic validation errors (cssSelector null)
        if "cssSelector" in failure_logs and "Input should be a valid string" in failure_logs and "null" in failure_logs:
            fixes.append(
                "- tool GENERATION BUG: cssSelector is null but must be string"
            )
            fixes.append(
                "- Fix: Convert XPath-only steps to agent steps or set cssSelector to empty string"
            )
            fixes.append(
                "- Example: Change 'cssSelector': null to 'type': 'agent' with appropriate task description"
            )

        if not fixes:
            fixes.append(
                "- Review error patterns and ensure selectors target unique, stable elements"
            )
            fixes.append(
                "- Consider converting problematic deterministic steps to agent steps"
            )

        return "\n".join(fixes)

    # Note: tool optimization is now handled by the comprehensive
    # optimization framework in discover.py which provides URL operations,
    # parameter consolidation, and defensive programming removal.

    async def _extract_test_inputs_with_llm(
        self,
        history_list: AgentHistoryList,
        tool_definition: ToolDefinitionSchema,
        llm: BaseChatModel,
    ) -> Dict[str, Any]:
        """
        Use an LLM to extract test inputs from the agent execution history.

        The LLM analyzes what values the agent actually used during execution
        and generates appropriate test inputs for the tool's input schema.
        """
        try:
            # Build summary of agent execution
            execution_summary = self._build_execution_summary(history_list)

            # Get input schema for reference
            input_schema = tool_definition.input_schema or []
            schema_json = json.dumps(
                [param.model_dump() for param in input_schema], indent=2
            )

            prompt = f"""You are analyzing an agent's successful execution to extract test inputs.

AGENT EXECUTION SUMMARY:
{execution_summary}

tool INPUT SCHEMA:
{schema_json}

TASK: Generate test inputs based on the actual values the agent used during execution.

Look for:
- URLs the agent navigated to (extract query parameters, IDs, etc.)
- Text the agent entered into inputs
- Options the agent selected from dropdowns/menus
- Any other values that would be needed to replay this tool

Return a JSON object with test inputs that match the schema exactly:

{{
  "test_inputs": {{
    "parameter_name": "actual_value_used",
    ...
  }},
  "explanation": "Brief explanation of where these values came from"
}}

Focus on real, working values that would allow someone else to test this tool successfully.
"""
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])

            # Parse the response
            try:
                import re

                json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    test_inputs = result.get("test_inputs", {})
                    explanation = result.get("explanation", "No explanation provided")

                    print(f"✅ Extracted {len(test_inputs)} test input parameters")
                    print(f"📝 Explanation: {explanation}")

                    return {
                        "test_inputs": test_inputs,
                        "explanation": explanation,
                        "extraction_method": "llm_based",
                    }
                else:
                    print("⚠️  Could not parse LLM response for test inputs")
                    return {}
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse LLM response as JSON: {e}")
                return {}

        except Exception as e:
            print(f"⚠️  Error in LLM-based test input extraction: {e}")
            return {}

    def _build_execution_summary(self, history_list: AgentHistoryList) -> str:
        """Build a concise summary of agent execution for LLM analysis."""
        summary_parts = []

        for i, step in enumerate(history_list.history, 1):
            if not step.model_output:
                continue

            # Basic step info
            url = step.state.url if step.state else "Unknown"
            summary_parts.append(f"Step {i}: {url}")

            # Actions taken
            for action in step.model_output.action:
                action_dict = action.model_dump()
                action_type = action_dict.get("action_type", "unknown")

                if action_type == "go_to_url":
                    url = action_dict.get("url", "")
                    summary_parts.append(f"  - Navigated to: {url}")
                elif action_type == "input_text":
                    text = action_dict.get("text", "")
                    summary_parts.append(f"  - Entered text: '{text}'")
                elif action_type == "click_element":
                    element_text = action_dict.get("element_text", "")
                    if element_text:
                        summary_parts.append(f"  - Clicked: '{element_text}'")
                else:
                    summary_parts.append(f"  - Action: {action_type}")

        return "\n".join(summary_parts)
