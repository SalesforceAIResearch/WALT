#!/usr/bin/env python3
"""
tool testing phase.
Tests generated tools for correctness.
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import sys

from walt.tools.discovery.utils import (
    setup_browser_environment,
    cleanup_browser_environment,
    evaluate_agent_result_with_tolerance,
    process_log_redirect,
)
from walt.browser_use.custom.utils import create_llm
import random


async def test_tool(
    args,
    tool_info: Dict[str, Any],
    tool_dirs: List[str],
    tool_type: str,
    process_id: str,
    max_steps: int = 20,
    include_validation: bool = False,
    attempt_number: int = 0,
    retry_reason: str = None,
) -> Dict[str, Any]:
    """Test a single tool in isolation."""
    # Import here to avoid circular imports
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from walt.browser_use import Controller
    from walt.browser_use.agent.service import Agent
    from walt.tools.discovery.register import _register_single_tool
    from pathlib import Path

    action_name = tool_info["action_name"]
    browser = None
    browser_context = None

    try:
        # Get auth file if provided
        storage_state = getattr(args, 'auth_file', None)
        
        # Setup browser
        browser, browser_context = await setup_browser_environment(
            base_url=args.base_url,
            storage_state=storage_state,
            headless=True,
        )

        # Generate test inputs for the tool based on its input schema
        tool_name = tool_info.get("action_name", "").replace("tool_", "")
        test_inputs = _generate_test_inputs(
            tool_info.get("file_path"), args.base_url, tool_name, attempt_number
        )

        task = _generate_single_tool_task(
            tool_info, test_inputs, include_validation=include_validation
        )

        # Create a regular agent with tool-aware system prompt
        llm = create_llm("openai", "gpt-5-mini")

        # Enhanced instructions for retry attempts
        retry_instructions = ""
        if retry_reason and "FAKE_VALIDATION" in retry_reason:
            retry_instructions = f"""
🚨 RETRY ATTEMPT - PREVIOUS FAILURE: {retry_reason}

CRITICAL: You FAILED the previous attempt because you claimed to execute the tool but never actually called it.

MANDATORY tool CALLING INSTRUCTIONS:
1. You MUST include the tool function call in your JSON action response
2. Use EXACTLY this format: {{"action": [{{"tool_function_name": {{parameters}}}}]}}
3. Do NOT just navigate and claim success - you must call the actual tool function
4. Do NOT write fake validation reports - the system verifies actual tool execution

EXAMPLE of CORRECT tool call:
{{"action": [{{"tool_search_listings": {{"sPattern": "bluetooth", "sCategory": "Electronics"}}}}]}}

EXAMPLE of WRONG behavior (what you did before):
- Navigate to page ✓
- Skip tool call ❌ 
- Write fake success report ❌

"""

        # System prompt that explicitly tells agent about registered tool functions
        tool_system_prompt = f"""
{retry_instructions}
tool ACTIONS AVAILABLE:
You have access to **tool actions** (names starting with "tool_") that implement recorded UI sequences reliably and deterministically.

tool USAGE:
- When instructed to use a tool action, you MUST use it in your structured JSON response.
- Include tool actions in your action array: {{"tool_name": {{parameters}}}}
- Do NOT perform manual UI actions when a tool action is available and requested
- tool actions handle all UI interactions internally

CONTEXT REQUIREMENTS:
- Context-preserving tools require you to navigate to the appropriate page first
- Absolute tools can be called from any starting point
- Always read the tool description carefully

Available navigation helpers:
- Search results: http://localhost:9980/index.php?page=search&sPattern=test
- Listing detail: http://localhost:9980/index.php?page=item&id=84144
- Homepage: http://localhost:9980/
"""

        # Create controller and register tool BEFORE creating agent
        controller = Controller()

        # Register the specific tool we want to test
        tool_file_path = tool_info.get("file_path")
        if tool_file_path and Path(tool_file_path).exists():
            _register_single_tool(
                controller=controller,
                tool_file=Path(tool_file_path),
                llm=llm,
                page_extraction_llm=llm,
                logger=print,
            )

        # Create generic agent
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            use_vision=True,
            extend_system_message=tool_system_prompt,
        )

        start_time = datetime.now()
        agent_result = await agent.run(max_steps=max_steps)
        end_time = datetime.now()

        # Agent.run() returns (AgentHistoryList, Page), we need just the AgentHistoryList
        if isinstance(agent_result, tuple):
            result, page = agent_result
        else:
            result = agent_result

        # Enhanced success detection using shared logic from utils
        # Extract expected tool name from the tool info for validation
        expected_tool_name = (
            tool_info.get("action_name") if tool_info else None
        )
        overall_success, evaluation_reason = evaluate_agent_result_with_tolerance(
            result, expected_tool_name
        )

        # Handle fake validation with retry and better instructions
        if not overall_success and attempt_number < 2:
            print(
                f"🔍 Fake validation detected for {expected_tool_name} (attempt {attempt_number + 1}), retrying with enhanced instructions..."
            )

            # Retry with enhanced instructions
            retry_result = await test_tool(
                args=args,
                tool_info=tool_info,
                tool_dirs=tool_dirs,
                tool_type=tool_type,
                process_id=process_id,
                max_steps=max_steps,
                include_validation=include_validation,
                attempt_number=attempt_number + 1,
                retry_reason=evaluation_reason,
            )

            # Return the retry result
            return retry_result

        # Always extract failure logs if the tool wasn't successful OR if there are errors
        failure_logs = ""
        if not overall_success or result.has_errors():
            failure_logs = _extract_failure_logs(result)

        return {
            "tool_name": action_name,
            "task_description": _generate_natural_language_template(tool_info),
            "success": overall_success,
            "steps": len(result.history),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "errors": result.errors(),
            "action_names": result.action_names(),
            "failure_logs": failure_logs,
        }

    except Exception as e:
        return {
            "tool_name": action_name,
            "task_description": _generate_natural_language_template(tool_info),
            "success": False,
            "steps": 0,
            "duration_seconds": 0,
            "errors": [str(e)],
            "action_names": [],
            "failure_logs": f"Exception during tool execution: {str(e)}",
        }

    finally:
        # Cleanup browser environment using shared utility
        await cleanup_browser_environment(browser, browser_context)


def create_skipped_test_result(args) -> Dict[str, Any]:
    """Create a test result for when testing is skipped."""
    return {
        "success": True,
        "errors": [],
        "failure_logs": "",
        "steps": 0,
        "duration": 0,
        "final_url": args.base_url,
        "action_names": [],
        "skipped": "Testing disabled via --no-test flag",
    }


def _generate_natural_language_template(tool_info: Dict[str, Any]) -> str:
    """Generate natural language description for tool."""
    description = tool_info.get("description", "No description available")
    return f"Execute tool: {description}"


def _generate_test_inputs(
    tool_file_path: str,
    domain: str = "classifieds",
    tool_name: str = None,
    attempt_number: int = 0,
) -> Dict[str, Any]:
    """Generate test inputs with simple priority: extracted > domain defaults > type defaults."""
    if not tool_file_path or not os.path.exists(tool_file_path):
        return {}

    try:
        # 1. Load extracted test inputs from tool generation (now fixed!)
        extracted_inputs = _load_extracted_test_inputs(tool_file_path)

        # 2. Load tool schema
        input_schema = _load_tool_schema(tool_file_path)

        # 3. Get domain-specific defaults
        domain_defaults = _get_domain_defaults(domain)

        # 4. Generate test inputs with simple priority
        test_inputs = {}
        extracted_count = 0

        for param in input_schema:
            param_name = param.get("name", "")
            param_enum = param.get("enum", None)

            # Handle enum parameters first
            if param_enum:
                # Prefer extracted value if it's in the enum
                if (
                    param_name in extracted_inputs
                    and extracted_inputs[param_name] in param_enum
                ):
                    test_inputs[param_name] = extracted_inputs[param_name]
                    extracted_count += 1
                    print(
                        f"  ✅ Using extracted enum value for {param_name}: {extracted_inputs[param_name]}"
                    )
                else:
                    test_inputs[param_name] = param_enum[0]
                    print(
                        f"  📋 Using first enum value for {param_name}: {param_enum[0]}"
                    )
                    continue

            # Priority 1: Extracted inputs (should work 90%+ of the time now!)
            if param_name in extracted_inputs:
                test_inputs[param_name] = extracted_inputs[param_name]
                extracted_count += 1
                print(
                    f"  ✅ Using extracted value for {param_name}: {extracted_inputs[param_name]}"
                )

            # Priority 2: Domain-specific defaults
            elif param_name in domain_defaults:
                test_inputs[param_name] = domain_defaults[param_name]
                print(
                    f"  🎯 Using domain default for {param_name}: {domain_defaults[param_name]}"
                )

            # Priority 3: Type-based defaults (only for required params)
            elif param.get("required", False):
                default_value = _get_type_default(param)
                if default_value is not None:
                    test_inputs[param_name] = default_value
                    print(f"  🔧 Using type default for {param_name}: {default_value}")
            else:
                print(f"  ⏭️  Skipping optional parameter {param_name}")

        # Summary
        total_count = len([p for p in input_schema if p.get("name") in test_inputs])
        if total_count > 0:
            print(
                f"  📊 Test input sources: {extracted_count}/{total_count} from extraction, {total_count - extracted_count} from defaults"
            )

        return test_inputs

    except Exception as e:
        print(f"Warning: Could not generate test inputs for {tool_file_path}: {e}")
        return {}


def _load_extracted_test_inputs(tool_file_path: str) -> Dict[str, Any]:
    """Load extracted test inputs from the .test_inputs.json file."""
    # Try versioned files first (v1, v2, v3, etc.), then fallback to unversioned
    base_path = tool_file_path.replace(".tool.json", "")
    test_inputs_file = None

    # Look for the highest version number first
    for version in range(10, 0, -1):  # Check v10 down to v1
        versioned_file = f"{base_path}.v{version}.test_inputs.json"
        if os.path.exists(versioned_file):
            test_inputs_file = versioned_file
            break

    # Fallback to unversioned file
    if not test_inputs_file:
        test_inputs_file = f"{base_path}.test_inputs.json"
        if not os.path.exists(test_inputs_file):
            return {}

    try:
        with open(test_inputs_file, "r") as f:
            extracted_data = json.load(f)

        if "test_inputs" in extracted_data:
            inputs = extracted_data["test_inputs"]

            # Handle double-nesting from old files (before the fix)
            if isinstance(inputs, dict) and "test_inputs" in inputs:
                inputs = inputs["test_inputs"]
                print(f"  🔧 Fixed double-nesting in extracted test inputs")

            print(f"  🎯 Found extracted test inputs: {inputs}")
            return inputs
        else:
            return {}

    except Exception as e:
        print(
            f"Warning: Could not load extracted test inputs from {test_inputs_file}: {e}"
        )
        return {}


def _get_type_default(param: Dict[str, Any]) -> Any:
    """Get a default value based on parameter type."""
    param_type = param.get("type", "string")

    if param_type == "string":
        return "test"
    elif param_type == "integer":
        return 1
    elif param_type == "boolean":
        return False
    elif param_type == "array":
        return []
    else:
        return "test"


def _load_tool_schema(tool_file_path: str) -> List[Dict[str, Any]]:
    """Load tool input schema from the tool JSON file."""
    try:
        with open(tool_file_path, "r") as f:
            tool_data = json.load(f)
        return tool_data.get("input_schema", [])
    except Exception as e:
        print(f"Warning: Could not load tool schema from {tool_file_path}: {e}")
        return []


def _get_domain_defaults(domain: str) -> Dict[str, Any]:
    """Get domain-specific default values for common parameters."""
    domain_defaults = {
        "reddit": {
            "base_url": "http://localhost:9999",
            "forum": "aww",
            "forum_slug": "aww",
            "sort": "hot",
            "sort_option": "hot",
            "layout": "card",
            "time_filter": "all",
        },
        "classifieds": {
            "base_url": "http://localhost:9980",
            "location": "san-francisco",
            "category": "all",
            "sort": "date",
            "price_min": 0,
            "price_max": 1000,
        },
        "shopping": {
            "base_url": "http://localhost:7770",
            "search_term": "laptop",
            "category": "electronics",
            "sort": "price",
            "min_price": 100,
            "max_price": 2000,
        },
    }

    return domain_defaults.get(
        domain, {"base_url": "http://localhost:8080"}  # Generic fallback
    )


def q(param: Dict[str, Any]) -> Any:
    """Get a sensible default value based on parameter type."""
    param_type = param.get("type", "string")
    param_name = param.get("name", "")

    # Special cases based on parameter name patterns
    if "url" in param_name.lower():
        return "http://localhost:8080"
    elif "id" in param_name.lower():
        return "test-id"
    elif "email" in param_name.lower():
        return "test@example.com"
    elif "phone" in param_name.lower():
        return "555-123-4567"
    elif "name" in param_name.lower():
        return "Test Name"

    # Type-based defaults
    if param_type == "string":
        return "test"
    elif param_type in ("number", "integer"):
        return 1
    elif param_type in ("boolean", "bool"):
        return True
    else:
        return None


def _load_tool_description(tool_file_path: str) -> str:
    """Load the actual tool description from the JSON file."""
    try:
        import json

        with open(tool_file_path, "r") as f:
            tool_data = json.load(f)
        return tool_data.get("description", "No description available")
    except Exception as e:
        print(
            f"Warning: Could not load tool description from {tool_file_path}: {e}"
        )
        return "No description available"


def _generate_single_tool_task(
    tool_info: Dict[str, Any],
    test_inputs: Dict[str, Any] = None,
    include_validation: bool = False,
) -> str:
    """Generate context-aware task description for calling the registered tool function.

    Args:
        tool_info: tool metadata including action_name and file_path
        test_inputs: Parameters to pass to the tool function
        include_validation: If True, includes detailed validation instructions
    """
    action_name = tool_info.get("action_name", "tool")

    # Load the actual tool description from the JSON file
    tool_file_path = tool_info.get("file_path")
    if tool_file_path:
        tool_description = _load_tool_description(tool_file_path)
    else:
        tool_description = tool_info.get(
            "description", "No description available"
        )

    # Build context-aware task with tool description
    context_task = f"""You need to execute the tool: "{action_name}"

tool DESCRIPTION: {tool_description}

IMPORTANT CONTEXT REQUIREMENTS:
- Read the tool description carefully to understand any context prerequisites
- If the description mentions "on the results page", "from a results page", "current results", etc., you must first navigate to a search results page
- If it mentions "from a listing detail page", "on listing detail page", etc., you must first navigate to a listing detail page  
- If it mentions "from any listing detail page provided by the user", you should navigate to a test listing page first
- For tools that start with specific contexts, establish that context BEFORE calling the tool function

EXECUTION STEPS:
1. Analyze the tool description to determine if you need to establish a specific context first
2. If context is needed, navigate to the appropriate page (e.g., search results, listing detail)
3. Once the proper context is established, call the tool function"""

    # Add parameters if provided
    if test_inputs:
        params_dict = {key: value for key, value in test_inputs.items()}
        params_json = json.dumps(params_dict, indent=2)
        context_task += f"""
4. IMPORTANT: Use the tool action in your structured response. Set the '{action_name}' field with these parameters:
{params_json}

Example response format:
{{"current_state": {{"evaluation_previous_goal": "...", "memory": "...", "next_goal": "..."}}, "action": [{{""{action_name}"": {params_json}}}]}}"""
    else:
        context_task += f"""
4. IMPORTANT: Use the tool action in your structured response. Set the '{action_name}' field with empty parameters:
{{}}

Example response format:
{{"current_state": {{"evaluation_previous_goal": "...", "memory": "...", "next_goal": "..."}}, "action": [{{""{action_name}"": {{}}}}]}}"""

    # Add validation if requested
    if include_validation:
        validation_task = f"""

After executing the tool, validate that it worked correctly:
1. Analyze the page state after the tool execution
2. Verify that the intended changes were applied based on the tool description: {tool_description}
{f"3. Check if the results match the expected parameters: {json.dumps(test_inputs)}" if test_inputs else "3. Verify the tool executed without errors"}
4. Report any issues, errors, or unexpected behavior you observe
5. Provide specific feedback on what worked well and what could be improved

Your response should include both the tool execution result AND a detailed validation report."""

        return context_task + validation_task
    else:
        return context_task


def _extract_failure_logs(agent_result) -> str:
    """Extract detailed failure information from agent result."""
    failure_info = []

    # Extract error messages if any
    errors = agent_result.errors()
    if errors and any(e for e in errors):
        error_summary = []
        for error in errors[:3]:  # Limit to first 3 errors
            if error:
                error_summary.append(str(error))
        if error_summary:
            failure_info.append("ERRORS:\n" + "\n".join(error_summary))

    # Extract the agent's detailed analysis from the final action
    if hasattr(agent_result, "history") and agent_result.history:
        last_action = agent_result.history[-1]

        # Check if the last action was a 'done' action with detailed feedback
        if hasattr(last_action, "result") and last_action.result:
            # last_action.result is a list of ActionResult objects
            if isinstance(last_action.result, list) and last_action.result:
                action_result = last_action.result[
                    0
                ]  # Get the first (usually only) result

                # Extract the detailed analysis text
                if (
                    hasattr(action_result, "extracted_content")
                    and action_result.extracted_content
                ):
                    analysis_text = action_result.extracted_content
                    if (
                        analysis_text and len(analysis_text.strip()) > 50
                    ):  # Only include substantial feedback
                        failure_info.append("AGENT ANALYSIS:\n" + analysis_text)

                # Also check for success flag
                if hasattr(action_result, "success") and not action_result.success:
                    failure_info.append(
                        "AGENT STATUS: Explicitly marked as unsuccessful"
                    )

    return "\n\n".join(failure_info) if failure_info else ""


async def test_existing_tools(args) -> int:
    """Test all existing tools in the output directory."""

    # Use the same discovery logic as registration to avoid versioned files
    from register import _discover_tool_files

    # Get finalized tools (optimized with fallback to base)
    finalized_files = _discover_tool_files(args.output_dir, "optimized")

    if not finalized_files:
        print("❌ No finalized tools found to test")
        return 0

    # Filter tools if specific ones are requested
    if args.tools:
        requested_tools = set(args.tools)
        filtered_files = []
        for tool_file in finalized_files:
            # Extract tool name from filename
            if ".optimized.json" in tool_file.name:
                tool_name = tool_file.name.replace(".optimized.json", "")
            else:
                tool_name = tool_file.name.replace(".tool.json", "")

            if tool_name in requested_tools:
                filtered_files.append(tool_file)

        finalized_files = filtered_files
        if not finalized_files:
            print(f"❌ None of the requested tools {args.tools} were found")
            return 0

        print(f"🎯 Testing only requested tools: {', '.join(args.tools)}")

    # Count optimized vs base fallback
    optimized_count = sum(1 for f in finalized_files if ".optimized.json" in f.name)
    base_fallback_count = len(finalized_files) - optimized_count

    print(f"🧪 Found {len(finalized_files)} finalized tools:")
    print(f"  - {optimized_count} optimized tools")
    print(f"  - {base_fallback_count} base fallback tools")

    successful_tests = 0

    # Test all finalized tools (optimized with base fallbacks)
    for tool_file in finalized_files:
        tool_dir = os.path.dirname(tool_file)

        # Determine tool name and type
        if ".optimized.json" in tool_file.name:
            tool_name = tool_file.name.replace(".optimized.json", "")
            tool_type = "optimized"
            type_label = "optimized"
        else:
            tool_name = tool_file.name.replace(".tool.json", "")
            tool_type = "base"
            type_label = "base (fallback)"

        tool_info = {
            "action_name": f"tool_{tool_name}",  # Use tool_ prefix to distinguish from regular actions
            "description": f"Test finalized tool: {tool_name}",
            "file_path": str(tool_file),
        }

        print(f"  🧪 Testing {type_label} tool: {tool_name}")
        print(f"    📁 File: {tool_file}")

        # Show test inputs being generated
        try:
            # Extract tool name from file path for test input generation
            tool_name = tool_file.stem  # Gets filename without extension
            test_inputs = _generate_test_inputs(
                str(tool_file),
                args.website,
                tool_name,
                0,  # Default to first option for batch testing
            )
            if test_inputs:
                inputs_str = ", ".join([f"{k}={v}" for k, v in test_inputs.items()])
                print(f"    🎯 Test inputs: {inputs_str}")
            else:
                print(f"    ⚠️  No test inputs generated")
        except Exception as e:
            print(f"    ❌ Failed to generate test inputs: {e}")

        test_result = await test_tool(
            args=args,
            tool_info=tool_info,
            tool_dirs=[tool_dir],
            tool_type=tool_type,
            process_id=f"test_{tool_name}",
            max_steps=20,
        )

        if test_result["success"]:
            duration = test_result.get("duration_seconds", 0)
            steps = test_result.get("steps", 0)
            print(
                f"  ✅ {tool_name} {type_label} tool passed ({steps} steps, {duration:.1f}s)"
            )
            successful_tests += 1
        else:
            errors = test_result.get("errors", [])
            error_summary = [str(e) for e in errors if e][:2]  # Show first 2 errors
            print(f"  ❌ {tool_name} {type_label} tool failed:")
            for error in error_summary:
                print(f"    💥 {error}")
            if len(errors) > 2:
                print(f"    ... and {len(errors) - 2} more errors")

        print()  # Add spacing between tests

    print(
        f"🧪 Testing complete: {successful_tests}/{len(finalized_files)} tools passed"
    )
    return successful_tests


def _sync_test_wrapper(args_tool_info_process_id):
    """Sync wrapper for ProcessPoolExecutor - unpacks arguments and runs async function."""
    args, tool_info, process_id = args_tool_info_process_id
    return asyncio.run(_process_single_test(args, tool_info, process_id))


async def _process_single_test(
    args, tool_info: Dict[str, Any], process_id: int
) -> bool:
    """Process a single tool test with logging."""
    tool_name = tool_info["action_name"].replace(
        "tool_", ""
    )  # Remove prefix for logging

    # Use process-specific logging (same as generate.py)
    log_dir = os.path.join(args.output_dir, "logs", "test_logs", "processes")
    os.makedirs(log_dir, exist_ok=True)
    with process_log_redirect(log_dir, process_id, tool_name):
        print(f"🔄 Process {process_id}: Testing {tool_name}")

        test_result = await test_tool(
            args=args,
            tool_info=tool_info,
            tool_dirs=[os.path.dirname(tool_info["file_path"])],
            tool_type=(
                "optimized"
                if ".optimized.json" in tool_info["file_path"]
                else "base"
            ),
            process_id=f"test_{tool_name}",
            max_steps=20,
        )

        success = test_result["success"]
        if success:
            duration = test_result.get("duration_seconds", 0)
            steps = test_result.get("steps", 0)
            print(f"  ✅ {tool_name} passed ({steps} steps, {duration:.1f}s)")
        else:
            errors = test_result.get("errors", [])
            error_summary = [str(e) for e in errors if e][:2]
            print(f"  ❌ {tool_name} failed:")
            for error in error_summary:
                print(f"    💥 {error}")

        return success


async def _run_parallel_tests(
    args, tool_infos: List[Dict[str, Any]], max_processes: int
) -> int:
    """Run tool tests in parallel using ProcessPoolExecutor."""
    successful_count = 0

    try:
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all tasks - pack args with each task
            future_to_task = {
                executor.submit(_sync_test_wrapper, (args, tool_info, i)): i
                for i, tool_info in enumerate(tool_infos)
            }

            completed_count = 0
            total_processes = len(tool_infos)
            print(f"Submitted {total_processes} test processes for execution")

            # Wait for completion
            for future in concurrent.futures.as_completed(future_to_task):
                task_idx = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        successful_count += 1
                    completed_count += 1
                    status = "✅" if result else "❌"
                    print(
                        f"Process {task_idx} completed {status} ({completed_count}/{total_processes})"
                    )
                except Exception as e:
                    print(f"Process {task_idx} failed: {repr(e)}")
                    completed_count += 1

    except Exception as e:
        print(f"Error in parallel execution: {repr(e)}")

    return successful_count


async def test_existing_tools_parallel(args, max_processes: int = 4) -> int:
    """Test all existing tools in parallel (used by command-line interface)."""

    # Use the same discovery logic as the sequential version
    from register import _discover_tool_files

    # Get finalized tools (optimized with fallback to base)
    finalized_files = _discover_tool_files(args.output_dir, "optimized")

    if not finalized_files:
        print("❌ No finalized tools found to test")
        return 0

    # Filter tools if specific ones are requested
    if args.tools:
        requested_tools = set(args.tools)
        filtered_files = []
        for tool_file in finalized_files:
            # Extract tool name from filename
            if ".optimized.json" in tool_file.name:
                tool_name = tool_file.name.replace(".optimized.json", "")
            else:
                tool_name = tool_file.name.replace(".tool.json", "")

            if tool_name in requested_tools:
                filtered_files.append(tool_file)

        finalized_files = filtered_files
        if not finalized_files:
            print(f"❌ None of the requested tools {args.tools} were found")
            return 0

        print(f"🎯 Testing only requested tools: {', '.join(args.tools)}")

    # Count optimized vs base fallback
    optimized_count = sum(1 for f in finalized_files if ".optimized.json" in f.name)
    base_fallback_count = len(finalized_files) - optimized_count

    print(f"🧪 Found {len(finalized_files)} finalized tools:")
    print(f"  - {optimized_count} optimized tools")
    print(f"  - {base_fallback_count} base fallback tools")
    print(f"🚀 Starting parallel testing with {max_processes} processes")

    # Prepare tool info for parallel execution
    tool_infos = []
    for tool_file in finalized_files:
        if ".optimized.json" in tool_file.name:
            tool_name = tool_file.name.replace(".optimized.json", "")
            type_label = "optimized"
        else:
            tool_name = tool_file.name.replace(".tool.json", "")
            type_label = "base (fallback)"

        tool_info = {
            "action_name": f"tool_{tool_name}",
            "description": f"Test finalized tool: {tool_name}",
            "file_path": str(tool_file),
            "type_label": type_label,
        }
        tool_infos.append(tool_info)

    # Execute in parallel or sequentially
    if max_processes == 1:
        # Sequential execution for debugging
        successful_count = 0
        for i, tool_info in enumerate(tool_infos):
            try:
                result = await _process_single_test(args, tool_info, i)
                if result:
                    successful_count += 1
            except Exception as e:
                print(f"tool {tool_info['action_name']} failed: {repr(e)}")
    else:
        # Parallel execution
        successful_count = await _run_parallel_tests(
            args, tool_infos, max_processes
        )

    print(
        f"🧪 Testing complete: {successful_count}/{len(finalized_files)} tools passed"
    )
    print(
        "📊 Check individual process logs in test_logs/processes/ for detailed execution logs"
    )

    return successful_count


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Test existing tools")
    parser.add_argument(
        "--website", default="classifieds", help="Website domain (default: classifieds)"
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for the website (auto-detected from domain if not provided)",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip authentication (default: use auth)",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=16,
        help="Maximum number of parallel processes (default: 16, use 1 for sequential). Overridden by --debug flag.",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        help="Test only specific tools (e.g., --tools search_listings create_listing). If not specified, tests all tools.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip resetting the domain environment before testing",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="The benchmark to test",
        default="vwa",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: disables multiprocessing and runs tests sequentially for easier debugging",
    )

    args = parser.parse_args()

    # Import here to avoid circular import
    from utils import run_domain_reset

    if not args.skip_reset:
        asyncio.run(run_domain_reset(args.website))

    args.output_dir = f"outputs/{args.website}-discovered"
    # Auto-detect base URL if not provided
    if not args.base_url:
        from utils import website2url_wa, website2url_vwa
        if args.benchmark == "wa":
            args.base_url = website2url_wa[args.website]
        else:
            args.base_url = website2url_vwa[args.website]

    # Set login flag (opposite of no_login)
    args.login = not args.no_login

    print(f"Testing tools in: {args.output_dir}")
    print(f"Website: {args.website}")
    print(f"Base URL: {args.base_url}")
    print(f"Authentication: {'enabled' if args.login else 'disabled'}")

    # Override max_processes if debug mode is enabled
    effective_max_processes = 1 if args.debug else args.max_processes
    debug_info = " (debug mode - sequential)" if args.debug else ""
    print(f"Max processes: {effective_max_processes}{debug_info}")
    print()

    # Run the async test function with parallel support
    result = asyncio.run(
        test_existing_tools_parallel(args, effective_max_processes)
    )

    # Exit with appropriate code
    exit(0 if result > 0 else 1)
