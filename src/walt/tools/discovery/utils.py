"""
Utility functions for tool discovery and testing.
Generic implementations - no benchmark-specific code.
"""

import os
import sys
import json
import shutil
from typing import Tuple, Optional, Dict, Any, List
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def setup_browser_environment(
    base_url: str,
    storage_state: Optional[str] = None,
    headless: bool = True,
) -> Tuple[Any, Any]:
    """
    Generic browser setup for any website.
    
    Args:
        base_url: Base URL of the website
        storage_state: Optional path to Playwright storage_state JSON for authentication
        headless: Whether to run browser in headless mode
    
    Returns:
        Tuple of (browser, browser_context)
    """
    from walt.browser_use.browser.browser import Browser, BrowserConfig
    from walt.browser_use.browser.context import BrowserContext, BrowserContextConfig
    
    browser = Browser(config=BrowserConfig(headless=headless))
    
    config_dict = {}
    if storage_state and os.path.exists(storage_state):
        config_dict["storage_state"] = storage_state
        print(f"🔑 Using authentication from {storage_state}")
    
    context = BrowserContext(browser=browser, config=BrowserContextConfig(**config_dict))
    
    # Initialize the browser session (this creates the Playwright context and initial page)
    await context.get_current_page()
    
    return browser, context


async def cleanup_browser_environment(browser, browser_context):
    """
    Clean up browser environment resources.

    Args:
        browser: Browser instance to close
        browser_context: Browser context to close
    """
    if browser_context:
        try:
            await browser_context.close()
        except Exception as e:
            print(f"Warning: Failed to close browser context: {e}")

    if browser:
        try:
            await browser.close()
        except Exception as e:
            print(f"Warning: Failed to close browser: {e}")


# ============================================================================
# Authentication Management  
# ============================================================================
# Note: Benchmark-specific authentication should be handled in benchmark scripts
# Generic discovery accepts a storage_state file path for authentication


# ============================================================================
# File Management
# ============================================================================


class FileManager:
    """Manages file operations for tool discovery and generation."""

    @staticmethod
    def clean_versioned_files(tool_dir: str, tool_name: str):
        """Clean up stale versioned files from previous runs."""
        import glob

        versioned_patterns = [
            f"{tool_name}.v*.tool.json",
            f"{tool_name}.v*.optimized.json",
        ]
        for pattern in versioned_patterns:
            for versioned_file in glob.glob(os.path.join(tool_dir, pattern)):
                try:
                    os.remove(versioned_file)
                    print(
                        f"  🧹 Removed stale versioned file: {os.path.basename(versioned_file)}"
                    )
                except Exception as e:
                    print(f"  ⚠️ Could not remove {versioned_file}: {e}")

    @staticmethod
    def load_existing_results(results_file: str) -> Optional[Dict[str, Any]]:
        """Load existing results if they exist."""
        if not os.path.exists(results_file):
            return None

        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️ Could not read existing results: {e}")
            return None

    @staticmethod
    def should_skip_tool(
        existing_results: Optional[Dict], force_regenerate: bool = False
    ) -> bool:
        """Check if tool should be skipped based on existing results."""
        if not existing_results or force_regenerate:
            return False

        final_status = existing_results.get("final_status", "unknown")
        return final_status in ["both_successful", "optimization_recovery", "base_only"]

    @staticmethod
    def save_tool_json(tool, file_path: str):
        """Save tool to JSON file, handling Pydantic models."""
        with open(file_path, "w") as f:
            if hasattr(tool, "model_dump"):
                json.dump(tool.model_dump(), f, indent=2)
            elif hasattr(tool, "dict"):
                json.dump(tool.dict(), f, indent=2)
            else:
                json.dump(tool, f, indent=2)

    @staticmethod
    def save_results(results: Dict[str, Any], results_file: str):
        """Save results to JSON file."""
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @staticmethod
    def cleanup_orphaned_directories(
        output_dir: str, current_tools: List[Dict[str, Any]]
    ):
        """Remove tool directories that are no longer in the current discovery list."""
        if not os.path.exists(output_dir):
            return

        # Get list of current tool names from discovery
        current_tool_names = {
            tool.get("name", "") for tool in current_tools
        }
        current_tool_names.discard("")  # Remove empty names

        # Get list of existing tool directories
        existing_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item != "." and item != "..":
                # Skip special files like exploration_result.json
                if not item.endswith(".json"):
                    existing_dirs.append(item)

        # Find orphaned directories
        orphaned_dirs = []
        for dir_name in existing_dirs:
            if dir_name not in current_tool_names:
                orphaned_dirs.append(dir_name)

        # Remove orphaned directories
        if orphaned_dirs:
            print(
                f"🧹 Found {len(orphaned_dirs)} orphaned tool directories to clean up:"
            )
            for dir_name in orphaned_dirs:
                dir_path = os.path.join(output_dir, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"  ✅ Removed: {dir_name}/")
                except Exception as e:
                    print(f"  ❌ Failed to remove {dir_name}/: {e}")
        else:
            print("🧹 No orphaned tool directories found")


# ============================================================================
# Agent Evaluation
# ============================================================================


def evaluate_agent_result_with_tolerance(
    result, expected_tool_name: str = None
) -> Tuple[bool, str]:
    """
    Enhanced agent evaluation that's tolerant of agent pessimism.

    This function implements consistent logic for evaluating whether an agent
    execution was successful, taking into account cases where agents mark
    themselves as failed despite producing meaningful results (e.g., search
    returning "no results").

    Args:
        result: AgentHistoryList object from agent execution
        expected_tool_name: Optional tool function name that should have been called (e.g., "tool_edit_listing")

    Returns:
        Tuple[bool, str]: (success, reason_description)
            - success: True if the agent execution should be considered successful
            - reason_description: Human-readable explanation of the evaluation
    """
    # Get agent's explicit self-evaluation (True/False/None)
    agent_explicit_success = result.is_successful()

    # Check objective criteria (completed and no errors)
    objective_success = result.is_done() and not result.has_errors()

    # CRITICAL: If testing a tool, verify the tool function was actually called
    # if expected_tool_name:
    #     tool_called = _verify_tool_called_in_history(result, expected_tool_name)
    #     if not tool_called:
    #         return False, f"FAKE_VALIDATION: Agent claimed success but never called {expected_tool_name}. The agent must use the tool action in its JSON response, e.g.: {{\"action\": [{{\"'{expected_tool_name}'\": {{...parameters...}}}}]}}"

    # Case 1: Agent completed objectively but explicitly marked itself as failed
    if objective_success and agent_explicit_success is False:
        # Check if the agent actually accomplished meaningful work
        has_meaningful_content = has_meaningful_agent_content(result)

        if has_meaningful_content:
            return True, "Agent marked itself as failed but produced meaningful results"
        else:
            return False, "Agent failed evaluation with no meaningful output"

    # Case 2: Agent didn't complete or had errors - this is a real failure
    elif not objective_success:
        return False, "Agent didn't complete or had errors"

    # Case 3: Success cases - agent says success or is neutral, and objective criteria pass
    else:
        return True, "Agent completed successfully"


def _verify_tool_called_in_history(result, expected_tool_name: str) -> bool:
    """
    Verify that the expected tool function was actually called in the agent's action history.

    Args:
        result: AgentHistoryList object from agent execution
        expected_tool_name: tool function name to look for (e.g., "tool_edit_listing")

    Returns:
        bool: True if the tool was called, False otherwise
    """
    try:
        # Iterate through each step in the agent's history
        for step in result:
            # Check if the step has model_output (the agent's response)
            if hasattr(step, "model_output") and step.model_output:
                # Check the action list in model_output
                if hasattr(step.model_output, "action") and step.model_output.action:
                    for action_item in step.model_output.action:
                        # Convert action to dict if it's an object
                        if hasattr(action_item, "__dict__"):
                            action_dict = action_item.__dict__
                        else:
                            action_dict = action_item

                        # Look for tool function name in the action
                        if (
                            isinstance(action_dict, dict)
                            and expected_tool_name in action_dict
                        ):
                            return True

                        # Also check string representation
                        if expected_tool_name in str(action_dict):
                            return True

            # Also check the legacy actions attribute
            if hasattr(step, "actions") and step.actions:
                for action in step.actions:
                    # Check if action contains the expected tool call
                    if (
                        hasattr(action, "extracted_content")
                        and action.extracted_content
                    ):
                        action_data = action.extracted_content
                        # Look for tool function name in the action data
                        if (
                            isinstance(action_data, dict)
                            and expected_tool_name in action_data
                        ):
                            return True
                        # Also check if it's in a JSON string representation
                        if (
                            isinstance(action_data, str)
                            and expected_tool_name in action_data
                        ):
                            return True

                    # Check the action's raw data structure
                    if hasattr(action, "__dict__"):
                        action_dict = action.__dict__
                        if expected_tool_name in str(action_dict):
                            return True

        return False
    except Exception as e:
        # If we can't parse the history, be conservative and assume tool wasn't called
        print(f"Warning: Failed to verify tool call in history: {e}")
        return False


def has_meaningful_agent_content(result) -> bool:
    """
    Check if an agent result contains meaningful extracted content.

    This is a helper function that checks for substantial content (>50 chars)
    in any of the agent's action results.

    Args:
        result: AgentHistoryList object from agent execution

    Returns:
        bool: True if meaningful content was found
    """
    return any(
        hasattr(h, "result")
        and h.result
        and any(
            r.extracted_content and len(r.extracted_content.strip()) > 50
            for r in h.result
            if hasattr(r, "extracted_content")
        )
        for h in result.history
    )


@contextmanager
def process_log_redirect(log_dir: str, process_id: int, tool_name: str):
    """Redirect all output (stdout, stderr, and logging) to process-specific log files."""
    import logging

    # Force unbuffered output
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.makedirs(log_dir, exist_ok=True)

    # Create process-specific log file
    log_file = os.path.join(log_dir, f"process_{tool_name}.log")

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Save original logging state
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    try:
        # Open log file and redirect (line buffered for real-time writing)
        with open(log_file, "w", encoding="utf-8", buffering=1) as f:
            # Use a simple class to tee output to both file and console
            class TeeOutput:
                def __init__(self, file_obj, console_obj):
                    self.file = file_obj
                    self.console = console_obj

                def write(self, text):
                    self.file.write(text)
                    self.file.flush()
                    # Force OS to write to disk immediately for real-time visibility
                    os.fsync(self.file.fileno())
                    self.console.write(text)
                    self.console.flush()

                def flush(self):
                    self.file.flush()
                    os.fsync(self.file.fileno())
                    self.console.flush()

            # Redirect both stdout and stderr to tee to both file and console
            tee_stdout = TeeOutput(f, original_stdout)
            tee_stderr = TeeOutput(f, original_stderr)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr

            # Clear existing logging handlers and add our custom handler
            root_logger.handlers.clear()

            # Also clear handlers from all existing loggers to ensure capture
            for name in logging.Logger.manager.loggerDict:
                logger = logging.getLogger(name)
                logger.handlers.clear()

            # Create a custom logging handler that writes to our tee output
            class TeeLogHandler(logging.StreamHandler):
                def __init__(self, tee_output):
                    super().__init__(tee_output)

                def emit(self, record):
                    try:
                        msg = self.format(record)
                        self.stream.write(msg + "\n")
                        self.stream.flush()
                    except Exception:
                        self.handleError(record)

            # Set up logging to use our tee handler
            handler = TeeLogHandler(tee_stdout)
            formatter = logging.Formatter("%(levelname)-8s [%(name)s] %(message)s")
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)  # Capture INFO and above (no DEBUG)

            # Also capture specific loggers that might be configured separately
            for logger_name in [
                "browser_use",
                "agent",
                "controller",
                "eval_envs",
                "telemetry",
            ]:
                specific_logger = logging.getLogger(logger_name)
                specific_logger.handlers.clear()
                specific_logger.addHandler(handler)
                specific_logger.setLevel(logging.INFO)  # INFO level for cleaner logs
                specific_logger.propagate = True  # Ensure it propagates to root

            yield log_file
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Restore original logging state
        root_logger.handlers.clear()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
