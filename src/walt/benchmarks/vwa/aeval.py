import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prevent browser-use from setting up its own logging
# os.environ["BROWSER_USE_LOGGING_LEVEL"] = "disabled"

import sys
import asyncio
import json
from dotenv import load_dotenv
import base64
from pydantic import BaseModel, Field
from typing import Optional, Union, List
import io
from langchain_openai import ChatOpenAI
import PIL
import requests
import argparse
from pathlib import Path
import logging
import numpy as np
import subprocess
import tempfile
import os
import traceback
import time
import shutil


def classify_task_error(error_type: str, error_message: str) -> str:
    """
    Classify errors leveraging browser-use's existing error handling infrastructure.

    Maps the sophisticated multi-level error handling from:
    - Agent Service: ValidationError, RateLimitError, etc.
    - VWA Environment: TimeoutError from @atimeout/@aretry_timeout decorators
    - Playwright: Page crashes, network errors, etc.

    Returns:
        "environment": Server/network/browser infrastructure issues
        "model": LLM/model-related issues
        "rate_limit": API rate limiting
        "auth": Authentication/login failures
        "evaluation": Evaluation framework issues (StringEvaluator, etc.)
        "task": Genuine task logic failures
    """

    # Agent-level categorization (from walt.browser_use/agent/service.py _handle_step_error)
    if error_type in ["ValidationError", "ValueError"]:
        return "model"  # LLM/model output format issues
    if error_type == "RateLimitError":
        return "rate_limit"  # API rate limits

    # Authentication failures (from our auth guardrails)
    if error_type == "RuntimeError" and "Authentication failed" in error_message:
        return "auth"  # Authentication/login failures

    # Evaluation framework errors (leveraging existing detection logic)
    if error_type == "BeartypeCallHintParamViolation":
        return "evaluation"  # Evaluation framework issues
    if "StringEvaluator" in error_message:
        return "evaluation"  # Evaluation framework issues

    # VWA environment categorization (from @atimeout/@aretry_timeout decorators)
    if error_type == "TimeoutError":
        return "environment"  # From VWA timeout decorators

    # Playwright/browser errors (everything else in agent's "else" branch gets exponential backoff)
    browser_error_patterns = [
        "Page crashed",
        "Target crashed",
        "Protocol error",
        "Page.goto:",
        "Connection",
        "Gateway",
        "502",
        "503",
        "net::ERR_CONNECTION",
        "navigation timeout",
        "Connection reset",
        "Connection refused",
    ]
    if any(pattern in error_message for pattern in browser_error_patterns):
        return "environment"

    return "task"  # Genuine task logic failures


class LazyCaptioningFunction:
    """Lazy-loading wrapper for the captioning model.

    Only loads the actual captioning model when first called, saving startup time
    and memory for processes that don't encounter captioning tasks.

    ~5% of VWA tasks use captioning, so this avoids loading heavy BLIP2 model
    in 95% of cases.
    """

    def __init__(self, model_device, dtype, model_name, logger):
        self.model_device = model_device
        self.dtype = dtype
        self.model_name = model_name
        self.logger = logger
        self._captioning_fn = None
        self._loading = False

    def __call__(self, *args, **kwargs):
        if self._captioning_fn is None and not self._loading:
            self._loading = True
            self.logger.info(
                f"🔄 Lazy loading captioning model {self.model_name} on {self.model_device}"
            )
            start_time = time.time()

            from walt.browser_use.custom.evaluators.vwa import image_utils

            self._captioning_fn = image_utils.get_captioning_fn(
                self.model_device, self.dtype, self.model_name
            )

            load_time = time.time() - start_time
            self.logger.info(f"✅ Captioning model loaded in {load_time:.2f}s")
            self._loading = False

        if self._captioning_fn is None:
            # Still loading in another thread, wait briefly
            time.sleep(0.1)
            return self(*args, **kwargs)

        return self._captioning_fn(*args, **kwargs)


# Configure logging BEFORE importing browser_use to prevent it from hijacking our setup
def setup_comprehensive_logging():
    """Set up comprehensive logging that captures everything."""

    # Clear any existing handlers to start fresh
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create our custom formatter
    class UnifiedFormatter(logging.Formatter):
        def format(self, record):
            # Clean up browser_use logger names for better readability
            if isinstance(record.name, str) and record.name.startswith("browser_use."):
                # Keep more context but make it readable
                parts = record.name.split(".")
                if len(parts) > 2:
                    record.name = f"browser_use.{parts[-1]}"
                else:
                    record.name = "browser_use"
            return super().format(record)

    # Set root logger level to capture everything
    root_logger.setLevel(logging.DEBUG)

    return UnifiedFormatter


# Initialize logging before any other imports
UNIFIED_FORMATTER = setup_comprehensive_logging()

import torch

import glob
from typing import Union, Any
from walt.browser_use import Controller
from walt.browser_use.controller.views import NoParamsAction
from walt.browser_use.browser.browser import BrowserConfig
from walt.browser_use.custom.browser_context_zoo import BrowserContextBugFix
from typing import Any
from walt.browser_use.custom.eval_envs.VWA import (
    VWABrowser,
    VWABrowserContext,
    VWABrowserContextConfig,
    VWAEnv,
)
from walt.browser_use.custom.agent_zoo import VWA_Agent
from walt.browser_use.custom.evaluators.vwa.vwa_evaluators import evaluator_router, Action
from walt.browser_use.agent.views import ActionResult, AgentHistoryList
from walt.browser_use.custom.retriever.SimpleRetriever import SimpleRetriever
from walt.browser_use.custom.knowledge import QueryRephraser, NarrativeMemorySummarizer
from walt.browser_use.custom.trajectory_parser import (
    agent_trajectory_parser,
    create_simplified_trajectory,
)
from walt.browser_use.custom.utils import (
    create_llm,
    summarize_usage_info_from_jsonfied_trajectory,
)
import multiprocessing as mp

# tool-use availability is now handled by the shared tool_agent module
from walt.tools.discovery.register import register_tools_from_directory


load_dotenv()


# Override browser-use logging after imports
def finalize_logging_setup():
    """Override any logging setup that browser-use might have done."""

    # Check if browser-use logging should be disabled
    browser_use_log_level = os.getenv("BROWSER_USE_LOGGING_LEVEL", "info").lower()

    # Force all browser_use loggers to propagate to root
    for name in logging.Logger.manager.loggerDict:
        if isinstance(name, str) and name.startswith("browser_use"):
            logger = logging.getLogger(name)

            if browser_use_log_level == "disabled":
                # Completely silence browser-use logs
                logger.setLevel(
                    logging.CRITICAL + 1
                )  # Higher than CRITICAL to silence everything
                logger.propagate = False
                # Remove any handlers browser-use added
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
            else:
                logger.propagate = True  # Override browser-use's propagate=False
                # Remove any handlers browser-use added
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)

    # Also handle any third-party loggers we want to capture
    interesting_loggers = ["playwright", "openai", "langchain", "httpx", "httpcore"]

    for logger_name in interesting_loggers:
        if logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            if browser_use_log_level != "disabled":
                logger.propagate = True
                # Set to INFO level to capture important events but not spam
                logger.setLevel(logging.INFO)
            else:
                # Keep third-party loggers at ERROR level even when browser-use is disabled
                logger.setLevel(logging.ERROR)


finalize_logging_setup()


def setup_task_logger(
    task_id: str,
    args: argparse.Namespace,
    process_id: int = None,
    task_index: int = None,
) -> tuple[logging.Logger, list]:
    """
    Set up comprehensive logging for a specific task that captures everything.

    Returns:
        task_logger: The logger to use for this task
        handlers: List of handlers to clean up later
    """

    # Create log directory
    log_dir = Path(args.result_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create task-specific logger
    task_logger_name = f"task_{task_id}"
    task_logger = logging.getLogger(task_logger_name)
    task_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in task_logger.handlers[:]:
        task_logger.removeHandler(handler)

    # Create file handler for this specific task
    task_log_file = log_dir / f"{task_id}.log"
    file_handler = logging.FileHandler(task_log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        UNIFIED_FORMATTER("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    )

    # Instead of modifying the shared root logger, create a dedicated task logger
    # that captures logs from specific loggers we care about
    task_logger.addHandler(file_handler)

    # Set up specific loggers to also write to this task's file
    # This avoids conflicts with concurrent tasks
    important_loggers = [
        "browser_use",
        "tool_use",
        "browser_use.controller",
        "browser_use.controller.service",
        "tool_use.tool.service",
        "tool_use.controller.service",
        "__main__",
        f"task_{task_id}",
    ]

    additional_handlers = []
    for logger_name in important_loggers:
        logger_instance = logging.getLogger(logger_name)
        # Create a separate file handler for each logger to avoid sharing issues
        task_file_handler = logging.FileHandler(
            task_log_file, mode="a", encoding="utf-8"
        )
        task_file_handler.setLevel(logging.DEBUG)
        task_file_handler.setFormatter(
            UNIFIED_FORMATTER("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
        )
        logger_instance.addHandler(task_file_handler)
        additional_handlers.append((logger_instance, task_file_handler))

    # Set up console handler for this specific process (if we want console output)
    console_handler = None
    if process_id is not None:
        # Only show console output for the VERY FIRST task (index 0) in process 0
        # or if specifically requested via environment variable
        show_console = (process_id == 0 and task_index == 0) or os.getenv(
            "SHOW_ALL_CONSOLE_LOGS", "false"
        ).lower() == "true"

        if show_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(
                logging.INFO
            )  # Only show INFO and above on console
            console_handler.setFormatter(
                UNIFIED_FORMATTER(
                    f"[Process {process_id}:Task {task_id}] %(levelname)-8s [%(name)s] %(message)s"
                )
            )

            # Add console handler to task logger instead of root logger to avoid conflicts
            task_logger.addHandler(console_handler)

    # Ensure root logger captures everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Force browser-use loggers to propagate again (in case they reset)
    finalize_logging_setup()

    # Additional enforcement: if browser-use logging is disabled, make sure controller service logger is silenced
    browser_use_log_level = os.getenv("BROWSER_USE_LOGGING_LEVEL", "info").lower()
    if browser_use_log_level == "disabled":
        # Specifically silence the controller service which generates action logs
        controller_logger = logging.getLogger("browser_use.controller.service")
        controller_logger.setLevel(logging.CRITICAL + 1)
        controller_logger.propagate = False

    handlers_to_cleanup = [file_handler] + additional_handlers
    if console_handler:
        handlers_to_cleanup.append(console_handler)

    return task_logger, handlers_to_cleanup


def cleanup_task_logger(handlers_to_cleanup: list):
    """Clean up logging handlers after task completion."""
    root_logger = logging.getLogger()

    for item in handlers_to_cleanup:
        if isinstance(item, tuple):
            # Handle (logger_instance, handler) tuples
            logger_instance, handler = item
            if handler in logger_instance.handlers:
                logger_instance.removeHandler(handler)
            try:
                handler.close()
            except:
                pass  # Ignore cleanup errors
        else:
            # Handle plain handlers (console handlers, etc.)
            handler = item
            if handler in root_logger.handlers:
                root_logger.removeHandler(handler)
            try:
                handler.close()
            except:
                pass  # Ignore cleanup errors


class BrowserUseFormatter(logging.Formatter):
    def format(self, record):
        if type(record.name) == str and record.name.startswith("browser_use."):
            record.name = record.name.split(".")[-2]
        return super().format(record)


DATASET = os.environ["DATASET"]
logger = logging.getLogger(__name__)


def register_tools(
    controller, tool_dir, llm, task_logger, fallback_to_agent=False
):
    """Register tool actions from JSON files in directory using shared implementation."""

    return register_tools_from_directory(
        controller=controller,
        tool_dir=tool_dir,
        llm=llm,
        logger=task_logger,
        tool_type="optimized",  # Use optimized tools by default
        page_extraction_llm=llm,
        fallback_to_agent=fallback_to_agent,
    )


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    # Config file support - can replace all other arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML file (experiment_configs/*.yaml)",
    )
    
    # env setup
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--headless", action="store_false")
    parser.add_argument("--platform", type=str, default="Linux")

    # eval setup
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--test_config_base_dir", type=str)
    parser.add_argument("--max_processes", type=int, default=2)  # Reduced from 4
    parser.add_argument(
        "--max_tasks_per_proc",
        type=int,
        default=16,
        help="Maximum number of concurrent tasks per process",
    )
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)
    parser.add_argument("--force_login_every_task", action="store_true")
    parser.add_argument("--run_as_debug_mode", action="store_true")
    parser.add_argument(
        "--task_timeout",
        type=int,
        default=1800,  # 30 minutes per individual task
        help="Timeout in seconds for individual task execution (default: 900)",
    )
    parser.add_argument(
        "--task_retries",
        type=int,
        default=1,  # Retry failed tasks once
        help="Number of retries for failed tasks (default: 1)",
    )

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on cuda.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Use multiprocessing for true parallelism instead of asyncio concurrency",
    )

    # lm config
    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "claude", "google"]
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_planner", action="store_true")
    parser.add_argument(
        "--planner_model_wo_vision", type=str, default="o3-mini-2025-01-31"
    )
    parser.add_argument(
        "--planner_model_with_vision", type=str, default="gpt-4.1-2025-04-14"
    )
    parser.add_argument("--planner_interval", type=int, default=15)
    parser.add_argument("--planner_temperature", type=float, default=0.0)
    parser.add_argument("--local_kb_path", type=str, default="./kb")
    parser.add_argument("--split", type=str, required=False)
    # query rephraser related
    parser.add_argument(
        "--query_rephraser_model", type=str, default="gpt-5-mini-mini-2024-07-18"
    )
    # retriever related
    parser.add_argument("--retriever_model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--retriever_cache_dir", type=str, default="./cache")
    parser.add_argument("--retriever_top_k", type=int, default=1)
    parser.add_argument("--retriever_threshold", type=float, default=0.3)
    # narrative memory summarizer related
    parser.add_argument(
        "--narrative_memory_summarizer_model", type=str, default="gpt-4.1-2025-04-14"
    )
    # browser agent related
    parser.add_argument("--browser_agent_model", type=str, default="gpt-4.1-2025-04-14")

    # API related
    parser.add_argument(
        "--expose_api_actions",
        action="store_true",
        help="Expose API actions to the browser agent.",
    )
    parser.add_argument(
        "--expose_multimodal_actions",
        action="store_true",
        help="Expose multimodal actions to the browser agent.",
    )

    # tool related
    parser.add_argument(
        "--expose_tool_actions",
        action="store_true",
        help="Expose tool actions to the browser agent.",
    )
    parser.add_argument(
        "--tool_dir",
        type=str,
        default="tools",
        help="Directory containing tool JSON files.",
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="")

    parser.add_argument(
        "--task_start_delay",
        type=float,
        default=2.0,  # Increased from 0.5
        help="Delay in seconds between starting tasks within a process",
    )
    parser.add_argument(
        "--verify_with_judge",
        action="store_true",
        help="Verify the task with a judge",
    )
    parser.add_argument(
        "--fallback_to_agent",
        action="store_true",
        help="Fallback to agent if the tool fails",
    )

    args = parser.parse_args()
    
    # Apply config file overrides if provided
    if args.config:
        from walt.config import ExperimentConfig
        exp_config = ExperimentConfig.load(args.config)
        
        # Override with config values (CLI args take precedence if explicitly set)
        # LLM settings
        if not any('--llm' in arg or '--browser_agent_model' in arg for arg in sys.argv):
            args.browser_agent_model = exp_config.llm.agent_model
        if not any('--planner' in arg for arg in sys.argv):
            if exp_config.llm.planner_model:
                args.planner_model_with_vision = exp_config.llm.planner_model
                args.planner_model_wo_vision = exp_config.llm.planner_model
        if not any('--temperature' in arg for arg in sys.argv):
            args.temperature = exp_config.llm.temperature
        
        # Agent settings
        if not any('--max_steps' in arg for arg in sys.argv):
            args.max_steps = exp_config.agent.max_steps
        
        # Benchmark settings
        if exp_config.benchmark:
            if not any('--split' in arg for arg in sys.argv) and exp_config.benchmark.website:
                args.split = exp_config.benchmark.website
            if not any('--test_config_base_dir' in arg for arg in sys.argv) and exp_config.benchmark.task_list:
                # Extract base dir from task_list path
                import os
                args.test_config_base_dir = os.path.dirname(exp_config.benchmark.task_list)
        
        # Execution settings
        if not any('--max_processes' in arg for arg in sys.argv):
            args.max_processes = exp_config.execution.parallel
        if not any('--save_trace_enabled' in arg for arg in sys.argv):
            if exp_config.execution.save_traces:
                args.save_trace_enabled = True
        
        # Output settings
        if not any('--result_dir' in arg for arg in sys.argv):
            args.result_dir = exp_config.output.dir
        
        # Tool directory (custom field in config)
        if hasattr(exp_config, 'tool_dir') or 'tool_dir' in vars(exp_config):
            if not any('--tool_dir' in arg for arg in sys.argv):
                tool_dir = getattr(exp_config, 'tool_dir', None)
                if tool_dir:
                    args.tool_dir = tool_dir
                    args.expose_tool_actions = True

    return args


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb




async def evaluate_task_core(
    args: argparse.Namespace,
    config_file: str,
    browser_agent_llm: ChatOpenAI,
    planner_llm_wo_vision: Union[ChatOpenAI, None],
    planner_llm_with_vision: Union[ChatOpenAI, None],
    retriever: Union[SimpleRetriever, None],
    narrative_memory_summarizer_llm: Union[ChatOpenAI, None],
    eval_caption_image_fn=None,
    task_logger=None,
):
    """Core logic for evaluating a single task."""
    jsonfied_trajectory = []
    score = 0
    token_usage_data = {}
    browser = None
    context = None
    temp_dir = None  # Initialize for cleanup in finally block
    try:
        if task_logger is None:
            this_task_logger = logger
        else:
            this_task_logger = task_logger

        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]
            task_id = _c["task_id"]
            image_paths = _c.get("image", None)
            images = []

            # automatically login
            if _c["storage_state"] and args.force_login_every_task:
                cookie_file_name = os.path.basename(_c["storage_state"])
                # VWA uses on-demand authentication only (no auth pools)
                comb = get_site_comb_from_filepath(cookie_file_name)

                # Create process-isolated temp directory with PID to avoid conflicts
                process_id = os.getpid()
                temp_dir = tempfile.mkdtemp(prefix=f"auth_{process_id}_")

                this_task_logger.info(
                    f"Generating auth for task {task_id}, sites: {comb}"
                )
                this_task_logger.info(f"Temp auth dir: {temp_dir}")

                # Build command with account credentials if available
                cmd = [
                    "python",
                    "helpers/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                    "--process_id",
                    str(os.getpid()),  # Use current process ID
                    "--task_info",
                    str(
                        task_id
                    ),  # Use task_id as unique identifier for hash distribution
                ]

                try:
                    # subprocess to renew the cookie
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minute timeout
                        cwd=os.getcwd(),  # Ensure we're in the right directory
                    )

                    if result.returncode != 0:
                        this_task_logger.error(
                            f"Auth generation failed for task {task_id}"
                        )
                        this_task_logger.error(f"Command: {' '.join(cmd)}")
                        this_task_logger.error(f"Exit code: {result.returncode}")
                        this_task_logger.error(f"STDOUT: {result.stdout}")
                        this_task_logger.error(f"STDERR: {result.stderr}")
                        # Cleanup temp dir
                        try:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Authentication failed for task {task_id}: Auth generation subprocess failed with exit code {result.returncode}"
                        )

                except Exception as e:
                    this_task_logger.error(
                        f"Unexpected error during authentication: {repr(e)}"
                    )
                    # Cleanup temp dir
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Authentication failed for task {task_id}: {str(e)}"
                    )

                # STRONG AUTH GUARDRAILS: Use existing validation from auto_login.py
                expected_cookie_file = f"{temp_dir}/{cookie_file_name}"

                # Basic file existence check
                if not os.path.exists(expected_cookie_file):
                    this_task_logger.error(
                        f"🚨 AUTH GUARDRAIL VIOLATION: Cookie file not created"
                    )
                    this_task_logger.error(f"Expected: {expected_cookie_file}")
                    this_task_logger.error(
                        f"Files in temp dir: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'Directory does not exist'}"
                    )
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Authentication failed: Cookie file not generated for task {task_id}"
                    )

                # Use existing validation from auto_login.py
                from auto_login import (
                    is_expired_async,
                    SITES,
                    URLS,
                    KEYWORDS,
                    EXACT_MATCH,
                )
                from pathlib import Path

                validation_failed = False
                validation_errors = []

                for site in comb:
                    if site in SITES:
                        site_idx = SITES.index(site)
                        url = URLS[site_idx]
                        keyword = KEYWORDS[site_idx]
                        exact_match = EXACT_MATCH[site_idx]

                        # For Reddit, we need to use the actual username from auth metadata
                        if site == "reddit":
                            try:
                                with open(expected_cookie_file, "r") as f:
                                    auth_data = json.load(f)
                                actual_username = auth_data.get("metadata", {}).get(
                                    "account_email", "unknown"
                                )
                                if actual_username != "unknown":
                                    from auto_login import REDDIT

                                    url = f"{REDDIT}/user/{actual_username}/account"
                                    this_task_logger.info(
                                        f"🔍 Using Reddit validation URL for user: {actual_username}"
                                    )
                                else:
                                    this_task_logger.warning(
                                        f"⚠️ Could not determine Reddit username from auth metadata"
                                    )
                            except Exception as e:
                                this_task_logger.warning(
                                    f"⚠️ Could not read Reddit username from auth file: {e}"
                                )

                        try:
                            # Note: is_expired_async returns True if auth is INVALID/expired (opposite of is_expired)
                            if await is_expired_async(
                                Path(expected_cookie_file), url, keyword, exact_match
                            ):
                                validation_failed = True
                                validation_errors.append(
                                    f"Authentication validation failed for {site}"
                                )
                                this_task_logger.error(
                                    f"🚨 AUTH GUARDRAIL VIOLATION: {site} authentication invalid"
                                )
                            else:
                                this_task_logger.info(
                                    f"✅ AUTH VALIDATION PASSED: {site}"
                                )
                        except Exception as e:
                            validation_failed = True
                            validation_errors.append(
                                f"Validation error for {site}: {str(e)}"
                            )
                            this_task_logger.error(
                                f"🚨 AUTH GUARDRAIL VIOLATION: Error validating {site}: {str(e)}"
                            )
                    else:
                        this_task_logger.warning(
                            f"⚠️ Unknown site '{site}' - skipping validation"
                        )

                if validation_failed:
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Authentication failed for task {task_id}: {'; '.join(validation_errors)}"
                    )

                this_task_logger.info(
                    f"🔐 Authentication successful for task {task_id} - all sites validated"
                )
                _c["storage_state"] = expected_cookie_file

                # update the config file
                config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                with open(config_file, "w") as f:
                    json.dump(_c, f)

            # Load input images for the task, if any.
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                for image_path in image_paths:
                    # Load image either from the web or from a local path.
                    if image_path.startswith("http"):
                        input_image = PIL.Image.open(
                            requests.get(image_path, stream=True).raw
                        )
                    else:
                        input_image = PIL.Image.open(image_path)

                    images.append(input_image)
        if len(images) > 0:
            planner_llm = planner_llm_with_vision
        else:
            planner_llm = planner_llm_wo_vision

        use_vision_for_planner = True  # always use multimodal planner
        this_task_logger.info(f"[Config file]: {config_file}")
        this_task_logger.info(f"[Intent]: {intent}")
        this_task_logger.info(
            f"[Use vision for planner]: {use_vision_for_planner} because there are {len(images)} images"
        )
        if args.headless:
            is_headless = True
        else:
            is_headless = False
        browser_config = BrowserConfig(headless=is_headless)
        browser = VWABrowser(browser_config)
        if args.sleep_after_execution != 0:
            this_task_logger.warning("sleep_after_execution > 0 not implemented!")
        if args.save_trace_enabled:
            trace_path = os.path.join(args.result_dir, "traces", f"{task_id}")
        else:
            trace_path = None

        context_config = VWABrowserContextConfig(
            browser_window_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            trace_path=trace_path,
        )
        # Detect if this is a shopping task to use shopping-specific DOM service
        with open(config_file) as f:
            task_config = json.load(f)
        sites = task_config.get("sites", [])
        domain_type = "shopping" if sites[0] == "shopping" else None
        
        context = VWABrowserContext(
            browser=browser, config=context_config, som_color="black_transparent", domain_type=domain_type
        )

        env = VWAEnv(browser=browser, context=context)
        await env.areset(options={"config_file": config_file})

        # Simple narrative memory loading with sync validation (the essential fix)
        narrative_memory_path = os.path.join(args.local_kb_path, "narrative_memory.json")
        
        try:
            with open(narrative_memory_path, "r") as f:
                narrative_memory = json.load(f)
        except FileNotFoundError:
            this_task_logger.info(f"Narrative memory file not found, starting with empty memory.")
            narrative_memory = {}
        except Exception as e:
            this_task_logger.warning(f"Failed to read narrative memory: {e}. Starting with empty memory.")
            narrative_memory = {}
        
        task_pools = [*narrative_memory.keys()]
        
        # Thread-safe retriever cache management
        if task_pools:
            # Check if retriever cache is in sync with narrative memory
            retriever_cache_path = os.path.join(args.retriever_cache_dir, "documents.json")
            cache_in_sync = False
            
            try:
                if os.path.exists(retriever_cache_path):
                    with open(retriever_cache_path, "r") as f:
                        cached_docs = json.load(f)
                    # Check if cached documents match current narrative memory keys
                    if set(cached_docs) == set(task_pools):
                        cache_in_sync = True
                        this_task_logger.info(f"Retriever cache is in sync with narrative memory ({len(task_pools)} tasks)")
                    else:
                        this_task_logger.warning(f"Retriever cache out of sync: {len(cached_docs)} cached vs {len(task_pools)} in memory")
            except Exception as e:
                this_task_logger.warning(f"Failed to check cache sync: {e}")
            
            # Add documents to retriever (thread-safe with internal caching)
            try:
                retriever.add_documents(task_pools)
            except Exception as e:
                this_task_logger.error(f"Failed to add documents to retriever: {e}")
                # Continue without retriever functionality
                task_pools = []
        
        # retrieve the similar tasks
        result = []
        if task_pools:
            try:
                result = retriever.get_retrieved_docs(
                    intent, top_k=args.retriever_top_k, threshold=args.retriever_threshold
                )
            except Exception as e:
                if retriever.index is None:
                    result = []
                    this_task_logger.info(
                        f"Retriever index is None as the narrative memory is empty."
                    )
                else:
                    this_task_logger.error(f"Failed to retrieve similar tasks: {e}")
                    result = []
        
        # Build retrieved narrative memory with validation
        retrieved_narrative_memory = []
        for r in result:
            if r in narrative_memory:
                similar_task = r
                similar_experience = narrative_memory[r]
                retrieved_narrative_memory.append(
                    {"task": similar_task, "experience": similar_experience}
                )
            else:
                this_task_logger.warning(f"Retrieved task '{r}' not found in current narrative memory - cache may be stale")
                # This indicates cache/memory sync issue but we continue without this result
        if len(retrieved_narrative_memory) == 0:
            logger.info(
                f"No similar tasks found. You may want to adjust the retriever threshold."
            )

        # add custom actions
        controller = Controller(exclude_actions=["search_google"])

        @controller.action(
            "Call external planner agent to revise the current plan. This action should be called when the browser agent feels unable to make progress, repeating the same (set of) action(s), or cannot recover from the error.",
            param_model=NoParamsAction,
        )
        async def replan(
            param_model: NoParamsAction, browser: BrowserContextBugFix
        ) -> ActionResult:
            return ActionResult(
                extracted_content="Re-plan signal received", include_in_memory=True
            )

        extend_system_message = ""
        from auto_login import (
            ACCOUNTS,
            get_shopping_account_for_task,
            get_reddit_account_for_task,
        )

        # Only provide credentials for sites that actually need authentication
        if args.split in ACCOUNTS:
            if args.split == "shopping":
                # Use distributed account selection for shopping to prevent race conditions
                account = get_shopping_account_for_task(
                    process_id=os.getpid(), task_info=task_id
                )
                extend_system_message += f"In the event that you have been logged out and need to log in, use the following credentials: {{'username': '{account['email']}', 'password': '{account['password']}'}}\n"
            elif args.split == "reddit":
                # Use distributed account selection for Reddit to prevent rate limit conflicts
                account = get_reddit_account_for_task(
                    process_id=os.getpid(), task_info=task_id
                )
                extend_system_message += f"In the event that you have been logged out and need to log in, use the following credentials: {{'username': '{account['email']}', 'password': '{account['password']}'}}\n"
            else:
                # For other sites, use the original hardcoded accounts
                extend_system_message += f"In the event that you have been logged out and need to log in, use the following credentials: {ACCOUNTS[args.split]}\n"

        if args.expose_multimodal_actions:
            # Load multimodal skills from skills/multimodal directory
            from walt.browser_use.custom.skills import register_generic_skills

            register_generic_skills(controller)

        assert args.split in [
            "classifieds",
            "shopping",
            "reddit",
        ], "Invalid split specified"

        # Register tool actions if enabled
        if args.expose_tool_actions:
            # Resolve tool directory path (handle relative paths)
            tool_dir = Path(args.tool_dir) / Path(args.split + "-discovered")
            if not os.path.isabs(tool_dir):
                # Make relative paths relative to the script directory
                tool_dir = Path(__file__).parent / tool_dir

            tool_count = register_tools(
                controller,
                tool_dir,
                browser_agent_llm,
                this_task_logger,
                fallback_to_agent=args.fallback_to_agent,
            )
            # manually exclude for now
            # controller.registry.exclude_actions.append("delete_listing_tool")

            if tool_count > 0:
                extend_system_message += f"""
                In addition to UI actions, you have access to **tool actions** that implement recorded UI sequences deterministically.
                These are highly reliable and efficient compared to manual UI actions.
                - There are {tool_count} tool actions available, named based on their functionality (e.g., login_tool, search_for_listing_tool)
                - PREFER using tool actions when available as they are more robust than performing individual UI actions
                - If the plan generated by the planner lists browser actions that can be replaced by a tool, do so when possible. 
                - Use common sense to determine the right tool parameters even if they do not match exactly: example if a tool allows sorting by `Newly Listed' and the task asks for sorting by `published date', infer that they are likely one and the same.
                """
                this_task_logger.info(f"Registered {tool_count} tool actions")
            else:
                this_task_logger.warning("No tool actions were registered")

        extend_system_message += """
        **IMPORTANT SEARCH GUIDELINES**
        - Webpages may provide functionality search eg. to search, sort, filter, and paginate results. Use them when available, but be aware that the search engine may be rudimentary and struggle with:
            - compound queries (e.g red car): try simplifying the query (eg. try searching for just "car" if "red car" doesn't return any results)
            - queries about the appearance of the items: the search engine may only do retrieval based on text descriptions, and so may miss visual attributes unless they are explicitly mentioned in the text. ALWAYS use your visual understanding to complement the search engine.              
        - If no results are returned, do NOT conclude that no such items exist. Instead, try relaxing the search criteria. Pro-tip: applying filters with an empty search query is often a good fallback strategy                        
        - If a LOT of results are returned that cannot be directly filtered, note that they may span multiple pages. Make sure to check atleast a few pages before concluding that no such items exist.
        - Given a list of results, always carefully cross-reference text observations against the provided screenshot using element IDs
        """
        if args.expose_multimodal_actions:
            extend_system_message += """
            - extract_content() is a useful action to extract goal-relevant MULTIMODAL structured information from search results (both text as well as listing images where available)
            """
        else:
            extend_system_message += """
            - extract_content() is a useful action to extract goal-relevant TEXT-ONLY structured information from search results
            """
        if len(images) == 0:
            task_images = None
        else:
            task_images = images

        if args.verify_with_judge:
            from walt.browser_use.custom.skills import verify_with_judge
            from walt.browser_use.custom.skills.models import VerifyAction

            async def verify_with_judge_callback(
                task: str,
                task_images: Optional[List[PIL.Image.Image]],
                agent_history: AgentHistoryList,
                browser: VWABrowserContext,
            ) -> ActionResult:
                """
                Verify agent task completion using webjudge evaluation system. Can only be used after the agent issues a done action.
                """
                # Save task images to temporary files if provided
                task_image_paths = None
                if task_images:
                    import tempfile

                    temp_dir = tempfile.mkdtemp()
                    task_image_paths = []
                    for i, task_image in enumerate(task_images):
                        try:
                            task_image_path = os.path.join(
                                temp_dir, f"task_image_{i}.png"
                            )
                            task_image.save(task_image_path)
                            task_image_paths.append(task_image_path)
                        except Exception as e:
                            this_task_logger.warning(
                                f"Failed to save task image {i}: {e}"
                            )

                # Create VerifyAction params
                params = VerifyAction(
                    task=task,
                    task_image_paths=task_image_paths,
                    judge_model="gpt-5-mini",
                    score_threshold=5,
                )

                # Use centralized verify_with_judge function
                return await verify_with_judge(params, agent_history, browser)

        agent = VWA_Agent(
            task=intent,
            task_image=task_images,
            llm=browser_agent_llm,
            browser_context=context,
            controller=controller,
            generate_gif=False,
            extend_system_message=extend_system_message,
            planner_llm=planner_llm,
            planner_interval=args.planner_interval,  # large number to encourage auto-replan
            use_vision_for_planner=use_vision_for_planner,
            planner_inputs={"retrived_narrative_memory": retrieved_narrative_memory},
            expose_api_actions=args.expose_api_actions,
            expose_multimodal_actions=args.expose_multimodal_actions,
            expose_tool_actions=args.expose_tool_actions,
            register_done_callback=(
                verify_with_judge_callback if args.verify_with_judge else None
            ),
        )

        history, current_page = await agent.run(max_steps=args.max_steps)

        # extracted_content
        final_result = history.final_result()
        last_action = Action(answer=final_result)
        trajectory = [last_action]
        evaluator = evaluator_router(config_file, captioning_fn=eval_caption_image_fn)
        score = await evaluator(
            trajectory=trajectory, config_file=config_file, page=current_page
        )
        if int(score) == 1:
            this_task_logger.info(f"[Result] (PASS) {config_file}")
        else:
            this_task_logger.info(f"[Result] (FAIL) {config_file}")

        jsonfied_trajectory = agent.get_jsonfied_trajectory()
        trajectory_path = os.path.join(args.result_dir, "trajectory")
        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path, exist_ok=True)

        agent_generated_trajectory_path = os.path.join(
            trajectory_path, f"{task_id}.json"
        )
        with open(agent_generated_trajectory_path, "w") as f:
            traj_data = {
                "task_prompt": agent.task,
                "is_successful": True if int(score) == 1 else False,
                "steps": jsonfied_trajectory,
            }
            json.dump(traj_data, f, indent=4)

        # Save simplified trajectory as well
        try:

            simplified_trajectory = create_simplified_trajectory(traj_data)
            simplified_trajectory_dir = trajectory_path.replace(
                "trajectory", "trajectory_simplified"
            )
            if not os.path.exists(simplified_trajectory_dir):
                os.makedirs(simplified_trajectory_dir, exist_ok=True)
            simplified_trajectory_path = os.path.join(
                simplified_trajectory_dir, f"{task_id}.json"
            )

            with open(simplified_trajectory_path, "w") as f:
                json.dump(simplified_trajectory, f, indent=2)

            this_task_logger.info(
                f"Simplified trajectory saved to {simplified_trajectory_path}"
            )
        except Exception as e:
            this_task_logger.warning(f"Failed to create simplified trajectory: {e}")

        # Update narrative memory for both successes and failures, but not for errors
        # If we got here without an exception, the agent completed the task (success or failure)
        # Errors/crashes would have been caught in the except blocks above
        this_task_logger.info(f"Task completed with score {score} - updating narrative memory")
        
        # update the narrative memory
        narrative_memory_summarizer = NarrativeMemorySummarizer(
            llm=narrative_memory_summarizer_llm,
            platform=args.platform,
            local_kb_path=args.local_kb_path,
        )

        trajectory_description = agent_trajectory_parser(
            agent_generated_trajectory_path
        )

        trajectort_description_dir = os.path.join(
            args.result_dir, "trajectory_description"
        )
        if not os.path.exists(trajectort_description_dir):
            os.makedirs(trajectort_description_dir, exist_ok=True)
        # save the trajectory description
        with open(
            os.path.join(trajectort_description_dir, f"{task_id}.json"), "w"
        ) as f:
            trajectory_description_dict = {
                "task": intent,
                "trajectory": trajectory_description,
            }
            json.dump(trajectory_description_dict, f, indent=4)

        narrative_memory_summary, narrative_memory_summary_usage = (
            narrative_memory_summarizer.summarize_narrative_memory(
                trajectory_description
            )
        )
        # make sure write to the narrative memory is thread-safe
        narrative_memory_summarizer.save_narrative_memory(
            narrative_memory_summary, intent, force_overwrite=True
        )

        browser_agent_usage = summarize_usage_info_from_jsonfied_trajectory(
            jsonfied_trajectory
        )

        token_usage_data = {
            "browser_agent_usage": browser_agent_usage,
            "narrative_memory_summary_usage": narrative_memory_summary_usage,
        }

    finally:
        # Cleanup temp directories (auth files are automatically cleaned up with temp dirs)
        pass

        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                if task_logger:
                    task_logger.info(f"Cleaned up temp directory: {temp_dir}")
                else:
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                if task_logger:
                    task_logger.warning(
                        f"Failed to clean up temp directory {temp_dir}: {e}"
                    )
                else:
                    logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

        if browser is not None:
            try:
                # Add timeout protection for browser cleanup
                await asyncio.wait_for(browser.close(), timeout=30.0)
                if task_logger:
                    task_logger.debug("Browser closed successfully")
            except asyncio.TimeoutError:
                if task_logger:
                    task_logger.warning("Browser close timed out after 30 seconds")
                else:
                    logger.warning("Browser close timed out after 30 seconds")
            except Exception as e:
                if task_logger:
                    task_logger.warning(f"Failed to close browser: {e}")
                else:
                    logger.warning(f"Failed to close browser: {e}")

        if context is not None:
            try:
                # Add timeout protection for context cleanup
                await asyncio.wait_for(context.close(), timeout=30.0)
                if task_logger:
                    task_logger.debug("Context closed successfully")
            except asyncio.TimeoutError:
                if task_logger:
                    task_logger.warning("Context close timed out after 30 seconds")
                else:
                    logger.warning("Context close timed out after 30 seconds")
            except Exception as e:
                if task_logger:
                    task_logger.warning(f"Failed to close context: {e}")
                else:
                    logger.warning(f"Failed to close context: {e}")

        # Force garbage collection to help cleanup
        import gc

        gc.collect()
    return score, token_usage_data


async def single_task_worker(
    semaphore: asyncio.Semaphore,
    args,
    config_file: str,
    browser_agent_llm,
    planner_llm_wo_vision,
    planner_llm_with_vision,
    retriever,
    narrative_memory_summarizer_llm,
    eval_caption_image_fn,
    process_logger,
    task_index: int = None,
) -> tuple[float, dict]:
    """Worker function to execute a single task."""
    async with semaphore:
        task_id = os.path.basename(config_file).split(".")[0]
        try:
            # Add task-level timeout protection
            return await asyncio.wait_for(
                execute_single_task(
                    args,
                    config_file,
                    browser_agent_llm,
                    planner_llm_wo_vision,
                    planner_llm_with_vision,
                    retriever,
                    narrative_memory_summarizer_llm,
                    eval_caption_image_fn,
                    process_logger,
                    task_index,
                ),
                timeout=args.task_timeout,
            )
        except asyncio.TimeoutError:
            if process_logger:
                process_logger.error(
                    f"Task {task_id} timed out after {args.task_timeout} seconds"
                )

            # Create timeout error performance file to track the failure
            try:
                performance_path = os.path.join(args.result_dir, "performance")
                if not os.path.exists(performance_path):
                    os.makedirs(performance_path, exist_ok=True)

                performance_file = os.path.join(performance_path, f"{task_id}.json")

                # Classify timeout error
                timeout_error_classification = classify_task_error(
                    "TimeoutError", f"Task timed out after {args.task_timeout} seconds"
                )

                timeout_performance = {
                    "config_file": config_file,
                    "usage": {},  # Empty usage to indicate timeout
                    "score": 0.0,
                    "time (min)": args.task_timeout / 60.0,
                    "error": f"Task timed out after {args.task_timeout} seconds",
                    "error_type": "TimeoutError",
                    "error_classification": timeout_error_classification,
                }

                temp_file = performance_file + f".tmp.{os.getpid()}"
                with open(temp_file, "w") as f:
                    json.dump(timeout_performance, f, indent=4)
                os.rename(temp_file, performance_file)

                if process_logger:
                    process_logger.info(
                        f"Created timeout performance file for task {task_id}"
                    )
            except Exception as perf_error:
                if process_logger:
                    process_logger.error(
                        f"Failed to create timeout performance file for {task_id}: {perf_error}"
                    )

            return 0.0, {}
        except Exception as e:
            if process_logger:
                process_logger.error(
                    f"Unexpected error in task worker for {task_id}: {repr(e)}"
                )
            return 0.0, {}


async def execute_single_task(
    args,
    config_file: str,
    browser_agent_llm,
    planner_llm_wo_vision,
    planner_llm_with_vision,
    retriever,
    narrative_memory_summarizer_llm,
    eval_caption_image_fn,
    process_logger,
    task_index: int = None,
) -> tuple[float, dict]:
    """Execute a single evaluation task."""
    task_id = os.path.basename(config_file).split(".")[0]
    if process_logger:
        process_logger.info(f"Starting evaluation for {task_id}")

    # Get process_id from process_logger if available, otherwise use PID
    process_id = getattr(process_logger, "process_id", None) or os.getpid()

    task_logger, handlers_to_cleanup = setup_task_logger(
        task_id, args, process_id, task_index
    )
    try:
        start_time = time.time()
        task_logger.info(f"Starting evaluation for {config_file}")

        task_score, token_usage_data = await evaluate_task_core(
            args,
            config_file,
            browser_agent_llm,
            planner_llm_wo_vision,
            planner_llm_with_vision,
            retriever,
            narrative_memory_summarizer_llm,
            eval_caption_image_fn,
            task_logger,
        )

        time_spent = time.time() - start_time
        task_logger.info(f"Task {config_file} took {time_spent} seconds")

        task_logger.info(f"usage: {token_usage_data}")
        performance = {
            "config_file": config_file,
            "usage": token_usage_data,
            "score": task_score,
            "time (min)": time_spent / 60,
        }

        # Atomic write to prevent corruption
        performance_path = os.path.join(args.result_dir, "performance")
        if not os.path.exists(performance_path):
            os.makedirs(performance_path, exist_ok=True)

        performance_file = os.path.join(performance_path, f"{task_id}.json")
        temp_file = performance_file + f".tmp.{os.getpid()}"

        with open(temp_file, "w") as f:
            json.dump(performance, f, indent=4)
        os.rename(temp_file, performance_file)

        return task_score, token_usage_data

    except Exception as e:
        import datetime

        task_logger.error(f"Error in task {task_id}: {repr(e)}")
        error_dir = os.path.join(args.result_dir, "error")
        if not os.path.exists(error_dir):
            os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, f"{task_id}.txt")
        with open(error_file, "w") as f:
            f.write(f"[Process {os.getpid()}] [Config file]: {config_file}\n")
            f.write(f"[Timestamp]: {datetime.datetime.now().isoformat()}\n")
            f.write(f"[Error Type]: {type(e).__name__}\n")
            f.write(f"[Unhandled Error] {repr(e)}\n")

            # Determine error classification
            error_classification = classify_task_error(type(e).__name__, str(e))
            f.write(f"[Error Classification]: {error_classification}\n")

            f.write(f"[Full Traceback]:\n")
            f.write(traceback.format_exc())

            # Try to include task configuration for debugging
            try:
                with open(config_file, "r") as config_f:
                    config_content = config_f.read()
                f.write(f"\n[Task Configuration]:\n{config_content}\n")
            except Exception as config_error:
                f.write(
                    f"\n[Task Configuration]: Could not read config file: {config_error}\n"
                )

        # Still write a minimal performance file to indicate the task was attempted
        # This helps with tracking and prevents confusion in reporting
        try:
            performance_path = os.path.join(args.result_dir, "performance")
            if not os.path.exists(performance_path):
                os.makedirs(performance_path, exist_ok=True)

            performance_file = os.path.join(performance_path, f"{task_id}.json")
            error_performance = {
                "config_file": config_file,
                "usage": {},  # Empty usage to indicate error
                "score": 0.0,
                "time (min)": 0.0,
                "error": str(e),
                "error_type": type(e).__name__,
                "error_classification": error_classification,
            }

            temp_file = performance_file + f".tmp.{os.getpid()}"
            with open(temp_file, "w") as f:
                json.dump(error_performance, f, indent=4)
            os.rename(temp_file, performance_file)

            task_logger.info(f"Created error performance file for task {task_id}")
        except Exception as perf_error:
            task_logger.error(f"Failed to create error performance file: {perf_error}")

        return 0.0, {}
    finally:
        cleanup_task_logger(handlers_to_cleanup)




def single_process_worker_sync(*args, **kwargs):
    return asyncio.run(single_process_worker(*args, **kwargs))


async def single_process_worker(
    args: argparse.Namespace,
    config_files: list[str],
    process_id: int,
    process_logger: logging.Logger,
) -> list[tuple[float, dict]]:
    """Asyncio task runner inside a process."""
    try:
        # Add staggered startup to reduce server overload - optimized delays
        startup_delay = (
            process_id * 5
        )  # Restored to 5 seconds to prevent server overload
        if startup_delay > 0:
            process_logger.info(
                f"Process {process_id}: Waiting {startup_delay}s before starting to reduce server load"
            )
            await asyncio.sleep(startup_delay)

        # Import and create components within the process for isolation
        from walt.browser_use.custom.utils import create_llm
        from walt.browser_use.custom.evaluators.vwa import image_utils
        from walt.browser_use.custom.retriever.SimpleRetriever import SimpleRetriever

        process_logger.info(f"Process {process_id}: Initializing components")
        # convert args from dict to namespace
        args = argparse.Namespace(**args)
        # Create all components within this process for complete isolation
        retriever = SimpleRetriever(
            model_name=args.retriever_model_name, cache_dir=args.retriever_cache_dir
        )

        narrative_memory_summarizer_llm = create_llm(
            args.provider, args.narrative_memory_summarizer_model
        )

        if args.use_planner:
            planner_llm_wo_vision = create_llm(
                args.provider, args.planner_model_wo_vision, args.planner_temperature
            )
            planner_llm_with_vision = create_llm(
                args.provider, args.planner_model_with_vision, args.planner_temperature
            )
        else:
            planner_llm_wo_vision = None
            planner_llm_with_vision = None

        browser_agent_llm = create_llm(args.provider, args.browser_agent_model)

        # Create captioning function - each process gets its own instance
        if torch.cuda.is_available():
            devices = [
                "cuda:{}".format(el) for el in np.arange(torch.cuda.device_count())
            ]
            dtype = torch.float16
            model_device = devices[process_id % len(devices)]
        else:
            devices = ["cpu"]
            dtype = torch.float32
            model_device = "cpu"

        if args.run_as_debug_mode:
            eval_caption_image_fn = None
        else:
            # Use lazy loading for captioning model - only load when actually needed
            # This saves startup time and memory since only ~5% of tasks need captioning
            eval_caption_image_fn = LazyCaptioningFunction(
                model_device, dtype, args.eval_captioning_model, process_logger
            )

        process_logger.info(f"Process {process_id}: Using simple concurrent processing")
        # Reduce concurrent tasks per process to further reduce server load
        max_concurrent_tasks = max(
            1, min(args.max_tasks_per_proc, 2)
        )  # Cap at 2 concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Create tasks for concurrent execution within this process
        tasks = []
        for i, config_file in enumerate(config_files):
            # Add larger delay between task starts to reduce server load
            async def delayed_task(delay_seconds, task_index, *task_args):
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                return await single_task_worker(*task_args, task_index)

            task_delay = i * max(
                args.task_start_delay, 2.0
            )  # Minimum 2 seconds between tasks
            task = delayed_task(
                task_delay,
                i,  # Pass task index
                semaphore,
                args,
                config_file,
                browser_agent_llm,
                planner_llm_wo_vision,
                planner_llm_with_vision,
                retriever,
                narrative_memory_summarizer_llm,
                eval_caption_image_fn,
                process_logger,
            )
            tasks.append(task)

        # Run all tasks concurrently within this process
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in results
        results = []
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                process_logger.error(f"Task {config_files[i]} failed: {repr(result)}")
                results.append((0.0, {}))
            else:
                results.append(result)

        process_logger.info(
            f"Process {process_id}: Completed {len(results)} tasks using simple approach"
        )

        return results

    except Exception as e:
        process_logger.error(
            f"Process {process_id}: Error in single_process_worker: {repr(e)}"
        )
        return [(0.0, {}) for _ in config_files]
    finally:
        # Force cleanup of any remaining browser processes
        process_logger.info(f"Process {process_id}: Starting aggressive cleanup")

        try:
            # Force cleanup any remaining Playwright processes
            import subprocess
            import os
            import signal

            # Kill any Chrome processes that might be hanging
            try:
                # Get current process group to avoid killing parent processes
                current_pid = os.getpid()
                subprocess.run(
                    ["pkill", "-f", f"chrome.*remote-debugging.*{current_pid}"],
                    capture_output=True,
                    timeout=5,
                )
            except Exception as e:
                process_logger.debug(
                    f"Process {process_id}: Could not kill Chrome processes: {e}"
                )

            # Force garbage collection
            import gc

            gc.collect()

            process_logger.info(f"Process {process_id}: Aggressive cleanup completed")

        except Exception as cleanup_error:
            process_logger.warning(
                f"Process {process_id}: Cleanup failed: {cleanup_error}"
            )

        # Add a small delay to ensure cleanup completes
        await asyncio.sleep(1)




def evaluate_multiprocessing(
    args: argparse.Namespace, config_file_list: list[str]
) -> None:
    """Run evaluation using multiprocessing with asyncio within each process."""
    if len(config_file_list) == 0:
        logger.info("No tasks to evaluate.")
        return

    logger.info(
        f"Starting multiprocessing evaluation with {args.max_processes} processes"
    )
    logger.info(f"Each process will run {args.max_tasks_per_proc} tasks concurrently")
    logger.info(f"Total tasks to evaluate: {len(config_file_list)}")

    # Convert args to dict for serialization
    args_dict = vars(args)

    # Distribute tasks across processes
    tasks_per_proc = len(config_file_list) // args.max_processes
    remainder = len(config_file_list) % args.max_processes

    process_tasks = []
    start_idx = 0

    for i in range(args.max_processes):
        # Distribute remainder tasks among first few processes
        proc_task_count = tasks_per_proc + (1 if i < remainder else 0)
        end_idx = start_idx + proc_task_count
        process_logger = logging.getLogger(f"multiprocess.process_{i}")
        process_logger.setLevel(logging.INFO)

        # Add process_id attribute for use in task logging
        process_logger.process_id = i

        # Use our unified logging approach - only show console for process 0 by default
        if i == 0 or os.getenv("SHOW_ALL_CONSOLE_LOGS", "false").lower() == "true":
            process_logger.propagate = (
                True  # Let it flow to our unified console handler
            )
        else:
            process_logger.propagate = False  # Silence console for other processes

        # Don't add separate console handlers here - let our unified system handle it
        if start_idx < len(config_file_list):
            process_tasks.append(
                (args_dict, config_file_list[start_idx:end_idx], i, process_logger)
            )

        start_idx = end_idx

    # Use ProcessPoolExecutor for better timeout control
    try:
        from concurrent.futures import ProcessPoolExecutor
        import concurrent.futures

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(single_process_worker_sync, *task): i
                for i, task in enumerate(process_tasks)
            }

            results = [None] * len(process_tasks)
            completed_count = 0

            # Add progress tracking
            total_processes = len(process_tasks)
            logger.info(f"Submitted {total_processes} processes for execution")

            # Wait for results
            try:
                for future in concurrent.futures.as_completed(future_to_task):
                    task_idx = future_to_task[future]
                    try:
                        results[task_idx] = future.result()
                        completed_count += 1
                        logger.info(
                            f"Process {task_idx} completed successfully ({completed_count}/{total_processes})"
                        )
                    except Exception as e:
                        logger.error(f"Process {task_idx} failed: {repr(e)}")
                        results[task_idx] = [
                            (0.0, {}) for _ in process_tasks[task_idx][1]
                        ]
                        completed_count += 1

            except Exception as e:
                logger.error(f"Error in process execution: {repr(e)}")

                # Fill in empty results for unfinished processes
                for i, result in enumerate(results):
                    if result is None:
                        logger.error(f"Process {i} never completed - marking as failed")
                        results[i] = [(0.0, {}) for _ in process_tasks[i][1]]

        # Collect results by actually reading performance files to ensure accuracy
        # This avoids the critical bug where results are incorrectly mapped to task IDs
        total_score = 0
        successful_tasks = 0
        completed_tasks = 0
        failed_tasks = []
        error_counts = {
            "environment": 0,
            "model": 0,
            "rate_limit": 0,
            "auth": 0,
            "evaluation": 0,
            "task": 0,
            "missing": 0,
        }

        for config_file in config_file_list:
            task_id = os.path.basename(config_file).split(".")[0]
            performance_file = os.path.join(
                args.result_dir, "performance", f"{task_id}.json"
            )

            if os.path.exists(performance_file):
                try:
                    with open(performance_file, "r") as f:
                        perf_data = json.load(f)
                    score = perf_data.get("score", 0.0)
                    completed_tasks += 1

                    if score > 0:
                        successful_tasks += 1
                        total_score += score
                        logger.info(f"Task {task_id}: PASS (score: {score})")
                    else:
                        # Track error classification for failed tasks
                        error_classification = perf_data.get(
                            "error_classification", "task"
                        )
                        error_counts[error_classification] += 1
                        logger.info(
                            f"Task {task_id}: FAIL ({error_classification} error)"
                        )
                except Exception as e:
                    logger.error(
                        f"Error reading performance file for task {task_id}: {e}"
                    )
                    failed_tasks.append(task_id)
                    error_counts["task"] += 1
                    logger.info(f"Task {task_id}: FAIL (performance file corrupted)")
            else:
                failed_tasks.append(task_id)
                error_counts["missing"] += 1
                logger.info(f"Task {task_id}: FAIL (no performance file)")

        # Calculate adjusted success rate excluding environment errors
        environment_errors = error_counts["environment"]
        total_valid_tasks = len(config_file_list) - environment_errors
        adjusted_success_rate = (
            successful_tasks / total_valid_tasks * 100 if total_valid_tasks > 0 else 0
        )

        logger.info(f"Task completion summary:")
        logger.info(f"  Total tasks: {len(config_file_list)}")
        logger.info(f"  Completed: {completed_tasks}")
        logger.info(f"  Successful: {successful_tasks}")
        logger.info(f"  Failed/Missing: {len(failed_tasks)}")
        logger.info(f"  Error breakdown:")
        logger.info(
            f"    Environment errors: {error_counts['environment']} (server/network issues)"
        )
        logger.info(
            f"    Authentication errors: {error_counts['auth']} (login/auth failures)"
        )
        logger.info(
            f"    Model errors: {error_counts['model']} (LLM/validation issues)"
        )
        logger.info(f"    Rate limit errors: {error_counts['rate_limit']} (API limits)")
        logger.info(
            f"    Evaluation errors: {error_counts['evaluation']} (evaluation framework issues)"
        )
        logger.info(f"    Task logic errors: {error_counts['task']} (genuine failures)")
        logger.info(
            f"    Missing files: {error_counts['missing']} (no performance data)"
        )
        logger.info(
            f"  Raw success rate: {successful_tasks}/{len(config_file_list)} = {successful_tasks/len(config_file_list)*100:.1f}%"
        )
        logger.info(
            f"  Adjusted success rate (excluding environment errors): {successful_tasks}/{total_valid_tasks} = {adjusted_success_rate:.1f}%"
        )

        if failed_tasks:
            logger.warning(
                f"Tasks that failed or didn't complete: {', '.join(failed_tasks[:10])}"
            )
            if len(failed_tasks) > 10:
                logger.warning(f"... and {len(failed_tasks) - 10} more")

        if successful_tasks > 0:
            avg_score = total_score / len(config_file_list)
            logger.info(f"Average score: {avg_score:.3f}")
        else:
            logger.info("No tasks completed successfully")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error in evaluate_multiprocessing: {repr(e)}")
        raise


def prepare(args: argparse.Namespace) -> None:
    result_dir = args.result_dir
    if not result_dir:
        raise ValueError("result_dir is not provided")
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    """
    Determine which tasks need to be run or rerun.

    A task is considered finished if:
    1. It has a performance file with non-empty usage (successful or failed completion)

    Tasks should be rerun if:
    1. No performance file exists
    2. Performance file exists but no usage data (indicates errored/incomplete execution)
    """
    result_files = glob.glob(f"{result_dir}/performance/*.json")

    finished_files = []
    for f in result_files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
                task_id = os.path.basename(f).split(".")[0]

                # Check if we have evidence of actual execution
                error_file = os.path.join(result_dir, "error", f"{task_id}.txt")
                trajectory_file = os.path.join(
                    result_dir, "trajectory", f"{task_id}.json"
                )

                # Task is finished if:
                # 1. It has non-empty usage (successful completion or valid failure)
                # Only errored tasks (no usage data) should be retried
                has_completed_execution = data.get("usage", {}) != {}

                if has_completed_execution:
                    finished_files.append(f)
                else:
                    logger.debug(
                        f"Task {task_id} marked as unfinished - performance file exists but no completion evidence"
                    )

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            # If we can't read the performance file, consider it unfinished
            task_id = os.path.basename(f).split(".")[0] if f else "unknown"
            logger.debug(
                f"Task {task_id} marked as unfinished - corrupted performance file: {e}"
            )
            continue

    task_ids = [os.path.basename(f).split(".")[0] for f in finished_files]

    # Determine unfinished configs (same logic for debug and regular mode)
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)

    logger.info(
        f"Found {len(finished_files)} finished tasks, {len(unfinished_configs)} remaining"
    )
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "run_config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump run config to {config_file}")


def main():
    # Set multiprocessing start method for better compatibility
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, ignore
            pass

    args = config()
    prepare(args)

    # Set up main run logging with our unified system
    log_dir = Path(args.result_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main run log file for coordination across all processes
    main_log_file = log_dir / f"run_{args.test_start_idx}-{args.test_end_idx}.log"

    # Set up root logger with both file and console handlers
    root_logger = logging.getLogger()

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Main run file handler - captures everything
    main_file_handler = logging.FileHandler(main_log_file, mode="w", encoding="utf-8")
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(
        UNIFIED_FORMATTER("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    )
    root_logger.addHandler(main_file_handler)

    # Console handler for main process coordination (will be supplemented by task-specific console in process 0)
    main_console_handler = logging.StreamHandler(sys.stdout)
    main_console_handler.setLevel(logging.INFO)
    main_console_handler.setFormatter(
        UNIFIED_FORMATTER("%(levelname)-8s [%(name)s] %(message)s")
    )
    root_logger.addHandler(main_console_handler)

    # Set root logger level
    root_logger.setLevel(logging.DEBUG)

    # Re-finalize browser-use logging setup
    finalize_logging_setup()

    # Additional enforcement for browser-use action logging
    browser_use_log_level = os.getenv("BROWSER_USE_LOGGING_LEVEL", "info").lower()
    if browser_use_log_level == "disabled":
        # Make sure all browser-use loggers stay silenced
        for name in [
            "browser_use",
            "browser_use.controller",
            "browser_use.controller.service",
        ]:
            logger_instance = logging.getLogger(name)
            logger_instance.setLevel(logging.CRITICAL + 1)
            logger_instance.propagate = False

    test_config_base_dir = args.test_config_base_dir
    all_test_files = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        fname = os.path.join(test_config_base_dir, f"{i}.json")
        if os.path.exists(fname):
            all_test_files.append(fname)

    # Process tasks with optional retries for robustness
    for retry_attempt in range(args.task_retries + 1):
        test_file_list = get_unfinished(all_test_files, args.result_dir)

        if len(test_file_list) == 0:
            logger.info("All tasks completed successfully!")
            break

        if retry_attempt == 0:
            logger.info(f"Total {len(test_file_list)} tasks to process")
        else:
            logger.info(
                f"Retry attempt {retry_attempt}: {len(test_file_list)} tasks remaining"
            )

        # Log tool configuration
        if args.expose_tool_actions and retry_attempt == 0:
            logger.info(f"tool actions enabled")

        if retry_attempt == 0:
            dump_config(args)

        # If in debug mode, run only the first task in the main process
        if args.run_as_debug_mode:
            if not test_file_list:
                logger.warning("No tasks to run in debug mode.")
                break
            else:
                logger.info("Running in single-process debug mode...")
                # Create a logger for the single process run
                debug_process_logger = logging.getLogger("debug_process")
                debug_process_logger.setLevel(logging.INFO)
                debug_process_logger.process_id = 0  # Add process_id attribute
                debug_process_logger.propagate = (
                    True  # Let it use our main console handler
                )

                # Run the first task synchronously in the main process
                single_process_worker_sync(
                    vars(args), [test_file_list[0]], 0, debug_process_logger
                )
                break  # Debug mode only runs once
        else:
            # Choose between multiprocessing and asyncio
            evaluate_multiprocessing(args, test_file_list)

    # Final summary after all retries
    final_test_file_list = get_unfinished(all_test_files, args.result_dir)
    if len(final_test_file_list) == 0:
        logger.info("✅ All tasks completed successfully!")
    else:
        logger.warning(
            f"⚠️  {len(final_test_file_list)} tasks still incomplete after all retry attempts"
        )
        logger.warning(
            f"Incomplete tasks: {[os.path.basename(f).split('.')[0] for f in final_test_file_list[:10]]}"
        )
        if len(final_test_file_list) > 10:
            logger.warning(f"... and {len(final_test_file_list) - 10} more")

    # Clean up main logging handlers
    root_logger.removeHandler(main_file_handler)
    root_logger.removeHandler(main_console_handler)
    main_file_handler.close()
    main_console_handler.close()


if __name__ == "__main__":
    main()
