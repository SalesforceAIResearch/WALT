#!/usr/bin/env python3
"""
tool generation phase.
Handles the generation of tools from candidates.
"""

import asyncio
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from contextlib import contextmanager
from typing import Dict, Any, List
from datetime import datetime

from langchain_openai import ChatOpenAI
from walt.tools.demonstrator.service import DemonstratorService
from walt.browser_use.custom.utils import create_llm

from walt.tools.discovery.utils import FileManager, process_log_redirect
from walt.tools.discovery import test, optimize


def _is_auth_breaking_tool(tool_data: Dict[str, Any]) -> bool:
    """Check if tool should be skipped to preserve authentication (credential changes, admin access, or ownership issues)."""
    action = tool_data.get("action", "").lower()
    description = tool_data.get("description", "").lower()
    name = tool_data.get("name", "").lower()

    # Combine all text fields to search
    text = f"{action} {description} {name}"

    # Patterns that break authentication by modifying core account credentials
    auth_breaking_patterns = [
        # Account deletion
        "delete account",
        "close account",
        "deactivate account",
        "remove account",
        "terminate account",
        "cancel account",
        "delete user",
        "remove user",
        "deactivate user",
        # Credential changes that break login
        "change username",
        "change password",
        "change email",
        "update username",
        "update password",
        "update email",
        "modify username",
        "modify password",
        "modify email",
        # Admin-level tools that require admin authentication
        "admin",
        "administration",
        "moderate",
        "moderation",
        "manage user",
        "manage listing",
        "bulk edit",
        "bulk delete",
        "system settings",
        "site settings",
        "configure",
        "administration panel",
        "admin panel",
        "backend",
        "dashboard admin",
    ]

    return any(pattern in text for pattern in auth_breaking_patterns)


async def generate_tools(args, tools_json: List[Dict[str, Any]]) -> int:
    """Phase 2: Generate, test, and optimize tools from candidates."""
    if not tools_json:
        print("No tools to generate.")
        return 0

    # Initialize cost tracking
    print("💰 Starting tool generation with token tracking enabled")
    print(f"📊 Model: {args.llm} | Processing {len(tools_json)} candidates")

    # Filter out auth-breaking tools (account deletion, credential changes)
    original_count = len(tools_json)
    filtered_tools = [
        w for w in tools_json if not _is_auth_breaking_tool(w)
    ]
    filtered_count = len(filtered_tools)

    if filtered_count < original_count:
        skipped = original_count - filtered_count
        print(
            f"⚠️  Filtered out {skipped} auth-breaking tool(s) to preserve authentication"
        )
        for tool_data in tools_json:
            if _is_auth_breaking_tool(tool_data):
                name = tool_data.get("name", "unnamed")
                print(f"   - Skipped: {name}")

    if not filtered_tools:
        print("No tools remaining after filtering.")
        return 0

    # Clean up orphaned directories if force regenerate is enabled
    if getattr(args, "force_regenerate", False):
        FileManager.cleanup_orphaned_directories(args.output_dir, filtered_tools)

    max_processes = getattr(args, "max_processes", 16)
    print(f"🚀 Starting generation pipeline with {max_processes} processes")
    print(f"Total tools to process: {filtered_count}")

    # Prepare tasks for parallel execution
    process_tasks = []
    for i, tool_data in enumerate(filtered_tools):
        tool_name = tool_data.get("name", f"tool_{i}")
        tool_dir = os.path.join(args.output_dir, tool_name)
        os.makedirs(tool_dir, exist_ok=True)

        process_tasks.append((tool_data, tool_dir, i))

    # Execute in parallel or sequentially
    successful_count = 0

    if max_processes == 1:
        # Sequential execution for debugging
        for tool_data, tool_dir, process_id in process_tasks:
            try:
                result = await _process_single_tool(
                    args, tool_data, tool_dir, process_id
                )
                if result:
                    successful_count += 1
            except Exception as e:
                print(
                    f"Error processing tool {tool_data.get('name', process_id)}: {e}"
                )
    else:
        # Parallel execution
        successful_count = await _run_parallel_generation(
            args, process_tasks, max_processes
        )

    print(
        f"💰 tool generation complete! {successful_count}/{len(filtered_tools)} tools succeeded"
    )
    print("📊 Check individual process logs above for detailed token usage and costs")

    return successful_count


def _sync_worker_wrapper(args_tool_data_tool_dir_process_id):
    """Sync wrapper for ProcessPoolExecutor - unpacks arguments and runs async function."""
    args, tool_data, tool_dir, process_id = (
        args_tool_data_tool_dir_process_id
    )
    return asyncio.run(
        _process_single_tool(args, tool_data, tool_dir, process_id)
    )


async def _run_parallel_generation(
    args, process_tasks: List, max_processes: int
) -> int:
    """Run tool generation in parallel using ProcessPoolExecutor."""
    successful_count = 0

    try:
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all tasks - pack args with each task
            future_to_task = {
                executor.submit(_sync_worker_wrapper, (args, *task)): i
                for i, task in enumerate(process_tasks)
            }

            completed_count = 0
            total_processes = len(process_tasks)
            print(f"Submitted {total_processes} processes for execution")

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


async def _process_single_tool(
    args, tool_data: Dict[str, Any], tool_dir: str, process_id: int
) -> bool:
    """Process a single tool through the full generation pipeline."""
    tool_name = tool_data.get("name", f"tool_{process_id}")

    # Use process-specific logging
    log_dir = os.path.join(args.output_dir, "logs", "processes")
    with process_log_redirect(log_dir, process_id, tool_name):
        print(f"🔄 Process {process_id}: Starting {tool_name}")

        # Check for existing results and resume capability
        results_file = os.path.join(tool_dir, "results.json")
        existing_results = FileManager.load_existing_results(results_file)

        if FileManager.should_skip_tool(
            existing_results, getattr(args, "force_regenerate", False)
        ):
            final_status = existing_results.get("final_status", "unknown")
            print(f"  ✅ {tool_name} already completed with status: {final_status}")
            print(f"     Use --force-regenerate to overwrite")
            return True

        # Clean up old files - with force regenerate, remove entire directory contents
        if getattr(args, "force_regenerate", False):
            print(f"  🧹 Force regenerate: clearing {tool_name}/ directory")
            # Remove all files in the tool directory but keep the directory itself
            import glob

            for file_path in glob.glob(os.path.join(tool_dir, "*")):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"    ✅ Removed: {os.path.basename(file_path)}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(
                            f"    ✅ Removed directory: {os.path.basename(file_path)}/"
                        )
                except Exception as e:
                    print(f"    ⚠️ Could not remove {file_path}: {e}")
        else:
            # Clean versioned files (normal behavior)
            FileManager.clean_versioned_files(tool_dir, tool_name)

            # Also remove stale results.json for failed tools so they start fresh
            if existing_results and existing_results.get("final_status") == "failed":
                results_file_to_remove = os.path.join(tool_dir, "results.json")
                if os.path.exists(results_file_to_remove):
                    try:
                        os.remove(results_file_to_remove)
                        print(f"  🧹 Removed stale results.json for failed tool")
                    except Exception as e:
                        print(f"  ⚠️ Could not remove results.json: {e}")

        try:
            # Run the generation, testing, and optimization pipeline
            results = await _generate_test_optimize_tool(
                args, tool_data, tool_dir, process_id
            )

            # Save results
            FileManager.save_results(results, results_file)

            print(f"  📋 Results saved for {tool_name}: {results['final_status']}")
            return results["final_status"] != "failed"

        except Exception as e:
            print(f"Process {process_id} failed: {e}")
            return False


async def _generate_test_optimize_tool(
    args, tool_data: Dict[str, Any], tool_dir: str, process_id: int
) -> Dict[str, Any]:
    """Full generation, testing, and optimization cycle for a single tool."""
    tool_name = tool_data.get("name", f"tool_{process_id}")

    # Setup
    llm = create_llm("openai", args.llm)
    page_extraction_llm = create_llm("openai", "gpt-5-mini")
    demonstrator_service = DemonstratorService(llm=llm)

    results = {
        "tool_name": tool_name,
        "base_attempts": [],
        "optimization": None,
        "final_status": "failed",
        "process_id": process_id,
    }

    base_tool = None
    max_attempts = 5

    # Phase 1: Generate and test base tool (up to 5 attempts)
    for attempt in range(1, max_attempts + 1):
        print(f"  🎯 Attempt {attempt}/{max_attempts}: Generating base tool...")

        try:
            if attempt == 1:
                # First attempt: Generate from scratch
                base_tool, extracted_test_inputs = (
                    await _generate_tool_from_candidate(
                        demonstrator_service, tool_data, args, process_id
                    )
                )
            else:
                # Regenerate with feedback from previous failure
                previous_failure = results["base_attempts"][-1]["failure_logs"]
                prompt = _build_tool_prompt(tool_data, args)

                base_tool, extracted_test_inputs = (
                    await demonstrator_service.regenerate_tool_with_feedback(
                        prompt=prompt,
                        previous_tool=base_tool,
                        failure_logs=previous_failure,
                        attempt_number=attempt,
                        agent_llm=llm,
                        extraction_llm=page_extraction_llm,
                        headless=True,
                        storage_state=getattr(args, 'auth_file', None),
                        max_steps=30,  # Reduced to prevent overly verbose tools
                    )
                )

            # Save versioned tool
            version_file = os.path.join(
                tool_dir, f"{tool_name}.v{attempt}.tool.json"
            )
            FileManager.save_tool_json(base_tool, version_file)

            # Save test inputs if they were extracted during generation
            if "extracted_test_inputs" in locals() and extracted_test_inputs:
                test_inputs_file = os.path.join(
                    tool_dir, f"{tool_name}.v{attempt}.test_inputs.json"
                )
                with open(test_inputs_file, "w") as f:
                    import json

                    json.dump({"test_inputs": extracted_test_inputs}, f, indent=2)
                print(f"  💾 Test inputs saved to {test_inputs_file}")

            # Test the tool (if testing is enabled)
            if args.test:
                print(f"  ✅ Generated tool v{attempt}, testing...")
                tool_info = {
                    "action_name": f"tool_{tool_name}",
                    "description": tool_data.get("description", ""),
                    "file_path": version_file,
                }

                test_result = await test.test_tool(
                    args=args,
                    tool_info=tool_info,
                    tool_dirs=[tool_dir],
                    tool_type="base",
                    process_id=f"test_{process_id}_{attempt}",
                    max_steps=20,
                    include_validation=True,  # Enable validation for generation phase
                    attempt_number=attempt
                    - 1,  # Convert to 0-based indexing for round-robin
                )
            else:
                print(f"  ✅ Generated tool v{attempt}, skipping test...")
                test_result = test.create_skipped_test_result(args)

            # Record attempt
            attempt_record = {
                "version": attempt,
                "success": test_result["success"],
                "test_results": test_result,
                "failure_logs": test_result.get("failure_logs", ""),
                "timestamp": datetime.now().isoformat(),
            }
            results["base_attempts"].append(attempt_record)

            if test_result["success"]:
                if args.test:
                    print(f"  🎉 Base tool successful on attempt {attempt}!")
                else:
                    print(
                        f"  ✅ Base tool generated on attempt {attempt} (testing skipped)"
                    )
                # Save final successful base tool
                final_base_file = os.path.join(
                    tool_dir, f"{tool_name}.tool.json"
                )
                shutil.copy(version_file, final_base_file)
                break
            else:
                print(
                    f"  ❌ Attempt {attempt} failed: {test_result.get('errors', ['Unknown error'])}"
                )

        except Exception as e:
            print(f"  ❌ Attempt {attempt} generation failed: {e}")

            # If this is a JSON parsing error, add extra context
            error_msg = str(e)
            if "Could not parse response" in error_msg or "JSON" in error_msg:
                error_msg += " (Likely caused by overly verbose LLM response)"

            attempt_record = {
                "version": attempt,
                "success": False,
                "test_results": {"errors": [error_msg]},
                "failure_logs": f"Generation failed: {error_msg}",
                "timestamp": datetime.now().isoformat(),
            }
            results["base_attempts"].append(attempt_record)

    # Check if we have a successful base tool
    successful_base = any(attempt["success"] for attempt in results["base_attempts"])

    if successful_base:
        results["final_status"] = "base_only"
    else:
        print(
            f"  💀 Failed to create working base tool after {max_attempts} attempts"
        )
        results["final_status"] = "failed"

    # Phase 2: Always try optimization (if enabled) - works even if base failed
    if args.optimize and base_tool:
        print(f"  ⚡ Generating optimized version...")
        try:
            optimized_tool = optimize.optimize_tool(base_tool)

            # Save optimized tool
            optimized_file = os.path.join(
                tool_dir, f"{tool_name}.optimized.json"
            )
            FileManager.save_tool_json(optimized_tool, optimized_file)

            # Test optimized tool (if testing is enabled)
            if args.test:
                optimized_info = {
                    "action_name": f"tool_{tool_name}",
                    "description": tool_data.get("description", ""),
                    "file_path": optimized_file,
                }

                optimized_test_result = await test.test_tool(
                    args=args,
                    tool_info=optimized_info,
                    tool_dirs=[tool_dir],
                    tool_type="optimized",
                    process_id=f"test_opt_{process_id}",
                    max_steps=20,
                    include_validation=True,  # Enable validation for optimization phase
                    attempt_number=0,  # Optimized tests always use first option
                )
            else:
                optimized_test_result = test.create_skipped_test_result(args)

            results["optimization"] = {
                "success": optimized_test_result["success"],
                "test_results": optimized_test_result,
                "timestamp": datetime.now().isoformat(),
            }

            if optimized_test_result["success"]:
                if successful_base:
                    print(f"  🎉 Optimized tool also successful!")
                    results["final_status"] = "both_successful"
                else:
                    print(f"  🎉 Optimization recovery successful!")
                    results["final_status"] = "optimization_recovery"
            else:
                print(f"  ⚠️ Optimized tool failed")
                # Remove failed optimized file
                if os.path.exists(optimized_file):
                    os.remove(optimized_file)

        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")
            results["optimization"] = {
                "success": False,
                "test_results": {"errors": [str(e)]},
                "timestamp": datetime.now().isoformat(),
            }
    else:
        if not args.optimize:
            print(f"  ⏭️ Optimization skipped (not requested)")
        else:
            print(f"  ⏭️ Optimization skipped (no base tool generated)")
        results["optimization"] = {
            "success": True,
            "test_results": {
                "skipped": (
                    "Optimization not requested (use --optimize to enable)"
                    if not args.optimize
                    else "No base tool to optimize"
                )
            },
            "timestamp": datetime.now().isoformat(),
        }

    return results


async def _generate_tool_from_candidate(
    demonstrator_service, tool_data: Dict[str, Any], args, process_id: str
):
    """Generate tool from candidate data using demonstrator service."""
    prompt = _build_tool_prompt(tool_data, args)
    storage_state = getattr(args, 'auth_file', None)

    print(
        f"🎯 [Process {process_id}] Starting tool generation for: {tool_data.get('name', 'Unknown')}"
    )

    return await demonstrator_service.generate_tool_from_prompt(
        prompt=prompt,
        agent_llm=create_llm("openai", args.llm),
        extraction_llm=create_llm("openai", args.planner_llm),
        headless=True,
        storage_state=storage_state,
        max_steps=30,
    )


def _build_tool_prompt(tool_data: Dict[str, Any], args) -> str:
    """Build prompt from tool data for agent generation."""
    elements = tool_data.get("elements", [])

    # Create element interaction descriptions
    element_descriptions = []
    for element in elements:
        element_type = element.get("type", "unknown")
        purpose = element.get("purpose", "")
        element_descriptions.append(f"- {element_type}: {purpose}")

    elements_text = (
        "\n".join(element_descriptions)
        if element_descriptions
        else "No specific elements identified"
    )

    prompt = f"""Your task is to demonstrate the tool: {tool_data.get('description', 'No description provided')}

Starting URL: {tool_data.get('start_url', 'Not specified')}

Key interactions to demonstrate:
{elements_text}

IMPORTANT: Actually interact with controls to record stable selectors - avoid relying on dynamic content analysis when direct interaction is possible.

Please demonstrate ALL of these key interactions by actually using them, and explore and demonstrate any additional filtering, sorting, or refinement controls you discover that would be useful for this tool.
"""
    return prompt
