#!/usr/bin/env python3
"""
Candidate tool proposal phase.
Explores websites and discovers potential tools.
"""

import os
import json
from typing import Dict, Any, List

from walt.browser_use.agent.service import Agent
from walt.browser_use.agent.message_manager.utils import extract_json_from_model_output
from walt.browser_use.custom.utils import create_llm

from .utils import setup_browser_environment, cleanup_browser_environment


async def discover_candidates(args) -> List[Dict[str, Any]]:
    """Phase 1: Discover candidate tools by exploring the website."""
    base_url = args.base_url
    print(f"🔍 Starting candidate discovery for {base_url}")

    exploration_file = os.path.join(args.output_dir, "exploration_result.json")

    # Discover new candidate tools
    print("🔍 Discovering new candidate tools...")
    tools_json = await _explore_and_extract_tools(args, base_url)

    # Save the results
    with open(exploration_file, "w") as f:
        json.dump({"tools": tools_json}, f, indent=4)

    print(f"📋 Discovered {len(tools_json)} candidate tools")
    return tools_json


def load_existing_candidates(args) -> List[Dict[str, Any]]:
    """Load existing candidates from exploration file."""
    exploration_file = os.path.join(args.output_dir, "exploration_result.json")
    
    if not os.path.exists(exploration_file):
        return []
    
    with open(exploration_file, "r") as f:
        data = json.load(f)
        return data.get("tools", [])


async def _explore_and_extract_tools(args, base_url: str) -> List[Dict[str, Any]]:
    """Explore the website and extract tool candidates."""
    # Get auth file if provided
    storage_state = getattr(args, 'auth_file', None)
    
    # Setup browser
    browser, browser_context = await setup_browser_environment(
        base_url=base_url,
        storage_state=storage_state,
        headless=True,
    )

    try:
        # Create exploration agent
        planner_llm = create_llm("openai", args.planner_llm)
        browser_agent_llm = create_llm("openai", args.llm)

        # Create generic agent
        agent = Agent(
            task=_build_exploration_prompt(base_url),
            planner_llm=planner_llm,
            planner_interval=15,
            llm=browser_agent_llm,
            browser=browser,
            browser_context=browser_context,
            use_vision=True,
            max_failures=3,
        )

        print(f"🤖 Agent exploring {base_url}...")
        history = await agent.run(max_steps=30)

        # Extract tools from agent result
        raw_result = history.final_result()

        try:
            parsed_json = extract_json_from_model_output(raw_result)
            return parsed_json.get("tools", [])
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Raw result: {raw_result[:200]}...")
            return []

    finally:
        # Cleanup browser environment
        await cleanup_browser_environment(browser, browser_context)


def _build_exploration_prompt(base_url: str) -> str:
    """Build the exploration prompt for the discovery agent."""
    from walt.prompts.discovery import get_exploration_prompt
    
    return get_exploration_prompt(base_url)
