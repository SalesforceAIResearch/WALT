"""
Generic Verify with Judge Skill

Browser-agnostic task verification skill using webjudge evaluation system.
"""

import tempfile
import base64
import io
import json
import os
import shutil
from typing import Union, Optional, List
import PIL
from langchain_core.language_models.chat_models import BaseChatModel
from walt.browser_use.agent.views import ActionResult, AgentHistoryList

# Import browser context types
from walt.browser_use.custom.eval_envs.VWA import VWABrowserContext
from walt.browser_use.custom.eval_envs.WA import WABrowserContext
from walt.browser_use.custom.utils import create_llm

from .models import VerifyAction
from .webjudge import webjudge_eval


async def verify_with_judge(
    params: VerifyAction,
    agent_history: AgentHistoryList,
    browser: Union["VWABrowserContext", "WABrowserContext"],
) -> ActionResult:
    """
    Verify agent task completion using webjudge evaluation system. Can only be used after the agent issues a done action.

    Analyzes the agent's action history, screenshots, and final results to determine
    whether the agent successfully completed the given task according to evaluation criteria.

    This is the preferred method for inline verification during agent execution.
    """
    if not agent_history.is_done():
        return ActionResult(
            extracted_content="Error: cannot verify task completion before agent self-reports task completion by issuing a done action"
        )

    try:
        # Get current page for context
        page = await browser.get_current_page()
        screenshots = agent_history.screenshots()
        # save to temp dir
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        for i, screenshot in enumerate(screenshots):
            if screenshot is None:
                continue  # Skip None screenshots

            try:
                screenshot_path = os.path.join(temp_dir, f"screenshot_{i}.png")
                with open(screenshot_path, "wb") as f:
                    # Handle data URL format
                    if screenshot.startswith("data:image/png;base64,"):
                        b64_data = screenshot.split(",")[1]
                    else:
                        b64_data = screenshot

                    image_bytes = base64.b64decode(b64_data)
                    image = PIL.Image.open(io.BytesIO(image_bytes))
                    image.save(f)
                    image_paths.append(screenshot_path)
            except Exception as e:
                print(f"Warning: Failed to process screenshot {i}: {e}")
                continue

        actions = agent_history.model_actions()

        thoughts = [
            step.model_output.current_state.next_goal
            for step in agent_history.history
            if step.model_output is not None
            and step.model_output.current_state is not None
        ]

        judge_model_llm = create_llm(provider="openai", model_name=params.judge_model)
        eval_results = await webjudge_eval(
            params.task,
            params.task_image_paths,
            thoughts,
            actions,
            image_paths,
            judge_model_llm,
            params.score_threshold,
        )

        # Extract reasoning from the JSON response
        response_content = eval_results["response"]
        
        # Defensive: ensure we have a string (handle any AIMessage objects that slip through)
        if hasattr(response_content, "content"):
            response_content = response_content.content
        response_content = str(response_content)
        
        try:
            response_json = json.loads(response_content)
            reasoning = response_json.get("thoughts", response_content)
        except json.JSONDecodeError:
            # If response isn't valid JSON, use it directly as reasoning
            reasoning = response_content

        evaluation_results = {
            "whether_agent_succeeded_according_to_judge": eval_results[
                "predicted_label"
            ],
            "reasoning": reasoning,
        }
        return ActionResult(
            extracted_content=json.dumps(evaluation_results),
            include_in_memory=True,
        )
    except Exception as e:
        print(f"Judge evaluation failed: {e}")
        return ActionResult(
            extracted_content=json.dumps(
                {
                    "whether_agent_succeeded_according_to_judge": "failure",
                    "reasoning": f"Judge evaluation failed: {str(e)}",
                }
            ),
            include_in_memory=True,
        )
