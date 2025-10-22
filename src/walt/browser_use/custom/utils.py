from pydantic import BaseModel
from walt.browser_use import Controller, ActionResult
from walt.browser_use.custom.browser_context_zoo import BrowserContextBugFix
import os
from langchain_openai import ChatOpenAI
import boto3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrock
import argparse
import ffmpy
import os
import json
import base64
import io
import subprocess
from PIL import Image, ImageFont
import platform
from walt.browser_use.agent.gif import _add_overlay_to_image
import asyncio
import logging
from typing import Optional, Union
from playwright.async_api import Page

# Import navigation functions from the new module
from walt.browser_use.custom.navigation_utils import robust_page_navigation_with_fallback

logger = logging.getLogger(__name__)


class LoginInCredentials(BaseModel):
    username: str
    password: str


def get_login_controller_and_action(credentials: LoginInCredentials):
    controller = Controller()

    @controller.action("Login to Salesforce website", param_model=LoginInCredentials)
    async def login_salesforce(
        params: LoginInCredentials, browser: BrowserContextBugFix
    ) -> ActionResult:
        page = await browser.get_current_page()
        await page.goto("https://login.salesforce.com")
        await page.get_by_label("Username").click()
        await page.get_by_label("Username").fill(params.username)
        await page.get_by_label("Password").click()
        await page.get_by_label("Password").fill(params.password)
        await page.get_by_role("button", name="Log In").click()
        return ActionResult(extracted_content="Salesfoce Login successful")

    initial_actions = [
        {
            "login_salesforce": {
                "username": credentials.username,
                "password": credentials.password,
            }
        }
    ]
    return controller, initial_actions


def create_llm(provider, model_name, temperature=0.0, random_seed=42):
    if provider == "openai":
        if (
            "o3" not in model_name
            and "o4" not in model_name
            and "gpt-5" not in model_name
        ):
            llm = ChatOpenAI(
                model=model_name, temperature=temperature, seed=random_seed
            )
        else:
            llm = ChatOpenAI(model=model_name, seed=random_seed)
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0
        )
    elif provider == "bedrock":
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2",
        )
        model_kwargs = {
            "max_tokens": 512,
            "temperature": 0.0,
        }
        llm = ChatBedrock(
            client=bedrock_runtime,
            model_id=model_name,
            model_kwargs=model_kwargs,
        )
    else:
        raise ValueError(f"Undefined provider:{provider}")
    return llm


def get_mp4(
    src_type,
    history_dir=None,
    history_filename=None,
    gif_dir=None,
    gif_filename=None,
    mp4_dir=None,
    mp4_file_name=None,
):
    font_size = 20
    title_font_size = 20
    goal_font_size = 20
    # Try to load nicer fonts
    try:
        # Try different font options in order of preference
        font_options = ["Helvetica", "Arial", "DejaVuSans", "Verdana"]
        font_loaded = False
        for font_name in font_options:
            try:
                if platform.system() == "Windows":
                    # Need to specify the abs font path on Windows
                    font_name = os.path.join(
                        os.getenv("WIN_FONT_DIR", "C:\\Windows\\Fonts"),
                        font_name + ".ttf",
                    )
                regular_font = ImageFont.truetype(font_name, font_size)
                title_font = ImageFont.truetype(font_name, title_font_size)
                goal_font = ImageFont.truetype(font_name, goal_font_size)
                font_loaded = True
                break
            except OSError:
                continue
        if not font_loaded:
            raise OSError("No preferred fonts found")

    except OSError:
        regular_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        goal_font = regular_font

    if not os.path.exists(mp4_dir):
        os.makedirs(mp4_dir)

    if src_type == "history":
        with open(os.path.join(history_dir, history_filename), "r") as f:
            history = json.load(f)
        steps = history["history"]
        images = []
        print(f"Converting {len(steps)} steps to mp4")
        for step in steps:
            # action_result = step['results']
            screenshot = step["state"]["screenshot"]
            step_number = step["metadata"]["step_number"] - 1
            next_goal = step["model_output"]["current_state"]["next_goal"]
            img_data = base64.b64decode(screenshot)
            image = Image.open(io.BytesIO(img_data))
            image = _add_overlay_to_image(
                image=image,
                step_number=step_number,
                goal_text=next_goal,
                regular_font=regular_font,
                title_font=title_font,
                margin=40,
                logo=None,
            )
            images.append(image)
        if images:
            temp_dir = os.path.join(mp4_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            fps = 24
            speed_factor = 0.02
            output_path = os.path.join(mp4_dir, mp4_file_name)
            for idx, img in enumerate(images):
                frame_path = os.path.join(
                    temp_dir, f"{idx:05d}.png"
                )  # Zero-padded filenames
                img.save(frame_path)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(temp_dir, "%05d.png"),  # Input sequence pattern
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-filter:v",
                f"setpts={1/speed_factor}*PTS",
                "-r",
                str(fps),  # Ensure correct output frame rate
                output_path,
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Video saved to {output_path}")
    elif src_type == "gif":
        ff = ffmpy.FFmpeg(
            inputs={f"{gif_dir}/{gif_filename}": None},
            outputs={f"{mp4_dir}/{mp4_file_name}": '-vf "setpts=PTS/2"'},
        )
        ff.run()
    else:
        raise ValueError("src_type must be either gif or history")


from PIL.Image import Image
from io import BytesIO


def pil_to_b64(img: Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def summarize_usage_info_from_jsonfied_trajectory(trajectory_jsonfied: dict) -> dict:
    """
    Get the summary of the trajectory.
    """
    summary = {
        "number_of_planner_called": 0,
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_prompt_tokens_cached": 0,
        "total_completion_tokens": 0,
        "reasoning_tokens": 0,
        "warnings": set(),
    }
    for step in trajectory_jsonfied:
        step_details = trajectory_jsonfied[step]
        usage_list = []
        if "get_plan" in step_details:
            summary["number_of_planner_called"] += 1
            get_plan_details = step_details["get_plan"]
            planner_usage = get_plan_details["usage"]
            usage_list.append(planner_usage)
        try:
            action_usage = step_details["get_next_action"]["usage"]
        except KeyError:
            # Step might be incomplete (e.g., hit step limit) - no get_next_action data
            action_usage = {}
        except Exception as e:
            print(f"Error getting action usage: {e}")
            action_usage = {}
        usage_list.append(action_usage)
        for usage in usage_list:
            if "total_cost" in usage:
                summary["total_cost"] += usage["total_cost"]
            else:
                summary["warnings"].add(
                    "No total cost found in the usage; the cost might be underestimated or the callback for the model provider is not used/implemented."
                )

            if "prompt_tokens" in usage:
                summary["total_prompt_tokens"] += usage["prompt_tokens"]
            else:
                summary["warnings"].add(
                    "No prompt tokens found in the usage; the cost might be underestimated or the callback for the model provider is not used/implemented."
                )

            if "prompt_tokens_cached" in usage:
                summary["total_prompt_tokens_cached"] += usage["prompt_tokens_cached"]
            else:
                summary["warnings"].add(
                    "No prompt tokens cached found in the usage; the cost might be underestimated or the callback for the model provider is not used/implemented."
                )

            if "completion_tokens" in usage:
                summary["total_completion_tokens"] += usage["completion_tokens"]
            else:
                summary["warnings"].add(
                    "No completion tokens found in the usage; the cost might be underestimated or the callback for the model provider is not used/implemented."
                )
            if "reasoning_tokens" in usage:
                summary["reasoning_tokens"] += usage["reasoning_tokens"]
            else:
                summary["warnings"].add(
                    "No reasoning tokens found in the usage; the cost might be underestimated or the callback for the model provider is not used/implemented."
                )
    # make warnings a list; so can be jsonified later
    summary["warnings"] = list(summary["warnings"])
    return summary