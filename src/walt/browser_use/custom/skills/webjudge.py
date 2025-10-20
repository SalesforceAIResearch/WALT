import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from .utils import (
    build_message_with_images,
    format_action_history,
)
from walt.browser_use.agent.message_manager.utils import (
    extract_json_from_model_output,
)

logger = logging.getLogger(__name__)


class Config:
    DEFAULT_SCORE_THRESHOLD = 5
    MAX_SCREENSHOTS = 5
    EARLY_SAMPLES = 3


# System message constants
KEY_POINTS_SYSTEM_MSG = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with a JSON object**:
{
  "key_points": [
    "First key point",
    "Second key point",
    "Third key point"
  ]
}"""

IMAGE_JUDGE_SYSTEM_MSG = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze each provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Per-Image Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Assign a score using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

**CRITICAL**: You MUST only evaluate the images provided in this specific batch. Use image_index values starting from 0 and going up to N-1, where N is the number of images in this batch. For example, if you receive 3 images, use indices 0, 1, 2 only. Do NOT use any other indices.

**Respond with a valid JSON object**:
{
  "evaluations": [
    {"image_index": 0, "reasoning": "Explanation for the first image", "score": 3},
    {"image_index": 1, "reasoning": "Explanation for the second image", "score": 4}
  ]
}
"""

FINAL_EVAL_SYSTEM_MSG = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, and selected important web pages from the agent's trajectory, your goal is to determine whether the agent has completed the task and achieved all requirements.

**Important Note**: You are provided with the most important screenshots from the agent's trajectory, focusing on recent steps and key early actions. These represent the most relevant moments for task completion evaluation.

Your response must follow the following evaluation criteria:
1: Carefully check whether these snapshots and action history meet the tasks's key points. 
2: Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied, eg. by using a sort or filter operation.
3: The agent can make use of both visual and textual information from the webpage to complete a certain step, and must not be penalized for the specific approach it takes to complete a certain step as long as it is reasonable (e.g. using an image thumbnail to ascertain the color of an object rather than navigating to its page).
4: Ensure that the agent's final result matches what the task description asks for (e.g. if it tasks for a URL, the agent's final result should contain a URL)

**Evaluation Levels**:
- **"success"**: All key points completed, all requirements met, task fully accomplished
- **"partial"**: Some key points completed, meaningful progress made, but task not fully accomplished (e.g., correct filters applied but missing final submission, most requirements met but one missing, got close to goal but didn't complete final step)
- **"failure"**: Little to no progress made, major requirements missed, agent stuck or made errors preventing task completion

**Respond with a valid JSON object only**:
{
  "thoughts": "Your detailed reasoning process based on double-checking each key point and the evaluation criteria",
  "status": "success"
}

IMPORTANT: 
- The status must be exactly "success", "partial", or "failure"
- Return ONLY the JSON object, no additional text before or after
- Ensure the JSON is properly formatted with opening and closing braces
"""


def build_evaluation_messages(
    system_msg: str,
    task_text: str,
    input_image_paths: list = None,
    trajectory_images: list = None,
) -> list:
    """Helper function to build consistent message structures for LLM calls

    Args:
        system_msg: The system message content
        task_text: The main task/prompt text
        input_image_paths: Optional list of input image paths
        trajectory_images: Optional list of trajectory image content (pre-built)

    Returns:
        List of messages in LangChain format
    """
    messages = [SystemMessage(content=system_msg)]

    # Add input images if provided
    if input_image_paths:
        input_content = build_message_with_images("The input images are:", input_image_paths)
        messages.append(HumanMessage(content=input_content))

    # Add main task content and trajectory images
    if trajectory_images:
        # Build combined content: text + trajectory images
        main_content = [{"type": "text", "text": task_text}]

        # trajectory_images is a list of lists from build_message_with_images
        # We need to flatten them properly
        for img_content_list in trajectory_images:
            main_content.extend(img_content_list)

        messages.append(HumanMessage(content=main_content))
    else:
        messages.append(HumanMessage(content=task_text))

    return messages


def select_important_screenshots(
    screenshot_paths,
    max_screenshots=Config.MAX_SCREENSHOTS,
    early_samples=Config.EARLY_SAMPLES,
):
    """Select the most important screenshots using smart windowing approach

    Args:
        screenshot_paths: List of all screenshot paths
        max_screenshots: Maximum number of screenshots to select
        early_samples: Number of early trajectory samples to include

    Returns:
        List of tuples (original_index, screenshot_path) for selected screenshots
    """
    if not screenshot_paths:
        return []

    total = len(screenshot_paths)

    # Ensure we don't request more than available
    max_screenshots = min(max_screenshots, total)
    early_samples = min(early_samples, total)

    if total <= max_screenshots:
        return [(i, path) for i, path in enumerate(screenshot_paths)]

    selected_indices = set()

    # Always take the final screenshots (at least 5, up to 70% of max_screenshots, but not more than total)
    final_count = max(5, min(int(0.7 * max_screenshots), total))
    final_count = min(final_count, total)  # Ensure we don't exceed total
    final_start = max(0, total - final_count)
    selected_indices.update(range(final_start, total))

    # Add early samples if we have room and early trajectory exists
    if len(selected_indices) < max_screenshots and final_start > 0:
        remaining_slots = min(early_samples, max_screenshots - len(selected_indices))
        if remaining_slots > 0:
            step = max(1, final_start // remaining_slots)
            early_indices = list(range(0, final_start, step))[:remaining_slots]
            selected_indices.update(early_indices)

    return [(i, screenshot_paths[i]) for i in sorted(selected_indices)]


def extract_prediction(response):
    """Extract status from JSON response: 'success', 'partial', or 'failure'"""
    if not response or not response.strip():
        logger.warning("Empty response received")
        return "failure"

    try:
        # Try to clean up common JSON formatting issues
        cleaned_response = response.strip()

        # Handle case where response starts with field name without opening brace
        # Example: '"thoughts": "..." becomes {"thoughts": "..."
        if cleaned_response.startswith('"') and not cleaned_response.startswith("{"):
            cleaned_response = "{" + cleaned_response

        # If response ends without closing brace, add it
        if not cleaned_response.rstrip().endswith("}"):
            cleaned_response = cleaned_response.rstrip() + "}"

        data = extract_json_from_model_output(cleaned_response)
        status = data.get("status", "").lower()
        if status in ["success", "partial", "failure"]:
            return status
        else:
            logger.warning(f"Invalid status '{status}', defaulting to failure")
            return "failure"  # Default for invalid status
    except Exception as e:
        # Last resort: try regex extraction from raw response
        import re

        status_match = re.search(
            r'"status":\s*"(success|partial|failure)"', response, re.IGNORECASE
        )
        if status_match:
            status = status_match.group(1).lower()
            logger.info(f"Extracted status '{status}' from raw response via regex fallback")
            return status

        logger.warning(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response was: {repr(response)}")
        return "failure"


async def identify_key_points(task, input_image_paths, model):
    """Extract key points from task description"""
    text = f"Task: {task}"

    # Build messages using helper function
    messages = build_evaluation_messages(KEY_POINTS_SYSTEM_MSG, text, input_image_paths)

    response = await model.ainvoke(messages)
    # Always return string content, not AIMessage object
    return response.content if hasattr(response, 'content') else str(response)


async def judge_image_batch(task, input_image_paths, image_paths, key_points, model, start_idx):
    """Judge a batch of images in a single API call"""

    batch_size = len(image_paths)
    prompt = """Task: {task}

Key Points: {key_points}

BATCH INFO: You are evaluating {batch_size} images in this batch. Use image_index values 0 to {max_index} only.

"""
    messages = [SystemMessage(content=IMAGE_JUDGE_SYSTEM_MSG)]
    if input_image_paths:
        input_content = build_message_with_images("The input images are:", input_image_paths)
        messages.append(HumanMessage(content=input_content))

    text = prompt.format(
        task=task, key_points=key_points, batch_size=batch_size, max_index=batch_size - 1
    )
    messages.append(HumanMessage(content=text))

    image_content = build_message_with_images(
        f"The {batch_size} images to be evaluated are:", image_paths
    )
    messages.append(HumanMessage(content=image_content))

    response = await model.ainvoke(messages)

    batch_result = extract_json_from_model_output(response.content)
    evaluations = batch_result.get("evaluations", [])

    # Filter out evaluations with invalid image_index to prevent downstream errors
    valid_evaluations = []
    for eval_item in evaluations:
        image_idx = eval_item.get("image_index")
        if isinstance(image_idx, int) and 0 <= image_idx < batch_size:
            valid_evaluations.append(eval_item)
        else:
            logger.warning(
                f"LLM returned invalid image_index {image_idx} for batch of size {batch_size}, skipping"
            )

    return valid_evaluations


async def judge_images(task, input_image_paths, image_paths, key_points, model):
    """Judge only the most important images using smart windowing (single batch)"""
    # Select the most important screenshots using smart windowing
    selected_screenshots = select_important_screenshots(image_paths)

    if not selected_screenshots:
        return []

    # Extract paths and indices for evaluation
    selected_indices, selected_paths = zip(*selected_screenshots)

    # Evaluate the selected screenshots in a single batch
    batch_result = await judge_image_batch(
        task, input_image_paths, selected_paths, key_points, model, 0
    )

    # Map the results back to their original global indices
    for eval_item in batch_result:
        # eval_item["image_index"] tells us which image this evaluation is for
        batch_idx = eval_item.get("image_index", -1)
        # Map back to the original index in the full image_paths list
        if 0 <= batch_idx < len(selected_indices):
            original_idx = selected_indices[batch_idx]
            eval_item["global_index"] = original_idx
        else:
            logger.warning(
                f"Invalid image_index {batch_idx} for batch of size {len(selected_indices)}"
            )
            eval_item["global_index"] = -1  # Invalid index
        eval_item["original_total"] = len(image_paths)

    return batch_result


async def extract_task_key_points(task, input_image_paths, model):
    """Extract and format key points for a task"""
    key_points_raw = await identify_key_points(task, input_image_paths, model)
    key_points_data = extract_json_from_model_output(key_points_raw)
    return "\n".join(
        f"{i+1}. {key_point}" for i, key_point in enumerate(key_points_data["key_points"])
    )


def process_image_evaluations(evaluations, images_path, score_threshold):
    """Process evaluation results and select high-scoring images with their thoughts"""
    whole_content_img = []
    whole_thoughts = []
    record = []

    for eval_item in evaluations:
        score = eval_item.get("score", 0)
        thought = eval_item.get("reasoning", "")
        global_idx = eval_item["global_index"]

        record.append({"Response": json.dumps(eval_item), "Score": score})

        if score >= score_threshold:
            if global_idx >= 0 and global_idx < len(images_path):
                image_path = images_path[global_idx]
                whole_content_img.append(
                    build_message_with_images(f"The image {global_idx} is:", [image_path])
                )
                if thought:
                    whole_thoughts.append(thought)
            elif global_idx == -1:
                # Skip invalid index (already logged in judge_images)
                pass
            else:
                # This should no longer happen with the fixed mapping
                logger.error(
                    f"Unexpected: global_idx {global_idx} is out of bounds for images_path (length {len(images_path)})"
                )
                # Skip this evaluation item to prevent crash

    return whole_content_img, whole_thoughts, record


def build_final_evaluation_text(
    task, last_actions, action_thoughts, key_points, thoughts, has_images=True
):
    """Build the final evaluation text prompt"""
    if has_images:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The most important screenshots from the agent's trajectory (focusing on recent steps and key early actions) and their evaluation reasons:
{thoughts}"""
    else:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""

    return prompt.format(
        task=task,
        last_actions=format_action_history(last_actions, action_thoughts),
        key_points=key_points,
        thoughts="\n".join(f"{i+1}. {thought}" for i, thought in enumerate(thoughts)),
    )


async def webjudge_eval(
    task,
    input_image_paths,
    action_thoughts,
    last_actions,
    images_path,
    model,
    score_threshold=Config.DEFAULT_SCORE_THRESHOLD,
):
    # Input validation
    if not task or not task.strip():
        raise ValueError("Task cannot be empty")

    if not model:
        raise ValueError("Model cannot be None")

    # Ensure lists are not None
    images_path = images_path or []
    input_image_paths = input_image_paths or []
    last_actions = last_actions or []
    action_thoughts = action_thoughts or []

    # Validate score threshold
    if not isinstance(score_threshold, (int, float)) or score_threshold < 1 or score_threshold > 5:
        logger.warning(
            f"Invalid score_threshold {score_threshold}, using default {Config.DEFAULT_SCORE_THRESHOLD}"
        )
        score_threshold = Config.DEFAULT_SCORE_THRESHOLD

    key_points = await extract_task_key_points(task, input_image_paths, model)

    evaluations = await judge_images(task, input_image_paths, images_path, key_points, model)

    # Process evaluation results and select high-scoring ones
    whole_content_img, whole_thoughts, record = process_image_evaluations(
        evaluations, images_path, score_threshold
    )

    # Build evaluation text
    has_images = len(whole_content_img) > 0
    text = build_final_evaluation_text(
        task, last_actions, action_thoughts, key_points, whole_thoughts, has_images
    )

    # Build messages using helper function
    messages = build_evaluation_messages(
        FINAL_EVAL_SYSTEM_MSG, text, input_image_paths, whole_content_img
    )
    # Make final LLM call to get prediction
    response_obj = await model.ainvoke(messages)
    predicted_label = extract_prediction(response_obj.content)

    # Return all evaluation results
    return {
        "messages": messages,
        "text": text,
        "system_msg": FINAL_EVAL_SYSTEM_MSG,
        "record": record,
        "key_points": key_points,
        "response": response_obj.content,
        "predicted_label": predicted_label,
    }
