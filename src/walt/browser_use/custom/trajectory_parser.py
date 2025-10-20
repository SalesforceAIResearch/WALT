import json
import re
import unicodedata
from typing import Dict, Any, List, Optional


def is_base64_data(text: str) -> bool:
    """Check if text contains base64 encoded data (likely images)."""
    if not isinstance(text, str):
        return False
    
    # Look for base64 data URI patterns
    base64_patterns = [
        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
        r'"image_url":\s*{\s*"url":\s*"data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
    ]
    
    for pattern in base64_patterns:
        if re.search(pattern, text):
            return True
    
    # Check for large base64-like strings (>100 chars of base64 characters)
    base64_like = re.findall(r'[A-Za-z0-9+/=]{100,}', text)
    return len(base64_like) > 0


def extract_system_prompt_from_trajectory(trajectory_data: Dict[str, Any]) -> Optional[str]:
    """Extract the system prompt from the first step."""
    steps = trajectory_data.get("steps", {})
    if not steps:
        return None
    
    first_step = list(steps.values())[0]
    input_messages = first_step.get("input_messages", {}).get("contents", [])
    
    for message in input_messages:
        if message.get("type") == "system":
            return message.get("content", "")
    
    return None


def extract_step_essentials_from_trajectory(step_num: str, step_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential information from a single step."""
    simplified_step = {
        "step": int(step_num),
        "has_screenshot": False,
        "goal": None,
        "action": None,
        "action_details": None,  # NEW: Store action arguments
        "evaluation": None,
        "memory": None,
        "url": None,
    }
    
    # Extract goal, evaluation, and memory from output messages
    output_messages = step_data.get("output_messages", {})
    tool_call_message = output_messages.get("tool_call_message", {})
    
    if "tool_calls" in tool_call_message:
        tool_calls = tool_call_message["tool_calls"]
        if tool_calls and len(tool_calls) > 0:
            last_tool_call = tool_calls[-1]
            args = last_tool_call.get("args", {})
            
            current_state = args.get("current_state", {})
            simplified_step["goal"] = current_state.get("next_goal")
            simplified_step["evaluation"] = current_state.get("evaluation_previous_goal")
            simplified_step["memory"] = current_state.get("memory")
            
            # Extract actions and their arguments
            actions = args.get("action", [])
            if actions:
                action_names = []
                action_details = []
                for action in actions:
                    if isinstance(action, dict):
                        for action_name, action_params in action.items():
                            action_names.append(action_name)
                            # Store action with its parameters for debugging
                            action_details.append({
                                "action": action_name,
                                "params": action_params
                            })
                simplified_step["action"] = action_names
                simplified_step["action_details"] = action_details
    
    # Check for images in input messages
    input_messages = step_data.get("input_messages", {}).get("contents", [])
    for message in input_messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    simplified_step["has_screenshot"] = True
                    break
        if simplified_step["has_screenshot"]:
            break
    
    # Extract URL if available
    if input_messages:
        for message in input_messages:
            content = message.get("content", "")
            if isinstance(content, str) and "Current URL:" in content:
                lines = content.split("\n")
                for line in lines:
                    if line.strip().startswith("Current URL:"):
                        simplified_step["url"] = line.replace("Current URL:", "").strip()
                        break
                break
    
    return simplified_step


def create_simplified_trajectory(trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a simplified version of the trajectory data."""
    
    # Extract basic info
    simplified = {
        "task_prompt": trajectory_data.get("task_prompt"),
        "is_successful": trajectory_data.get("is_successful"),
        "total_steps": len(trajectory_data.get("steps", {})),
        # "system_prompt": extract_system_prompt_from_trajectory(trajectory_data),
        "simplified_steps": [],
        "final_result": None
    }
    
    steps = trajectory_data.get("steps", {})
    
    # Process each step
    for step_num, step_data in steps.items():
        if step_num == "0":  # Skip initial actions step
            continue
            
        simplified_step = extract_step_essentials_from_trajectory(step_num, step_data)
        simplified["simplified_steps"].append(simplified_step)
    
    # Try to extract final result from the last step
    if steps:
        last_step_data = list(steps.values())[-1]
        controller_messages = last_step_data.get("controller_messages", {})
        action_results = controller_messages.get("action_result", [])
        
        for result in action_results:
            if isinstance(result, dict):
                content = result.get("content", "")
                if isinstance(content, str) and "Action result:" in content:
                    simplified["final_result"] = content.replace("Action result:", "").strip()
                    break
    
    return simplified


def human_trajectory_parser(
    path_to_trajectory: str, 
    include_observation: bool = False):
    with open(path_to_trajectory, 'r') as f:
        trajectory = json.load(f)
    
    task_query = trajectory['task_prompt']
    trajectory_description = f'TASK:{task_query}\n The task is successful. The trajectory is as follows:\n'
    for step_data in trajectory['details']:
        step_id = step_data['step']
        obs = ''
        if include_observation:
            interactive_elements = step_data['interactive_elements']
            available_tabs = step_data['available_tabs']
            available_tabs_str = '\n'.join([f"{struct['page_id']}: {struct['title']}" for struct in available_tabs])
            obs += f'The page description is:\n{interactive_elements}.\nThe available tabs are:\n{available_tabs_str}.'
        selectors = step_data['selectors']
        user_output = step_data['user_output']
        
        action_description = ''
        # the user_output is a list of one action for human mode
        action = json.loads(user_output['actions'][0])
        action_name, action_args = [*action.items()][0]
        action_description += f'Perform the action: {action_name}.'
        if 'index' in action_args:
            element_node = selectors[str(action_args['index'])]
            element_description = element_node["textual_description"]
            if element_description == "":
                element_description = "[ERROR: No element description]"
            action_description = action_description.split('.')[0]
            action_description += f' on the element: {element_description}.'
        if 'text' in action_args and action_args['text'] != '':
            text = action_args['text']
            action_description = action_description.split('.')[0]
            action_description += f' with the text: {text}.'
        if 'obs' in step_data:        
            step_description = f'STEP:{step_id+1}\n{obs}\n{action_description}\n'
        else:
            step_description = f'STEP:{step_id+1}\n{action_description}\n'
        trajectory_description += step_description + '\n'
    return trajectory_description

def tutorial_like_text_parser(
    path_to_text: str
    ):
    with open(path_to_text, 'r') as f:
        data = json.load(f)
        
    task_query = data['task_prompt']
    trajectory_description = f'TASK:{task_query}\n'
    is_successful = data['is_successful']
    if is_successful:
        trajectory_description += 'The task is successful. The trajectory is as follows:\n'
    else:
        trajectory_description += 'The task is failed. The trajectory is as follows:\n'
    steps = data['steps']
    trajectory_description += steps
    return trajectory_description


import re
import unicodedata

def clean_action_result(text):
    # Remove "Action result:" prefix
    text = re.sub(r'^Action result:\s*', '', text)

    # Remove all non-printable characters, emojis, and special unicode characters
    # Keep only standard ASCII characters, spaces, and basic punctuation
    cleaned_text = ''
    for char in text:
        # Only keep ASCII characters, spaces, and basic punctuation (including underscore)
        if (ord(char) < 128 and (char.isalnum() or char.isspace() or char in '.,!?:;()/-\'\"_')):
            cleaned_text += char
    
    # Remove multiple spaces and strip
    cleaned_text = ' '.join(cleaned_text.split())
    cleaned_text = cleaned_text.strip()
    
    # Transform any "with index X: Label" to "with label [Label]"
    pattern = r'(.*?) with index \d+: (.+)'
    match = re.match(pattern, cleaned_text)
    if match:
        action = match.group(1)  # The action part (e.g., "Clicked button", "Typed into textbox")
        label = match.group(2)   # The label part
        cleaned_text = f'{action} with label [{label}]'
    
    return cleaned_text


def agent_trajectory_parser( 
    path_to_trajectory: str, 
    include_observation: bool = False,
    include_plan: bool = False,
    ):
    
    with open(path_to_trajectory, 'r') as f:
        trajectory = json.load(f)
    
    task_query = trajectory['task_prompt']
    task_status = trajectory['is_successful']
    if task_status is True:
        task_status = 'successful'
    else:
        task_status = 'failed'  
    trajectory_description = f'TASK:{task_query}\nThe task is {task_status}. The trajectory is as follows:\n'
    steps = trajectory['steps']
    for step_id, step_data in steps.items():
        if step_id == '0':
            # skip the initial actions
            continue
        
        # Defensive programming: check if output_messages exists
        if 'output_messages' not in step_data:
            print(f"Warning: Step {step_id} missing output_messages, skipping...")
            continue
            
        try:
            predicted_actions_lst = step_data['output_messages']['tool_call_message']['tool_calls'][0]['args']['action']
        except (KeyError, IndexError) as e:
            print(f"Warning: Step {step_id} has malformed output_messages structure: {e}, skipping...")
            continue
        for action_dict in predicted_actions_lst:
            action_name = list(action_dict.keys())
            if len(action_name) == 0:
                # this means the agent failed to generate an action
                action_name = ''
            else:
                action_name = action_name[0]
            if action_name == 'done':
                step_description = f"Finish with the answer:\n{action_dict['done']['text']}"
                trajectory_description += f'STEP:{step_id}\n{step_description}\n'
                return trajectory_description        
        
        step_description = ""    
        action_result = step_data['controller_messages']['action_result']
        action_error = step_data['controller_messages']['action_error']
        if len(action_result) > 0:
            for item in action_result:
                step_description += clean_action_result(item['content']) + "."
        elif len(action_error) > 0:
            for item in action_error:
                step_description += item['content']

        trajectory_description += f'STEP:{step_id}\n{step_description}\n'
        
    return trajectory_description