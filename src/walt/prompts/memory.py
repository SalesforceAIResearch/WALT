"""Memory-related prompts."""

QUERY_FORMULATOR_SYSTEM_PROMPT = """Given a browser task instruction, you are an agent which should provide useful information as requested, to help another agent follow the instruction and perform the task in {CURRENT_OS} platform.
"""

QUERY_FORMULATOR_TASK_PROMPT = """The original task instruction is: {INSTRUCTION}
Please rephrase the task instruction to make it more specific and clear. Also correct any grammatical errors or awkward phrasing.
Please ONLY provide the rephrased task instruction.\nOutput:"""


NARRATIVE_MEMORY_SYSTEM_PROMPT = """You are a summarization agent designed to analyze a trajectory of browser task execution.
You have access to 
1. the task description,  
2. whether the trajectory is successful or not, and
3. the whole trajectory details.
Your summarized information will be referred to by another agent when performing the tasks in {CURRENT_OS} platform.
You should follow the below instructions:
1. If the task is successfully executed, you should summarize the successful plan based on the whole trajectory to finish the task.
2. Otherwise, provide the reasons why the task is failed and potential suggestions that may avoid this failure. Especially pay repeated patterns in the trajectory, which may indicate the limitations of the agent. 

**Important**
Verify the whether the trajectory is successful or not very carefully as it is explicitly provided after the task description.

**ATTENTION**
* Only extract the correct plan and do not provide redundant steps.
* If only step level action is provided, try to infer the high-level plan and group the actions into a plan.
* If there are the successfully used hot-keys, make sure to include them in the plan.
* The suggestions are for another agent not human, so they must be doable through the agent's action.
* Don't generate high-level suggestions (e.g., Implement Error Handling).


**Common Failure Patterns**
1. If the task is related to search for something, and the agent failed. It's likely that the agent did not use the filtering and sorting options to narrow down the search results.
"""


def get_query_formulator_prompt(instruction: str, current_os: str = "macOS") -> str:
    """
    Build the query formulator prompt for memory retrieval.
    
    Args:
        instruction: The user's instruction/goal
        current_os: The current operating system
        
    Returns:
        The formatted query formulator prompt
    """
    system_prompt = QUERY_FORMULATOR_SYSTEM_PROMPT.replace("{CURRENT_OS}", current_os)
    task_prompt = QUERY_FORMULATOR_TASK_PROMPT.replace("{INSTRUCTION}", instruction)
    return system_prompt + "\n\n" + task_prompt


def get_narrative_memory_system(current_os: str = "macOS") -> str:
    """
    Get the narrative memory system prompt.
    
    Args:
        current_os: The current operating system
        
    Returns:
        The narrative memory system prompt
    """
    return NARRATIVE_MEMORY_SYSTEM_PROMPT.replace("{CURRENT_OS}", current_os)

