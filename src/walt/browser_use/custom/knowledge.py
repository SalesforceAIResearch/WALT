from typing import Dict, Optional, Tuple
import os
import json
import sys
import time
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

# Platform-specific imports for file locking

import fcntl

from walt.browser_use.custom.agent_zoo import LLMAgent
from walt.prompts.memory import (
    QUERY_FORMULATOR_SYSTEM_PROMPT as query_formulator_system_prompt,
    QUERY_FORMULATOR_TASK_PROMPT as query_formulator_task_prompt,
    NARRATIVE_MEMORY_SYSTEM_PROMPT as narrative_mem_system_prompt
)
import logging

logger = logging.getLogger(__name__)

class BaseModule:
    def __init__(
        self,   
        llm: BaseChatModel,
        platform: str,
        ):
        self.llm = llm
        self.platform = platform

    def _create_agent(self, system_prompt: str) -> LLMAgent:
        """Create a new LMMAgent instance"""
        agent = LLMAgent(self.llm, system_prompt)
        return agent
    

    
class QueryRephraser(BaseModule):
    def __init__(
                self, 
                llm: BaseChatModel,
                platform: str,
                local_kb_path: str
                ):
        
        super().__init__(llm, platform)

        self.local_kb_path = local_kb_path
        # create self.local_kb_path if it doesn't exist
        if not os.path.exists(self.local_kb_path):
            os.makedirs(self.local_kb_path)
        
        system_prompt = query_formulator_system_prompt.format(CURRENT_OS=self.platform)
        self.agent = self._create_agent(system_prompt)

    def formulate_query(self, instruction: str, reset_message_manager: bool = False) -> Tuple[str, dict]:
        """Formulate search query based on instruction and current state"""
        query_path = os.path.join(self.local_kb_path, "formulate_query.json")
        formulate_query = {}

        if instruction in formulate_query:
            return formulate_query[instruction], {'information': 'get from local kb; no usage'}

        formatted_instruction = query_formulator_task_prompt.format(INSTRUCTION=instruction)
        ai_message, usage = self.agent.get_response(formatted_instruction,
                                                    reset_message_manager=reset_message_manager)
        response = ai_message.content
        rephrased_query = response.strip().replace('"', "")
        # import pdb; pdb.set_trace()        
        formulate_query[instruction] = rephrased_query
        # save formulate_query to query_path
        with open(query_path, "w") as f:
            json.dump(formulate_query, f, indent=4)
        return rephrased_query, usage



    
class NarrativeMemorySummarizer(BaseModule):
    def __init__(
        self,
        llm: BaseChatModel,
        platform: str,
        local_kb_path: str
    ):
        super().__init__(llm, platform)

        self.local_kb_path = local_kb_path
        # create self.local_kb_path if it doesn't exist
        if not os.path.exists(self.local_kb_path):
            os.makedirs(self.local_kb_path)
        # create narrative_memory.json if it doesn't exist
        if not os.path.exists(os.path.join(self.local_kb_path, "narrative_memory.json")):
            with open(os.path.join(self.local_kb_path, "narrative_memory.json"), "w") as f:
                json.dump({}, f, indent=4)
                logger.info(f"✅ Created an empty narrative memory file at {os.path.join(self.local_kb_path, 'narrative_memory.json')}")

        system_prompt = narrative_mem_system_prompt.format(CURRENT_OS=self.platform)
        self.agent = self._create_agent(system_prompt)
        
    def summarize_narrative_memory(self, formatted_instruction: str, reset_message_manager: bool = False) -> Tuple[str, dict]:
        """Summarize the narrative memory"""
        ai_message, usage = self.agent.get_response(formatted_instruction,
                                                                          reset_message_manager=reset_message_manager)
        response = ai_message.content
        return response, usage
    
    def save_narrative_memory(self, narrative_memory: str, formulated_search_query: str, force_overwrite: bool = False):
        """Save the narrative memory with file locking for multiprocessing safety"""
        narrative_memory_path = os.path.join(self.local_kb_path, "narrative_memory.json")
        
        # Retry mechanism for file locking
        max_retries = 10
        for attempt in range(max_retries):
            try:

                # Unix file locking (original implementation)
                with open(narrative_memory_path, "a+") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    f.seek(0)
                    content = f.read()
                    
                    if content.strip():
                        narrative_memory_dict = json.loads(content)
                    else:
                        narrative_memory_dict = {}
                    
                    if formulated_search_query in narrative_memory_dict:
                        logger.warning(f"⚠️ Formulated search query already exists in narrative memory.\n{formulated_search_query}\n")
                        if not force_overwrite:
                            logger.warning(f"⚠️ Will not overwrite the existing narrative memory.")
                            return
                    
                    narrative_memory_dict[formulated_search_query] = narrative_memory
                    
                    # Write atomically using temp file + rename
                    temp_path = narrative_memory_path + f".tmp.{os.getpid()}"
                    with open(temp_path, "w") as temp_f:
                        json.dump(narrative_memory_dict, temp_f, indent=4)
                    
                    os.rename(temp_path, narrative_memory_path)
            
                break
                
            except (IOError, OSError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed to write narrative memory: {e}. Retrying...")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Failed to write narrative memory after {max_retries} attempts: {e}")
                    raise
