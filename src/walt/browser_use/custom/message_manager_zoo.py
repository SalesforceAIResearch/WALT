from __future__ import annotations

import logging
from typing import Dict, List, Optional

from langchain_core.messages import (
	BaseMessage,
    AIMessage,
	HumanMessage,
	SystemMessage,
)
from pydantic import BaseModel
from PIL.Image import Image
from walt.browser_use.agent.message_manager.views import MessageMetadata
from walt.browser_use.agent.message_manager.service import MessageManagerSettings, MessageManager
from walt.browser_use.agent.views import MessageManagerState
from walt.browser_use.utils import time_execution_sync
import copy

logger = logging.getLogger(__name__)


class PlainMessageManager:
    def __init__(
        self,
        system_message: SystemMessage,
        settings: MessageManagerSettings = None,
        state: MessageManagerState = None,
    ):
        self.settings = settings if settings is not None else MessageManagerSettings()
        self.state = state if state is not None else MessageManagerState()
        self.original_state = copy.deepcopy(self.state)
        self.system_prompt = system_message

    def _add_message_with_tokens(self, message: BaseMessage, position: int | None = None) -> None:
        """Add message with token count metadata
        position: None for last, -1 for second last, etc.
        """

        # filter out sensitive data from the message
        if self.settings.sensitive_data:
            message = self._filter_sensitive_data(message)

        token_count = self._count_tokens(message)
        metadata = MessageMetadata(tokens=token_count)
        self.state.history.add_message(message, metadata, position)

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        tokens = len(text) // self.settings.estimated_characters_per_token  # Rough estimate if no tokenizer available
        return tokens
        
    def _count_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a message using the model's tokenizer"""
        tokens = 0
        if isinstance(message.content, list):
            for item in message.content:
                if 'image_url' in item:
                    tokens += self.settings.image_tokens
                elif isinstance(item, dict) and 'text' in item:
                    tokens += self._count_text_tokens(item['text'])
        else:
            msg = message.content
            if hasattr(message, 'tool_calls'):
                msg += str(message.tool_calls)  # type: ignore
            tokens += self._count_text_tokens(msg)
        return tokens        
    
    @time_execution_sync('--filter_sensitive_data')
    def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
        """Filter out sensitive data from the message"""

        def replace_sensitive(value: str) -> str:
            if not self.settings.sensitive_data:
                return value
            for key, val in self.settings.sensitive_data.items():
                if not val:
                    continue
                value = value.replace(val, f'<secret>{key}</secret>')
            return value

        if isinstance(message.content, str):
            message.content = replace_sensitive(message.content)
        elif isinstance(message.content, list):
            for i, item in enumerate(message.content):
                if isinstance(item, dict) and 'text' in item:
                    item['text'] = replace_sensitive(item['text'])
                    message.content[i] = item
        return message    
    
    @time_execution_sync('--get_messages')
    def get_messages(self) -> List[BaseMessage]:
        """Get current message list, potentially trimmed to max tokens"""

        msg = [m.message for m in self.state.history.messages]
        # debug which messages are in history with token count # log
        total_input_tokens = 0
        logger.debug(f'Messages in history: {len(self.state.history.messages)}:')
        for m in self.state.history.messages:
            total_input_tokens += m.metadata.tokens
            logger.debug(f'{m.message.__class__.__name__} - Token count: {m.metadata.tokens}')
        logger.debug(f'Total input tokens: {total_input_tokens}')

        return msg    
        
    def prepare_llm_input(self, instruction: str):
        # Only initialize messages if state is empty
        if len(self.state.history.messages) == 0:
            self._add_message_with_tokens(self.system_prompt)
            task_message = HumanMessage(content=instruction)
            self._add_message_with_tokens(task_message)
            if self.settings.sensitive_data:
                info = f'Here are placeholders for sensitve data: {list(self.settings.sensitive_data.keys())}'
                info += 'To use them, write <secret>the placeholder name</secret>'
                info_message = HumanMessage(content=info)
                self._add_message_with_tokens(info_message)            
        else:
            raise ValueError("state.history.messages is not empty.")
        
    def reset(self):
        self.state = copy.deepcopy(self.original_state)
        
        
from walt.browser_use.custom.utils import pil_to_b64
class MessageManagerWithImages(MessageManager):
    def __init__(
        self,
        task: str,
        task_image: List[Image] | None,
        system_message: SystemMessage,
        settings: MessageManagerSettings = MessageManagerSettings(),
        state: MessageManagerState = MessageManagerState(),
    ):
        self.task = task
        self.task_image = task_image
        self.settings = settings
        self.state = state
        self.system_prompt = system_message

        # Only initialize messages if state is empty
        if len(self.state.history.messages) == 0:
            self._init_messages()

    def _init_messages(self) -> None:
        """Initialize the message history with system message, context, task, and other initial messages"""
        self._add_message_with_tokens(self.system_prompt)

        if self.settings.message_context:
            context_message = HumanMessage(content='Context for the task' + self.settings.message_context)
            self._add_message_with_tokens(context_message)

        if self.task_image is None:
            task_message = HumanMessage(
                content=f'Your ultimate task is: """{self.task}""". If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
            )
        else:
            if len(self.task_image) > 0:
                content = [
                    {"type": "text",
                    "text": f'Your ultimate task is: """{self.task}""". This task involves image input(s) provided below. If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
                    }
                ]
                for idx, pil_image in enumerate(self.task_image):
                    content.extend(
                        [
                            {"type": "text", "text": f"Input Image {idx + 1}:"},
                            {"type": "image_url", "image_url": {"url": pil_to_b64(pil_image)}}
                        ]
                    )
            else:
                raise ValueError("task_image is empty")
            task_message = HumanMessage(content=content)
            
        self._add_message_with_tokens(task_message)

        if self.settings.sensitive_data:
            info = f'Here are placeholders for sensitve data: {list(self.settings.sensitive_data.keys())}'
            info += 'To use them, write <secret>the placeholder name</secret>'
            info_message = HumanMessage(content=info)
            self._add_message_with_tokens(info_message)

        placeholder_message = HumanMessage(content='Example output:')
        self._add_message_with_tokens(placeholder_message)

        tool_calls = [
            {
                'name': 'AgentOutput',
                'args': {
                    'current_state': {
                        'evaluation_previous_goal': 'Success - I opend the first page',
                        'memory': 'Starting with the new task. I have completed 1/10 steps',
                        'next_goal': 'Click on company a',
                    },
                    'action': [{'click_element': {'index': 0}}],
                },
                'id': str(self.state.tool_id),
                'type': 'tool_call',
            }
        ]

        example_tool_call = AIMessage(
            content='',
            tool_calls=tool_calls,
        )
        self._add_message_with_tokens(example_tool_call)
        self.add_tool_message(content='Browser started')

        placeholder_message = HumanMessage(content='[Your task history memory starts here]')
        self._add_message_with_tokens(placeholder_message)

        if self.settings.available_file_paths:
            filepaths_msg = HumanMessage(content=f'Here are file paths you can use: {self.settings.available_file_paths}')
            self._add_message_with_tokens(filepaths_msg)
        
        
        
        