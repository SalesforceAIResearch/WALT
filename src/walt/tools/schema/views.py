from typing import List, Literal, Optional, Union, Dict, Any

from pydantic import BaseModel, Field, model_validator


# --- Base Step Model ---
# Common fields for all step types
class BaseToolStep(BaseModel):
	description: Optional[str] = Field(None, description="Description of the step's purpose.")
	output: Optional[str] = Field(None, description='Context key to store step output under.')
	# Allow other fields captured from raw events but not explicitly modeled
	model_config = {'extra': 'allow'}


# --- Steps that require interaction with a DOM element ---
class SelectorToolSteps(BaseToolStep):
	cssSelector: Optional[str] = Field(None, description='CSS selector for the target element.')
	xpath: Optional[str] = Field(None, description='XPath selector (often informational).')
	elementTag: Optional[str] = Field(None, description='HTML tag (informational).')

	elementHash: Optional[str] = Field(None, description='Hash of the element for demonstration tracking.')


# --- Agent Step ---
class AgentTaskToolStep(BaseToolStep):
	type: Literal['agent']
	task: str = Field(..., description='The objective or task description for the agent.')
	max_steps: Optional[int] = Field(
		None,
		description='Maximum number of iterations for the agent (default handled in code).',
	)

	# Agent steps might also have 'params' for other configs, handled by extra='allow'


# --- Deterministic Action Steps (based on controllers and examples) ---


# Actions from src/tools/controller/service.py & Examples
class NavigationStep(BaseToolStep):
	"""Navigates using the 'navigation' action (likely maps to go_to_url)."""

	type: Literal['navigation']  # As seen in examples
	url: Optional[str] = Field(None, description='Target URL to navigate to. Can use {context_var}.')
	
	# NEW: Add url_operation as alternative to url
	url_operation: Optional[Dict[str, Any]] = Field(
		None, 
		description='Structured URL manipulation: {"base": "{current_url}", "replace": {"param": "new_value"}}'
	)
	
	# NEW: Validation to ensure exactly one is provided
	@model_validator(mode='after')
	def validate_url_or_operation(self):
		if not self.url and not self.url_operation:
			raise ValueError('Either url or url_operation must be provided')
		if self.url and self.url_operation:
			raise ValueError('Cannot specify both url and url_operation')
		
		# Validate url_operation structure if provided
		if self.url_operation:
			if not isinstance(self.url_operation, dict):
				raise ValueError(f'url_operation must be a dict, got: {type(self.url_operation)}')
			if 'base' not in self.url_operation:
				raise ValueError(f'url_operation must contain "base" key, got: {self.url_operation}')
			# Check if base is just the string "base" (common LLM error)
			if self.url_operation.get('base') == 'base':
				raise ValueError('url_operation base cannot be just "base" - should be an actual URL or template like "{{current_url}}"')
		
		return self


class ClickStep(SelectorToolSteps):
	"""Clicks an element using 'click' (maps to tool controller's click)."""

	type: Literal['click']  # As seen in examples


class InputStep(SelectorToolSteps):
	"""Inputs text using 'input' (maps to tool controller's input)."""

	description: Optional[str] = Field(
		None,
		description="Description of the step's purpose. If neccesary describe the format that data should be in.",
	)

	type: Literal['input']  # As seen in examples

	value: str = Field(..., description='Value to input. Can use {context_var}.')


class SelectChangeStep(SelectorToolSteps):
	"""Selects a dropdown option using 'select_change' (maps to tool controller's select_change)."""

	type: Literal['select_change']  # Assumed type for tool controller's select_change

	selectedText: str = Field(..., description='Visible text of the option to select. Can use {context_var}.')


class KeyPressStep(SelectorToolSteps):
	"""Presses a key using 'key_press' (maps to tool controller's key_press)."""

	type: Literal['key_press']  # As seen in examples

	key: str = Field(..., description="The key to press (e.g., 'Tab', 'Enter').")


class ScrollStep(BaseToolStep):
	"""Scrolls the page using 'scroll' (maps to tool controller's scroll)."""

	type: Literal['scroll']  # Assumed type for tool controller's scroll
	scrollX: int = Field(..., description='Horizontal scroll pixels.')
	scrollY: int = Field(..., description='Vertical scroll pixels.')


class PageExtractionStep(BaseToolStep):
	"""Extracts text from the page using 'extract_page_content' (maps to tool controller's page_extraction)."""

	type: Literal['extract_page_content']  # Type for tool controller's page_extraction
	goal: str = Field(..., description='The goal of the page extraction.')


class WaitStep(BaseToolStep):
	"""Waits for a specified number of seconds."""
	
	type: Literal['wait']
	seconds: float = Field(..., description='Number of seconds to wait.')


# --- Union of all possible step types ---
# This Union defines what constitutes a valid step in the "steps" list.
DeterministicToolStep = Union[
	NavigationStep,
	ClickStep,
	InputStep,
	SelectChangeStep,
	KeyPressStep,
	ScrollStep,
	PageExtractionStep,
	WaitStep,
]

AgenticToolStep = AgentTaskToolStep


ToolStep = Union[
	# Pure tool
	DeterministicToolStep,
	# Agentic
	AgenticToolStep,
]

allowed_controller_actions = []


# --- Input Schema Definition ---
# (Remains the same)
class ToolInputSchemaDefinition(BaseModel):
	name: str = Field(
		...,
		description='The name of the property. This will be used as the key in the input schema.',
	)
	type: Literal['string', 'number', 'bool']

	format: Optional[str] = Field(
		None,
		description='Format of the input. If the input is a string, you can specify the format of the string.',
	)

	description: Optional[str] = Field(
		None,
		description='Description of the input parameter that will be shown to the LLM as field documentation.',
	)

	required: Optional[bool] = Field(
		default=None,
		description='None if the property is optional, True if the property is required.',
	)
	enum: Optional[List[str]] = Field(
		default=None,
		description='List of allowed values for string types with enum constraints.',
	)


# --- Top-Level tool Definition File ---
# Uses the Union ToolStep type


class ToolDefinitionSchema(BaseModel):
	"""Pydantic model representing the structure of the tool JSON file."""
	
	tool_analysis: Optional[str] = Field(
		None,
		description='A chain of thought reasoning about the tool. Think about which variables should be extracted.',
	)
	
	name: str = Field(..., description='The name of the tool.')
	description: str = Field(..., description='A human-readable description of the tool.')
	version: str = Field(..., description='The version identifier for this tool definition.')
	steps: List[ToolStep] = Field(
		...,
		min_length=1,
		description='An ordered list of steps (actions or agent tasks) to be executed.',
	)
	input_schema: list[ToolInputSchemaDefinition] = Field(
		# default=ToolInputSchemaDefinition(),
		description='List of input schema definitions.',
	)

	# Add loader from json file
	@classmethod
	def load_from_json(cls, json_path: str):
		with open(json_path, 'r') as f:
			return cls.model_validate_json(f.read())
