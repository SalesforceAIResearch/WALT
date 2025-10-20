from typing import Any, Dict, Generic, List, Optional, TypeVar

from walt.browser_use.agent.views import ActionResult, AgentHistoryList
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)


class ToolRunOutput(BaseModel, Generic[T]):
	"""Output of a tool run"""

	step_results: List[ActionResult | AgentHistoryList]
	output_model: Optional[T] = None


class StructuredtoolOutput(BaseModel):
	"""Base model for structured tool outputs.

	This can be used as a parent class for custom output models that
	will be filled by convert_results_to_output_model method.
	"""

	raw_data: Dict[str, Any] = Field(default_factory=dict, description='Raw extracted data from tool execution')

	status: str = Field(default='success', description='Overall status of the tool execution')

	error_message: Optional[str] = Field(default=None, description='Error message if the tool failed')
