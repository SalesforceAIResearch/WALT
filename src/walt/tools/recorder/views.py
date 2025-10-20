from typing import Literal, Union

from pydantic import BaseModel

from walt.tools.schema.views import ToolDefinitionSchema

# --- Event Payloads ---


class RecordingStatusPayload(BaseModel):
	message: str


# --- Main Event Models (mirroring HttpEvent types from message-bus-types.ts) ---


class BaseHttpEvent(BaseModel):
	timestamp: int


class HttptoolUpdateEvent(BaseHttpEvent):
	type: Literal['tool_UPDATE'] = 'tool_UPDATE'
	payload: ToolDefinitionSchema


class HttpRecordingStartedEvent(BaseHttpEvent):
	type: Literal['RECORDING_STARTED'] = 'RECORDING_STARTED'
	payload: RecordingStatusPayload


class HttpRecordingStoppedEvent(BaseHttpEvent):
	type: Literal['RECORDING_STOPPED'] = 'RECORDING_STOPPED'
	payload: RecordingStatusPayload


# Union of all possible event types received by the recorder
RecorderEvent = Union[
	HttptoolUpdateEvent,
	HttpRecordingStartedEvent,
	HttpRecordingStoppedEvent,
]
