from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    ITERATION = "iteration"


class AgentEvent(BaseModel):
    event_type: EventType
    iteration: int = 0
    content: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""


class ToolCallEvent(BaseModel):
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class AgentResult(BaseModel):
    success: bool
    answer: str = ""
    events: List[AgentEvent] = Field(default_factory=list)
    tool_calls: List[ToolCallEvent] = Field(default_factory=list)
    iterations: int = 0
    error: str = ""
