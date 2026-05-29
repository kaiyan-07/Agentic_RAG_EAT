"""API and agent data models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


Role = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    role: Role
    content: str
    rag_trace: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    include_trace: bool = True


class ToolTrace(BaseModel):
    tool_name: str
    tool_query: str
    tool_used: bool = True
    hit: bool = False
    route_type: str | None = None
    rewritten_query: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_mode: str | None = None
    rerank_applied: bool = False
    elapsed_ms: float | None = None
    retrieved_chunk_count: int = 0
    retrieved_chunk_summaries: list[str] = Field(default_factory=list)
    retrieved_parent_count: int = 0
    retrieved_dish_names: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)


class AgentStep(BaseModel):
    name: str
    status: Literal["completed", "skipped", "failed"]
    detail: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    used_tool: bool
    tool_trace: ToolTrace | None = None
    agent_steps: list[AgentStep] = Field(default_factory=list)


def new_session_id() -> str:
    return uuid4().hex
