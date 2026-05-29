"""Tool wrapper around the existing RecipeRAGSystem.

This module intentionally treats the current RAG pipeline as a black box.
The agent calls this tool; the tool calls RecipeRAGSystem.ask_with_trace().
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .schemas import ToolTrace


TOOL_CALL_LIMIT_MESSAGE = "TOOL_CALL_LIMIT_REACHED: use existing retrieval result and answer directly."


@dataclass
class ToolRunResult:
    """RAG output split into concise agent context and full side-channel trace."""

    agent_context: str
    raw_trace: dict[str, Any]


class RecipeRAGTool:
    """Lazy, singleton-style wrapper for the existing recipe RAG."""

    name = "recipe_rag_search"
    description = (
        "Search the local recipe knowledge base for recipe recommendations, "
        "ingredients, steps, timing, temperature, cooking tips, and dish facts."
    )

    def __init__(self) -> None:
        self._rag: RecipeRAGSystem | None = None
        self._lock = threading.Lock()
        self._trace_state = threading.local()

    def _get_rag(self) -> RecipeRAGSystem:
        if self._rag is None:
            with self._lock:
                if self._rag is None:
                    from main import RecipeRAGSystem

                    rag = RecipeRAGSystem()
                    rag.initialize_system()
                    rag.build_knowledge_base()
                    self._rag = rag
        return self._rag

    def clear_trace(self) -> None:
        self._trace_state.last_rag_trace = None

    def last_trace(self) -> dict[str, Any] | None:
        return getattr(self._trace_state, "last_rag_trace", None)

    def invoke_llm(self, prompt: str) -> str:
        response = self.get_llm().invoke(prompt)
        return getattr(response, "content", str(response)).strip()

    def get_llm(self):
        rag = self._get_rag()
        return rag.generation_module.llm

    def run(
        self,
        query: str,
        calls_this_turn: int = 0,
        max_calls_per_turn: int = 1,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> ToolRunResult:
        if calls_this_turn >= max_calls_per_turn:
            self._emit_rag_step(
                emit_event,
                "⛔",
                "已达到工具调用上限",
                "本轮不再调用 RAG，使用已有结果或直接回答。",
            )
            trace = {
                "tool_name": self.name,
                "tool_query": query,
                "tool_used": False,
                "limit_reached": True,
                "response": TOOL_CALL_LIMIT_MESSAGE,
                "retrieved_chunk_count": 0,
                "retrieved_parent_count": 0,
                "retrieved_parent_dish_names": [],
                "retrieved_parent_sources": [],
                "retrieved_docs": [],
            }
            self._trace_state.last_rag_trace = trace
            return ToolRunResult(agent_context=TOOL_CALL_LIMIT_MESSAGE, raw_trace=trace)

        rag = self._get_rag()
        self._emit_rag_step(emit_event, "🔎", "正在检索知识库...", query)
        start = time.perf_counter()
        trace = rag.ask_with_trace(query, stream=False, include_raw_documents=False)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        trace = self._with_side_channel_metadata(query, trace, elapsed_ms)
        self._emit_rag_step(
            emit_event,
            "📚",
            "正在整理检索结果...",
            f"命中 {trace.get('retrieved_parent_count') or 0} 个父文档，耗时 {elapsed_ms} ms。",
        )
        self._trace_state.last_rag_trace = trace
        return ToolRunResult(agent_context=self._format_docs_for_agent(trace), raw_trace=trace)

    def _with_side_channel_metadata(
        self,
        query: str,
        trace: dict[str, Any],
        elapsed_ms: float,
    ) -> dict[str, Any]:
        enriched = dict(trace)
        filters = enriched.get("filters") or {}
        chunk_sources = enriched.get("retrieved_chunk_sources") or []
        chunk_dish_names = enriched.get("retrieved_chunk_dish_names") or []
        chunk_summaries = enriched.get("retrieved_chunk_summaries") or []
        parent_sources = enriched.get("retrieved_parent_sources") or []
        parent_dish_names = enriched.get("retrieved_parent_dish_names") or []

        retrieved_docs = []
        max_len = max(len(parent_sources), len(parent_dish_names), len(chunk_sources), len(chunk_dish_names))
        for index in range(max_len):
            source = parent_sources[index] if index < len(parent_sources) else (
                chunk_sources[index] if index < len(chunk_sources) else ""
            )
            dish_name = parent_dish_names[index] if index < len(parent_dish_names) else (
                chunk_dish_names[index] if index < len(chunk_dish_names) else ""
            )
            retrieved_docs.append(
                {
                    "rank": index + 1,
                    "source": source,
                    "dish_name": dish_name,
                    "score": None,
                    "parent_id": source,
                    "child_summary": chunk_summaries[index] if index < len(chunk_summaries) else "",
                }
            )

        enriched.update(
            {
                "tool_name": self.name,
                "tool_query": query,
                "tool_used": True,
                "hit": (enriched.get("retrieved_parent_count") or 0) > 0,
                "retrieval_mode": "metadata_filtered_hybrid" if filters else "hybrid",
                "rerank_applied": True,
                "elapsed_ms": elapsed_ms,
                "retrieved_docs": retrieved_docs,
            }
        )
        return enriched

    def _format_docs_for_agent(self, trace: dict[str, Any]) -> str:
        if not trace.get("hit"):
            return "No relevant documents found."

        dish_names = trace.get("retrieved_parent_dish_names") or []
        chunk_summaries = trace.get("retrieved_chunk_summaries") or []
        response = trace.get("response") or ""
        lines = [
            "RAG_RESULT:",
            f"- hit: yes",
            f"- dishes: {', '.join(dish_names) if dish_names else 'unknown'}",
        ]
        if chunk_summaries:
            lines.append(f"- matched_sections: {', '.join(chunk_summaries[:5])}")
        if response:
            lines.append(f"- answer: {response}")
        return "\n".join(lines)

    def to_tool_trace(self, query: str, trace: dict[str, Any]) -> ToolTrace:
        return ToolTrace(
            tool_name=self.name,
            tool_query=query,
            tool_used=trace.get("tool_used", True),
            hit=trace.get("hit", False),
            route_type=trace.get("route_type"),
            rewritten_query=trace.get("rewritten_query"),
            filters=trace.get("filters") or {},
            retrieval_mode=trace.get("retrieval_mode"),
            rerank_applied=trace.get("rerank_applied", False),
            elapsed_ms=trace.get("elapsed_ms"),
            retrieved_chunk_count=trace.get("retrieved_chunk_count") or 0,
            retrieved_chunk_summaries=trace.get("retrieved_chunk_summaries") or [],
            retrieved_parent_count=trace.get("retrieved_parent_count") or 0,
            retrieved_dish_names=trace.get("retrieved_parent_dish_names") or [],
            retrieved_sources=trace.get("retrieved_parent_sources") or [],
            retrieved_docs=trace.get("retrieved_docs") or [],
        )

    def _emit_rag_step(
        self,
        emit_event: Callable[[dict[str, Any]], None] | None,
        icon: str,
        label: str,
        detail: str,
    ) -> None:
        if not emit_event:
            return
        emit_event(
            {
                "type": "rag_step",
                "step": {
                    "icon": icon,
                    "label": label,
                    "detail": detail,
                },
            }
        )
