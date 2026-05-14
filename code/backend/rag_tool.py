"""Tool wrapper around the existing RecipeRAGSystem.

This module intentionally treats the current RAG pipeline as a black box.
The agent calls this tool; the tool calls RecipeRAGSystem.ask_with_trace().
"""

from __future__ import annotations

import threading
from typing import Any

from .schemas import ToolTrace


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

    def run(self, query: str) -> dict[str, Any]:
        rag = self._get_rag()
        return rag.ask_with_trace(query, stream=False, include_raw_documents=False)

    def to_tool_trace(self, query: str, trace: dict[str, Any]) -> ToolTrace:
        return ToolTrace(
            tool_name=self.name,
            tool_query=query,
            route_type=trace.get("route_type"),
            rewritten_query=trace.get("rewritten_query"),
            filters=trace.get("filters") or {},
            retrieved_chunk_count=trace.get("retrieved_chunk_count") or 0,
            retrieved_parent_count=trace.get("retrieved_parent_count") or 0,
            retrieved_dish_names=trace.get("retrieved_parent_dish_names") or [],
            retrieved_sources=trace.get("retrieved_parent_sources") or [],
        )
