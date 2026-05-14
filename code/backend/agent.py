"""Lightweight agent layer that decides when to call the RAG tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .rag_tool import RecipeRAGTool
from .schemas import AgentStep, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    use_tool: bool
    tool_query: str
    direct_answer: str


class RecipeAgent:
    """Agentic shell around the existing RAG pipeline.

    The agent owns conversation flow and tool use. The RAG tool owns retrieval
    and answer generation, so the current RAG internals stay unchanged.
    """

    def __init__(self, rag_tool: RecipeRAGTool) -> None:
        self.rag_tool = rag_tool

    def answer(
        self,
        message: str,
        history: list[ChatMessage],
        session_id: str,
        include_trace: bool = True,
    ) -> ChatResponse:
        steps: list[AgentStep] = []

        decision = self._decide(message, history)
        steps.append(
            AgentStep(
                name="agent_decision",
                status="completed",
                detail=(
                    f"调用 {self.rag_tool.name}: {decision.tool_query}"
                    if decision.use_tool
                    else "不调用工具，直接回答"
                ),
            )
        )

        if not decision.use_tool:
            answer = decision.direct_answer or "我主要负责食谱知识问答。你可以问我菜谱、食材、步骤、火候或推荐。"
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                used_tool=False,
                tool_trace=None,
                agent_steps=steps,
            )

        tool_trace = None
        try:
            rag_trace = self.rag_tool.run(decision.tool_query)
            tool_trace = self.rag_tool.to_tool_trace(decision.tool_query, rag_trace)
            steps.append(
                AgentStep(
                    name="tool_call",
                    status="completed",
                    detail=f"检索到 {tool_trace.retrieved_parent_count} 个父文档",
                )
            )
            answer = rag_trace.get("response") or "抱歉，食谱知识库没有返回可用答案。"
            steps.append(
                AgentStep(
                    name="final_response",
                    status="completed",
                    detail="使用 RAG 工具答案作为最终回答",
                )
            )
        except Exception as exc:
            logger.exception("RAG tool failed")
            steps.append(
                AgentStep(
                    name="tool_call",
                    status="failed",
                    detail=str(exc),
                )
            )
            answer = "抱歉，食谱知识库暂时没有成功返回结果。请稍后再试，或换一个更具体的菜名/问题。"

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            used_tool=True,
            tool_trace=tool_trace if include_trace else None,
            agent_steps=steps,
        )

    def _decide(self, message: str, history: list[ChatMessage]) -> AgentDecision:
        text = message.strip()
        if self._is_recipe_related(text, history):
            return AgentDecision(use_tool=True, tool_query=text, direct_answer="")

        return AgentDecision(
            use_tool=False,
            tool_query="",
            direct_answer="你好，我是食谱 Agent。你可以问我菜谱推荐、食材用量、制作步骤、火候时间或烹饪技巧。",
        )

    def _is_recipe_related(self, message: str, history: list[ChatMessage]) -> bool:
        recipe_keywords = [
            "菜",
            "食谱",
            "做法",
            "怎么做",
            "步骤",
            "食材",
            "原料",
            "调料",
            "用量",
            "火候",
            "温度",
            "多久",
            "几分钟",
            "推荐",
            "吃什么",
            "早餐",
            "午餐",
            "晚餐",
            "饮品",
            "甜品",
            "汤",
            "主食",
            "空气炸锅",
            "蒸",
            "煮",
            "炒",
            "煎",
            "烤",
        ]
        if any(keyword in message for keyword in recipe_keywords):
            return True

        follow_up_keywords = ["它", "这个", "刚才", "那道", "多少", "还要", "可以换", "注意"]
        recent_assistant = " ".join(item.content for item in history[-4:] if item.role == "assistant")
        return bool(recent_assistant and any(keyword in message for keyword in follow_up_keywords))

    def _format_history(self, history: list[ChatMessage], limit: int = 8) -> str:
        if not history:
            return "无"
        recent = history[-limit:]
        return "\n".join(f"{item.role}: {item.content}" for item in recent)
