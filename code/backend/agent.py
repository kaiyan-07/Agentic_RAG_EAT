"""Lightweight agent layer that decides when to call the RAG tool."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from .rag_tool import RecipeRAGTool
from .schemas import AgentStep, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    use_tool: bool
    tool_query: str
    direct_answer: str
    intent: str = "unknown"
    reason: str = ""


@dataclass
class TurnState:
    rag_calls: int = 0
    recursion_steps: int = 0


class RecipeAgent:
    """Agentic shell around the existing RAG pipeline.

    The agent owns conversation flow and tool use. The RAG tool owns retrieval
    and answer generation, so the current RAG internals stay unchanged.
    """

    system_prompt = """
你是一个食谱知识库 Agent。
工具规则：
- 当用户询问知识库中的菜谱、食材、步骤、时间、火候、技巧或推荐时，使用 recipe_rag_search。
- 每轮最多调用一次 recipe_rag_search。
- 拿到 recipe_rag_search 结果后，必须立即产出最终回答。
- 不要在收到 RAG 结果后再次调用工具。
- 如果检索上下文不足，明确说明不知道或没有找到，不要硬编。
""".strip()

    def __init__(
        self,
        rag_tool: RecipeRAGTool,
        max_rag_calls_per_turn: int = 1,
        recursion_limit: int = 8,
    ) -> None:
        self.rag_tool = rag_tool
        self.max_rag_calls_per_turn = max_rag_calls_per_turn
        self.recursion_limit = recursion_limit

    def answer(
        self,
        message: str,
        history: list[ChatMessage],
        session_id: str,
        include_trace: bool = True,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> ChatResponse:
        steps: list[AgentStep] = []
        turn_state = TurnState()
        self.rag_tool.clear_trace()

        decision = self._decide(message, history)
        turn_state.recursion_steps += 1
        steps.append(
            AgentStep(
                name="agent_decision",
                status="completed",
                detail=(
                    f"{decision.intent}: 调用 {self.rag_tool.name}: {decision.tool_query}"
                    if decision.use_tool
                    else f"{decision.intent}: 不调用工具，直接回答"
                ),
            )
        )
        self._emit(
            emit_event,
            {
                "type": "agent_step",
                "step": {
                    "name": "agent_decision",
                    "status": "completed",
                    "detail": steps[-1].detail,
                },
            },
        )

        if not decision.use_tool:
            answer = decision.direct_answer or "我主要负责食谱知识问答。你可以问我菜谱、食材、步骤、火候或推荐。"
            self._emit_text(emit_event, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                used_tool=False,
                tool_trace=None,
                agent_steps=steps,
            )

        tool_trace = None
        try:
            if turn_state.recursion_steps >= self.recursion_limit:
                raise RuntimeError("Agent recursion limit reached before tool call.")

            tool_result = self.rag_tool.run(
                decision.tool_query,
                calls_this_turn=turn_state.rag_calls,
                max_calls_per_turn=self.max_rag_calls_per_turn,
                emit_event=emit_event,
            )
            if tool_result.raw_trace.get("tool_used"):
                turn_state.rag_calls += 1
            rag_trace = tool_result.raw_trace
            tool_trace = self.rag_tool.to_tool_trace(decision.tool_query, rag_trace)
            turn_state.recursion_steps += 1
            steps.append(
                AgentStep(
                    name="tool_call",
                    status="completed",
                    detail=(
                        f"检索到 {tool_trace.retrieved_parent_count} 个父文档；"
                        f"耗时 {tool_trace.elapsed_ms} ms；"
                        f"本轮 RAG 调用 {turn_state.rag_calls}/{self.max_rag_calls_per_turn}"
                    ),
                )
            )
            self._emit(
                emit_event,
                {
                    "type": "agent_step",
                    "step": {
                        "name": "tool_call",
                        "status": "completed",
                        "detail": steps[-1].detail,
                    },
                },
            )
            answer = rag_trace.get("response") or "抱歉，食谱知识库没有返回可用答案。"
            if not tool_trace.hit:
                answer = "抱歉，我没有在当前食谱知识库里找到足够相关的信息。可以换一个更具体的菜名、食材或做法再问我。"
            self._emit_text(emit_event, answer)
            turn_state.recursion_steps += 1
            steps.append(
                AgentStep(
                    name="final_response",
                    status="completed",
                    detail="收到 RAG 结果后直接生成最终回答，未再次调用工具",
                )
            )
            self._emit(
                emit_event,
                {
                    "type": "agent_step",
                    "step": {
                        "name": "final_response",
                        "status": "completed",
                        "detail": steps[-1].detail,
                    },
                },
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
            self._emit_text(emit_event, answer)

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            used_tool=True,
            tool_trace=tool_trace if include_trace else None,
            agent_steps=steps,
        )

    async def astream_events(
        self,
        message: str,
        history: list[ChatMessage],
        session_id: str,
        include_trace: bool = True,
    ):
        """Stream agent events through one queue while the turn runs in the background."""
        output_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit_event(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(output_queue.put_nowait, event)

        async def run_turn() -> None:
            try:
                response = await asyncio.to_thread(
                    self.answer,
                    message,
                    history,
                    session_id,
                    include_trace,
                    emit_event,
                )
                await output_queue.put(
                    {
                        "type": "final",
                        "response": self._response_to_dict(response),
                    }
                )
            except Exception as exc:
                logger.exception("Streaming agent turn failed")
                await output_queue.put({"type": "error", "message": str(exc)})
            finally:
                await output_queue.put({"type": "done"})

        task = asyncio.create_task(run_turn())
        try:
            while True:
                event = await output_queue.get()
                yield event
                if event.get("type") == "done":
                    break
        finally:
            if not task.done():
                task.cancel()

    def _decide(self, message: str, history: list[ChatMessage]) -> AgentDecision:
        text = message.strip()
        llm_decision = self._llm_decide(text, history)
        if llm_decision:
            return llm_decision

        if self._is_recipe_related(text, history):
            return AgentDecision(
                use_tool=True,
                tool_query=text,
                direct_answer="",
                intent="recipe_search",
                reason="规则判断为食谱相关问题",
            )

        return AgentDecision(
            use_tool=False,
            tool_query="",
            direct_answer="你好，我是食谱 Agent。你可以问我菜谱推荐、食材用量、制作步骤、火候时间或烹饪技巧。",
            intent="small_talk",
            reason="规则判断为非食谱问题",
        )

    def _llm_decide(self, message: str, history: list[ChatMessage]) -> AgentDecision | None:
        prompt = f"""
{self.system_prompt}

请根据用户最新输入和最近对话，做一次结构化 Agent 决策。
只输出 JSON，不要输出 Markdown。

可选 intent：
- recipe_search：查询具体菜谱、食材、步骤、时间、火候、技巧
- recommendation：根据场景、食材、口味、时间推荐
- follow_up：依赖历史上下文的追问
- clarification：信息不足，需要先澄清
- small_talk：寒暄
- out_of_scope：非食谱知识库问题

输出格式：
{{
  "intent": "...",
  "use_tool": true,
  "tool_query": "适合检索的独立问题",
  "direct_answer": "不调用工具时给用户的回答",
  "reason": "一句话说明"
}}

最近历史：
{self._format_history(history)}

用户最新输入：
{message}
""".strip()

        try:
            raw = self.rag_tool.invoke_llm(prompt)
            data = self._parse_json_object(raw)
            if not data:
                return None

            intent = str(data.get("intent") or "unknown")
            use_tool = bool(data.get("use_tool"))
            if intent in {"small_talk", "out_of_scope", "clarification"}:
                use_tool = False
            elif intent in {"recipe_search", "recommendation", "follow_up"}:
                use_tool = True

            tool_query = str(data.get("tool_query") or message).strip()
            direct_answer = str(data.get("direct_answer") or "").strip()
            if not use_tool and not direct_answer:
                direct_answer = "我主要负责食谱知识问答。你可以问我菜谱、食材、步骤、火候或推荐。"

            return AgentDecision(
                use_tool=use_tool,
                tool_query=tool_query,
                direct_answer=direct_answer,
                intent=intent,
                reason=str(data.get("reason") or ""),
            )
        except Exception as exc:
            logger.warning("LLM agent decision failed; falling back to rules: %s", exc)
            return None

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start : end + 1])

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

    def summarize_history(self, messages: list[ChatMessage]) -> str:
        """Summarize older turns before they are compacted out of the prompt window."""
        if not messages:
            return ""

        text = "\n".join(
            f"{'用户' if item.role == 'user' else 'AI' if item.role == 'assistant' else '系统'}: {item.content}"
            for item in messages
        )
        prompt = f"""
请总结以下对话的关键信息，保留：
1. 用户偏好
2. 已确认事实
3. 项目背景
4. 仍未解决的问题
5. 后续待办

对话：
{text}

总结：
""".strip()

        try:
            return self.rag_tool.invoke_llm(prompt)
        except Exception:
            logger.exception("History summarization failed; falling back to extractive summary")
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: list[ChatMessage], limit: int = 12) -> str:
        recent = messages[-limit:]
        return "\n".join(f"- {item.role}: {item.content[:180]}" for item in recent)

    def _emit(self, emit_event: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
        if emit_event:
            emit_event(event)

    def _emit_text(self, emit_event: Callable[[dict[str, Any]], None] | None, text: str) -> None:
        if not emit_event:
            return
        for chunk in self._chunk_text(text):
            emit_event({"type": "token", "content": chunk})

    def _chunk_text(self, text: str, size: int = 12) -> list[str]:
        return [text[index : index + size] for index in range(0, len(text), size)] or [""]

    def _response_to_dict(self, response: ChatResponse) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return response.dict()
