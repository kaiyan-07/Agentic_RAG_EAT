"""LangGraph agent layer for the Agentic Recipe RAG backend."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from .rag_tool import RecipeRAGTool
from .schemas import AgentStep, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    rag_calls: int = 0


class RecipeAgent:
    """LangGraph ReAct-style agent around the existing RAG tool."""

    system_prompt = """
你是一个食谱知识库 Agent。
工具规则：
- 当用户询问知识库中的菜谱、食材、步骤、时间、火候、技巧或推荐时，使用 recipe_rag_search。
- 每轮最多调用一次 recipe_rag_search。
- 拿到 recipe_rag_search 结果后，必须立即产出最终回答。
- 不要在收到 RAG 结果后再次调用工具。
- 如果检索上下文不足，明确说明不知道或没有找到，不要硬编。
- 如果用户只是寒暄，直接简短回应，并提示可以询问菜谱。
- 如果问题明显超出食谱知识库范围，不要调用工具，说明你主要负责食谱知识问答。
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
        """Synchronous compatibility wrapper around the LangGraph stream."""
        events: list[dict[str, Any]] = []

        async def collect() -> ChatResponse:
            final_response = None
            async for event in self.astream_events(
                message=message,
                history=history,
                session_id=session_id,
                include_trace=include_trace,
                emit_event=emit_event,
            ):
                events.append(event)
                if event.get("type") == "final":
                    final_response = event.get("response")
            if not final_response:
                raise RuntimeError("LangGraph agent did not produce a final response.")
            return self._response_from_dict(final_response)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(collect())

        raise RuntimeError("RecipeAgent.answer() cannot be called from an active event loop.")

    async def astream_events(
        self,
        message: str,
        history: list[ChatMessage],
        session_id: str,
        include_trace: bool = True,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Run LangGraph in a background task and yield events from one queue."""
        output_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def enqueue(event: dict[str, Any]) -> None:
            if emit_event:
                emit_event(event)
            loop.call_soon_threadsafe(output_queue.put_nowait, event)

        async def run_turn() -> None:
            try:
                response = await self._run_langgraph_turn(
                    message=message,
                    history=history,
                    session_id=session_id,
                    include_trace=include_trace,
                    emit_event=enqueue,
                )
                await output_queue.put(
                    {
                        "type": "final",
                        "response": self._response_to_dict(response),
                    }
                )
            except Exception as exc:
                logger.exception("LangGraph agent turn failed")
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

    async def _run_langgraph_turn(
        self,
        message: str,
        history: list[ChatMessage],
        session_id: str,
        include_trace: bool,
        emit_event: Callable[[dict[str, Any]], None],
    ) -> ChatResponse:
        self.rag_tool.clear_trace()
        turn_state = TurnState()
        steps: list[AgentStep] = []
        final_text_parts: list[str] = []

        agent = self._build_langgraph_agent(turn_state, emit_event)
        messages = self._to_langchain_messages(history, message)

        steps.append(
            AgentStep(
                name="langgraph_agent",
                status="completed",
                detail="LangGraph ReAct agent started",
            )
        )
        emit_event(
            {
                "type": "agent_step",
                "step": {
                    "name": "langgraph_agent",
                    "status": "completed",
                    "detail": "LangGraph ReAct agent started",
                },
            }
        )

        async for chunk, metadata in agent.astream(
            {"messages": messages},
            config={"recursion_limit": self.recursion_limit},
            stream_mode="messages",
        ):
            if getattr(chunk, "tool_call_chunks", None):
                continue

            content = getattr(chunk, "content", "")
            if not content:
                continue

            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content)
            if not content:
                continue

            final_text_parts.append(content)
            emit_event({"type": "token", "content": content})

        answer = "".join(final_text_parts).strip()
        tool_trace = None
        raw_trace = self.rag_tool.last_trace()
        if raw_trace:
            query = raw_trace.get("tool_query") or message
            tool_trace = self.rag_tool.to_tool_trace(str(query), raw_trace)
            steps.append(
                AgentStep(
                    name="tool_call",
                    status="completed" if tool_trace.tool_used else "skipped",
                    detail=(
                        f"检索到 {tool_trace.retrieved_parent_count} 个父文档；"
                        f"耗时 {tool_trace.elapsed_ms} ms；"
                        f"本轮 RAG 调用 {turn_state.rag_calls}/{self.max_rag_calls_per_turn}"
                    ),
                )
            )

        if not answer:
            answer = "抱歉，我暂时没有生成可用回答。请换一个更具体的菜名、食材或做法再问我。"

        steps.append(
            AgentStep(
                name="final_response",
                status="completed",
                detail="LangGraph agent produced the final response",
            )
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            used_tool=bool(tool_trace and tool_trace.tool_used),
            tool_trace=tool_trace if include_trace else None,
            agent_steps=steps,
        )

    def _build_langgraph_agent(
        self,
        turn_state: TurnState,
        emit_event: Callable[[dict[str, Any]], None],
    ):
        try:
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph Agent requires langgraph and langchain-core. "
                "Run `python -m pip install -r requirements.txt` from the code directory."
            ) from exc

        @tool(self.rag_tool.name)
        def recipe_rag_search(query: str) -> str:
            """Search the local recipe knowledge base."""
            tool_result = self.rag_tool.run(
                query,
                calls_this_turn=turn_state.rag_calls,
                max_calls_per_turn=self.max_rag_calls_per_turn,
                emit_event=emit_event,
            )
            if tool_result.raw_trace.get("tool_used"):
                turn_state.rag_calls += 1
            return tool_result.agent_context

        llm = self.rag_tool.get_llm()
        return self._create_react_agent(create_react_agent, llm, [recipe_rag_search])

    def _create_react_agent(self, create_react_agent, llm, tools):
        try:
            return create_react_agent(llm, tools, prompt=self.system_prompt)
        except TypeError:
            return create_react_agent(llm, tools, state_modifier=self.system_prompt)

    def _to_langchain_messages(self, history: list[ChatMessage], message: str):
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph Agent requires langchain-core message classes. "
                "Run `python -m pip install -r requirements.txt` from the code directory."
            ) from exc

        converted = []
        for item in history[-12:]:
            if item.role == "user":
                converted.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                converted.append(AIMessage(content=item.content))
            elif item.role == "system":
                converted.append(SystemMessage(content=item.content))
        converted.append(HumanMessage(content=message))
        return converted

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

    def _response_to_dict(self, response: ChatResponse) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return response.dict()

    def _response_from_dict(self, data: dict[str, Any]) -> ChatResponse:
        return ChatResponse(**data)
