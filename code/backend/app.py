"""FastAPI app for the Agentic Recipe RAG interface."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import RecipeAgent
from .rag_tool import RecipeRAGTool
from .schemas import ChatMessage, ChatRequest, ChatResponse, new_session_id


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Agentic Recipe RAG", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

rag_tool = RecipeRAGTool()
agent = RecipeAgent(rag_tool)
sessions: dict[str, list[ChatMessage]] = {}

MAX_SESSION_MESSAGES = 50
SUMMARY_COMPACT_COUNT = 40


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/flow")
def data_flow() -> dict:
    return {
        "name": "Agentic RAG data flow",
        "steps": [
            "Browser UI submits message to POST /api/chat",
            "FastAPI loads the session history",
            "FastAPI compacts long session history into a summary when needed",
            "RecipeAgent applies tool-use rules and decides whether recipe_rag_search is needed",
            "recipe_rag_search calls RecipeRAGSystem.ask_with_trace()",
            "Existing RAG runs router, rewrite, hybrid retrieval, parent document expansion, and answer generation",
            "RecipeRAGTool keeps full RAG trace aside while exposing concise context to the Agent",
            "RecipeAgent returns the final assistant answer immediately after one RAG call",
            "FastAPI stores the turn in memory and returns answer plus trace to the UI",
            "POST /api/chat/stream streams token, rag_step, agent_step, final, and done events over SSE",
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = request.session_id or new_session_id()
    history = sessions.setdefault(session_id, [])
    _compact_history_if_needed(history)

    response = agent.answer(
        message=message,
        history=history,
        session_id=session_id,
        include_trace=request.include_trace,
    )

    history.append(ChatMessage(role="user", content=message))
    history.append(
        ChatMessage(
            role="assistant",
            content=response.answer,
            rag_trace=_model_to_dict(response.tool_trace) if response.tool_trace else None,
        )
    )
    return response


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = request.session_id or new_session_id()
    history = sessions.setdefault(session_id, [])
    _compact_history_if_needed(history)

    async def event_stream():
        final_response: dict | None = None
        async for event in agent.astream_events(
            message=message,
            history=history,
            session_id=session_id,
            include_trace=request.include_trace,
        ):
            if event.get("type") == "final":
                final_response = event.get("response") or {}
                history.append(ChatMessage(role="user", content=message))
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=str(final_response.get("answer") or ""),
                        rag_trace=final_response.get("tool_trace"),
                    )
                )
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _compact_history_if_needed(history: list[ChatMessage]) -> None:
    if len(history) <= MAX_SESSION_MESSAGES:
        return

    old_messages = history[:SUMMARY_COMPACT_COUNT]
    summary = agent.summarize_history(old_messages)
    remaining = history[SUMMARY_COMPACT_COUNT:]
    history[:] = [
        ChatMessage(
            role="system",
            content=f"之前的对话摘要：\n{summary}",
        )
    ] + remaining
