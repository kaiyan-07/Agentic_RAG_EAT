"""FastAPI app for the Agentic Recipe RAG interface."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
            "RecipeAgent asks the LLM whether recipe_rag_search is needed",
            "recipe_rag_search calls RecipeRAGSystem.ask_with_trace()",
            "Existing RAG runs router, rewrite, hybrid retrieval, parent document expansion, and answer generation",
            "RecipeAgent composes the final assistant answer with tool trace",
            "FastAPI stores the turn in memory and returns answer plus trace to the UI",
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = request.session_id or new_session_id()
    history = sessions.setdefault(session_id, [])

    response = agent.answer(
        message=message,
        history=history,
        session_id=session_id,
        include_trace=request.include_trace,
    )

    history.append(ChatMessage(role="user", content=message))
    history.append(ChatMessage(role="assistant", content=response.answer))
    return response
