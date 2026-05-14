# Agentic RAG Backend

This backend keeps the existing RAG pipeline unchanged and wraps it as an
agent tool.

## Data Flow

1. Browser sends `POST /api/chat`.
2. FastAPI loads in-memory session history.
3. `RecipeAgent` decides whether the user message should call
   `recipe_rag_search`.
4. `RecipeRAGTool` calls the existing `RecipeRAGSystem.ask_with_trace()`.
5. The existing RAG pipeline runs query routing, query rewrite, hybrid
   retrieval, parent document expansion, and answer generation.
6. FastAPI returns the answer, agent steps, and optional RAG trace to the UI.

## Run

From the `code` directory:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

Useful endpoints:

- `GET /api/health`
- `GET /api/flow`
- `POST /api/chat`

## Boundary

The RAG internals are intentionally not changed in this phase. The only
integration point is:

```python
RecipeRAGSystem.ask_with_trace(question, stream=False, include_raw_documents=False)
```
