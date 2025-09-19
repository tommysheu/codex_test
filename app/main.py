"""FastAPI backend for GPT-4o powered chatbot."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, List, Literal, Optional

from duckduckgo_search import DDGS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

API_TITLE = "GPT-4o Chatbot"
API_DESCRIPTION = (
    "Backend service that proxies chat requests to GPT-4o and optionally "
    "enriches prompts with web search snippets."
)

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

SYSTEM_PROMPT = (
    "You are a helpful bilingual assistant that can answer in Traditional "
    "Chinese and English. When web search context is provided, cite it "
    "succinctly in the reply."
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI]
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.warning("OPENAI_API_KEY not found. LLM requests will fail until it is set.")


class SearchResult(BaseModel):
    """Represents a single web search snippet."""

    title: str
    snippet: str
    url: str


class ChatMessage(BaseModel):
    """Chat message exchanged with the client."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Incoming chat payload from the front-end."""

    message: str = Field(..., description="The end-user prompt.")
    mode: Literal["llm_only", "search"] = Field(
        "llm_only", description="Choose between pure LLM or search-assisted mode."
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Conversation history to maintain context across turns.",
    )

    @validator("history", each_item=True)
    def _strip_empty_history(cls, msg: ChatMessage) -> ChatMessage:
        if not msg.content:
            raise ValueError("History messages cannot be empty.")
        return msg
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page front-end application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Front-end not found.")
    return FileResponse(index_file)


_ddgs_client = DDGS(timeout=10)
_ddgs_lock = Lock()


def _perform_search(query: str, max_results: int = 3) -> List[SearchResult]:
    """Query DuckDuckGo for lightweight search snippets."""

    query = query.strip()
    if not query:
        return []

    def _fetch(region: str) -> List[SearchResult]:
        with _ddgs_lock:
            hits = list(
                _ddgs_client.text(
                    query,
                    region=region,
                    safesearch="moderate",
                    max_results=max_results,
                )
            )
        results: List[SearchResult] = []
        for item in hits:
            url = item.get("href", "").strip()
            if not url:
                continue
            title = item.get("title") or item.get("body") or url
            snippet = item.get("body", "")
            results.append(
                SearchResult(title=title.strip(), snippet=snippet.strip(), url=url)
            )
            if len(results) >= max_results:
                break
        return results

    try:
        results = _fetch("tw-tzh")
        if not results:
            results = _fetch("wt-wt")
        return results
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("Search request failed: %s", exc)
        return []


def _build_messages(request: ChatRequest, search_results: Iterable[SearchResult]) -> List[dict]:
    """Compose the message array sent to GPT-4o."""

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(message.model_dump() for message in request.history)
    context_lines = []
    for result in search_results:
        context_lines.append(f"Title: {result.title}\nURL: {result.url}\nSnippet: {result.snippet}")
    if context_lines:
        context_blob = "\n\n".join(context_lines)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Web search results were found for the current question. "
                    "Incorporate the following evidence when helpful and cite the source URLs:\n"
                    f"{context_blob}"
                ),
            }
        )
    messages.append({"role": "user", "content": request.message})
    return messages


def _stream_llm(messages: List[dict]) -> Iterator[tuple[str, dict]]:
    """Stream chunks from GPT-4o and yield structured events."""

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    collected_parts: List[str] = []
    final_response = None

    try:
        with client.responses.stream(
            model="gpt-4o",
            input=[
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in messages
            ],
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    text = event.delta or ""
                    if text:
                        collected_parts.append(text)
                        yield "delta", {"text": text}
                elif event.type == "response.refusal.delta":
                    text = event.delta or ""
                    if text:
                        collected_parts.append(text)
                        yield "delta", {"text": text}
            final_response = stream.get_final_response()
    except Exception as exc:  # pragma: no cover - network failure path
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    full_text = "".join(collected_parts)
    if not full_text and final_response is not None:
        if hasattr(final_response, "output_text") and final_response.output_text:
            full_text = final_response.output_text
        else:
            for item in getattr(final_response, "output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            full_text += content.get("text", "")

    if not full_text:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")

    yield "done", {"text": full_text}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Primary chat endpoint used by the front-end with streaming replies."""

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    search_results: List[SearchResult] = []
    if request.mode == "search":
        search_results = _perform_search(request.message)

    messages = _build_messages(request, search_results)

    def _event_stream() -> Iterator[str]:
        metadata = {
            "used_search": bool(search_results),
            "search_results": [result.model_dump() for result in search_results],
        }
        yield _format_sse_event("meta", metadata)
        try:
            for event_type, payload in _stream_llm(messages):
                yield _format_sse_event(event_type, payload)
        except HTTPException as exc:
            logger.exception("Streaming failed with HTTP error")
            yield _format_sse_event("error", {"message": exc.detail})
            return
        except Exception as exc:  # pragma: no cover - network failure path
            logger.exception("Streaming failed")
            yield _format_sse_event("error", {"message": str(exc)})
            return

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def _format_sse_event(event_type: str, payload: dict) -> str:
    """Encode an SSE event as a string."""

    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


__all__ = ["app"]
