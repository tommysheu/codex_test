"""FastAPI backend for GPT-4o powered chatbot."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


class ChatResponse(BaseModel):
    """Response returned to the front-end."""

    reply: str
    used_search: bool
    search_results: List[SearchResult] = Field(default_factory=list)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page front-end application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Front-end not found.")
    return FileResponse(index_file)


def _perform_search(query: str, max_results: int = 3) -> List[SearchResult]:
    """Query DuckDuckGo for lightweight search snippets."""

    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure path
        logger.warning("Search request failed: %s", exc)
        return []

    payload = response.json()
    related_topics = payload.get("RelatedTopics", [])
    results: List[SearchResult] = []
    for topic in related_topics:
        if "Text" in topic and "FirstURL" in topic:
            results.append(
                SearchResult(
                    title=topic.get("Text", ""),
                    snippet=topic.get("Text", ""),
                    url=topic.get("FirstURL", ""),
                )
            )
        elif "Topics" in topic:
            for subtopic in topic.get("Topics", []):
                if "Text" in subtopic and "FirstURL" in subtopic:
                    results.append(
                        SearchResult(
                            title=subtopic.get("Text", ""),
                            snippet=subtopic.get("Text", ""),
                            url=subtopic.get("FirstURL", ""),
                        )
                    )
        if len(results) >= max_results:
            break
    return results[:max_results]


def _build_messages(request: ChatRequest, search_results: Iterable[SearchResult]) -> List[dict]:
    """Compose the message array sent to GPT-4o."""

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(message.dict() for message in request.history)
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


def _run_llm(messages: List[dict]) -> str:
    """Send messages to GPT-4o and extract the assistant reply."""

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )
    try:
        response = client.responses.create(model="gpt-4o", messages=messages)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    collected_parts: List[str] = []
    for item in getattr(response, "output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    collected_parts.append(content.get("text", ""))
    if not collected_parts:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")
    return "".join(collected_parts)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Primary chat endpoint used by the front-end."""

    search_results: List[SearchResult] = []
    if request.mode == "search":
        search_results = _perform_search(request.message)

    messages = _build_messages(request, search_results)
    reply = _run_llm(messages)
    return ChatResponse(reply=reply, used_search=bool(search_results), search_results=search_results)


__all__ = ["app"]
