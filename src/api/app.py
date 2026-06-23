"""
FastAPI REST API for the Enterprise RAG Agent.

Endpoints:
  POST /query        — Run the full agent pipeline on a question
  GET  /health       — Liveness check
  GET  /corpus/stats — Corpus index statistics
  POST /corpus/reload — Hot-reload the corpus without restarting the server

The app is a thin layer over the LangGraph agent. It handles:
  - Request/response validation (Pydantic models)
  - Shared state (retriever, generator, compiled graph) via FastAPI lifespan
  - Structured JSON error responses
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import Config
from ..types import AgentState
from ..retrieval.hybrid import HybridRetriever
from ..generation.generator import AnswerGenerator
from ..agent.graph import build_graph


# ---------------------------------------------------------------------------
# Pydantic models — request / response contracts
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks to retrieve")


class ChunkOut(BaseModel):
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    route_taken: str
    retrieval_method: str
    confidence: float
    latency_ms: float
    chunks: Optional[List[ChunkOut]] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    corpus_chunks: int


class CorpusStats(BaseModel):
    total_chunks: int
    total_documents: int
    corpus_dir: str


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

class AppState:
    retriever: HybridRetriever
    generator: AnswerGenerator
    graph: object   # compiled LangGraph
    cfg: Config


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan: build indices once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config.load()
    api_key = Config.anthropic_key()

    app_state.cfg = cfg
    app_state.retriever = HybridRetriever(cfg.retriever, corpus_dir=cfg.corpus_dir)
    app_state.retriever.build()
    app_state.generator = AnswerGenerator(api_key=api_key, cfg=cfg.generation)
    app_state.graph = build_graph(
        app_state.retriever, app_state.generator, api_key, cfg
    )
    yield
    # Cleanup (nothing to tear down here — no persistent connections)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Agent",
    description=(
        "Agentic enterprise question-answering system using LangGraph, "
        "Claude, hybrid BM25+FAISS retrieval, and MCP tool protocol."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model=app_state.cfg.generation.model,
        corpus_chunks=len(app_state.retriever._chunks),
    )


@app.get("/corpus/stats", response_model=CorpusStats)
async def corpus_stats():
    chunks = app_state.retriever._chunks
    sources = {c.source for c in chunks}
    return CorpusStats(
        total_chunks=len(chunks),
        total_documents=len(sources),
        corpus_dir=str(app_state.retriever.corpus_dir),
    )


@app.post("/corpus/reload")
async def corpus_reload():
    """Hot-reload the corpus — useful when new documents are added."""
    app_state.retriever.build()
    return {"status": "reloaded", "chunks": len(app_state.retriever._chunks)}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.perf_counter()

    initial_state = AgentState(question=req.question)
    try:
        final_state: AgentState = app_state.graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if final_state.error:
        raise HTTPException(status_code=500, detail=final_state.error)

    ans = final_state.answer
    if ans is None:
        raise HTTPException(status_code=500, detail="Agent produced no answer.")

    latency = round((time.perf_counter() - t0) * 1000, 1)

    chunks_out = None
    if final_state.retrieval_result:
        chunks_out = [
            ChunkOut(source=c.source, score=c.score, text=c.text[:300])
            for c in final_state.retrieval_result.chunks
        ]

    return QueryResponse(
        question=ans.question,
        answer=ans.answer,
        citations=ans.citations,
        route_taken=ans.route_taken.value,
        retrieval_method=ans.retrieval_method,
        confidence=ans.confidence,
        latency_ms=latency,
        chunks=chunks_out,
    )
