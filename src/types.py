"""Shared types for the Enterprise RAG Agent."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RouteDecision(str, Enum):
    RETRIEVE = "retrieve"       # Answer from internal document corpus
    SUMMARISE = "summarise"     # Summarise a provided document/text
    CLARIFY = "clarify"         # Question too ambiguous to route


@dataclass
class Chunk:
    """A retrieved document chunk with metadata."""
    chunk_id: str
    text: str
    source: str
    score: float
    retrieval_method: str  # "sparse" | "dense" | "hybrid"


@dataclass
class RetrievalResult:
    """Output of the retrieval step."""
    query: str
    chunks: List[Chunk]
    method_used: str


@dataclass
class GeneratedAnswer:
    """Final answer with citations."""
    question: str
    answer: str
    citations: List[str]         # Source document names cited
    route_taken: RouteDecision
    retrieval_method: str
    confidence: float            # 0.0 – 1.0, estimated from chunk scores


@dataclass
class AgentState:
    """
    LangGraph state object — passed between every node in the graph.
    Each node reads what it needs and writes what it produces.
    """
    question: str = ""
    route: Optional[RouteDecision] = None
    retrieval_result: Optional[RetrievalResult] = None
    answer: Optional[GeneratedAnswer] = None
    error: Optional[str] = None
    # Conversation memory: list of (question, answer) pairs
    history: List[dict] = field(default_factory=list)
