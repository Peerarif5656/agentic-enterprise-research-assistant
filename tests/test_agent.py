"""
Test suite for the Enterprise RAG Agent.

Tests are structured in three layers:
  1. Unit tests  — individual components with no network/API calls
  2. Integration — component combinations (retrieval pipeline end-to-end)
  3. API tests   — FastAPI endpoints via httpx TestClient

Run:
    pytest -q                      # all tests (needs ANTHROPIC_API_KEY)
    pytest -q -m "not api"         # skip API tests (no key needed)
"""
from __future__ import annotations

import os
import tempfile
import pytest
from pathlib import Path

from src.types import AgentState, RouteDecision, Chunk, RetrievalResult
from src.config import Config, RetrieverConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_corpus(docs: dict[str, str]) -> str:
    """Write docs to a temp directory and return its path."""
    tmpdir = tempfile.mkdtemp()
    for name, text in docs.items():
        Path(tmpdir, name).write_text(text, encoding="utf-8")
    return tmpdir


# ---------------------------------------------------------------------------
# Unit tests — types and config
# ---------------------------------------------------------------------------

class TestTypes:
    def test_agent_state_defaults(self):
        state = AgentState(question="What is RAG?")
        assert state.question == "What is RAG?"
        assert state.route is None
        assert state.answer is None
        assert state.history == []

    def test_route_decision_values(self):
        assert RouteDecision.RETRIEVE.value == "retrieve"
        assert RouteDecision.SUMMARISE.value == "summarise"
        assert RouteDecision.CLARIFY.value == "clarify"

    def test_chunk_fields(self):
        c = Chunk("id1", "Some text", "doc.txt", 0.87, "hybrid_rrf")
        assert c.chunk_id == "id1"
        assert c.score == 0.87


class TestConfig:
    def test_load_defaults(self):
        cfg = Config.load("nonexistent_path.yaml")
        assert cfg.retriever.top_k == 5
        assert cfg.retriever.sparse_weight == 0.4
        assert cfg.retriever.dense_weight == 0.6
        assert cfg.generation.temperature == 0.0

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            Config.anthropic_key()


# ---------------------------------------------------------------------------
# Unit tests — retriever (no API calls)
# ---------------------------------------------------------------------------

class TestRetriever:
    @pytest.fixture
    def retriever_with_corpus(self):
        from src.retrieval.hybrid import HybridRetriever
        corpus = {
            "ai_basics.txt": (
                "Artificial intelligence is the simulation of human intelligence "
                "by machines. Machine learning is a subset of AI that enables "
                "systems to learn from data without being explicitly programmed."
            ),
            "rag_overview.txt": (
                "Retrieval-Augmented Generation combines document retrieval with "
                "language model generation. It grounds model outputs in retrieved "
                "evidence, reducing hallucination in enterprise question answering."
            ),
            "enterprise_qa.txt": (
                "Enterprise question answering systems must handle large document "
                "corpora reliably. Hybrid retrieval combining sparse BM25 and dense "
                "vector search outperforms either method alone on most benchmarks."
            ),
        }
        corpus_dir = make_temp_corpus(corpus)
        cfg = RetrieverConfig(top_k=3, chunk_size=200, chunk_overlap=20)
        r = HybridRetriever(cfg, corpus_dir=corpus_dir)
        r.build()
        return r

    def test_build_creates_chunks(self, retriever_with_corpus):
        assert len(retriever_with_corpus._chunks) > 0

    def test_retrieve_returns_correct_count(self, retriever_with_corpus):
        result = retriever_with_corpus.retrieve("What is retrieval augmented generation?")
        assert len(result.chunks) <= 3
        assert result.method_used == "hybrid_rrf"

    def test_retrieve_scores_are_positive(self, retriever_with_corpus):
        result = retriever_with_corpus.retrieve("machine learning AI")
        for chunk in result.chunks:
            assert chunk.score > 0

    def test_retrieve_raises_before_build(self):
        from src.retrieval.hybrid import HybridRetriever
        r = HybridRetriever(RetrieverConfig(), corpus_dir="/tmp")
        with pytest.raises(RuntimeError, match="build()"):
            r.retrieve("test query")

    def test_empty_corpus_raises(self):
        from src.retrieval.hybrid import HybridRetriever
        empty_dir = make_temp_corpus({})
        r = HybridRetriever(RetrieverConfig(), corpus_dir=empty_dir)
        with pytest.raises(FileNotFoundError):
            r.build()

    def test_relevant_doc_surfaces_for_rag_query(self, retriever_with_corpus):
        result = retriever_with_corpus.retrieve("hallucination in language models")
        sources = [c.source for c in result.chunks]
        # rag_overview.txt explicitly mentions hallucination
        assert any("rag" in s.lower() for s in sources)


# ---------------------------------------------------------------------------
# Unit tests — RRF fusion function
# ---------------------------------------------------------------------------

class TestRRF:
    def test_rrf_merges_lists(self):
        from src.retrieval.hybrid import _rrf
        sparse = [(0, 1.0), (1, 0.8), (2, 0.5)]
        dense = [(1, 0.9), (2, 0.7), (3, 0.6)]
        fused = _rrf(sparse, dense, 0.4, 0.6)
        indices = [idx for idx, _ in fused]
        # Index 1 appears in both lists so should score highest
        assert indices[0] == 1

    def test_rrf_scores_are_positive(self):
        from src.retrieval.hybrid import _rrf
        sparse = [(0, 1.0), (1, 0.5)]
        dense = [(0, 0.9), (2, 0.4)]
        fused = _rrf(sparse, dense, 0.5, 0.5)
        assert all(score > 0 for _, score in fused)


# ---------------------------------------------------------------------------
# API tests (marked separately — require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestAPI:
    @pytest.fixture
    def client(self):
        """
        Creates a FastAPI test client with a temp corpus.
        Requires ANTHROPIC_API_KEY in the environment.
        """
        import os
        from fastapi.testclient import TestClient
        from src.api.app import app

        corpus = {
            "sample.txt": (
                "The Enterprise RAG Agent uses hybrid retrieval combining "
                "BM25 and FAISS to answer questions from internal documents. "
                "It is built with LangGraph, FastAPI, and the Claude API."
            )
        }
        corpus_dir = make_temp_corpus(corpus)
        os.environ["CORPUS_DIR"] = corpus_dir
        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["corpus_chunks"] > 0

    def test_corpus_stats_endpoint(self, client):
        resp = client.get("/corpus/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] >= 1

    def test_query_endpoint_returns_answer(self, client):
        resp = client.post(
            "/query",
            json={"question": "What retrieval methods does the system use?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "route_taken" in data
        assert data["confidence"] >= 0.0

    def test_query_too_short_rejected(self, client):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422  # Pydantic validation


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not api"])
