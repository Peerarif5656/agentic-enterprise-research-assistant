"""
Hybrid retriever combining BM25 (sparse) and FAISS (dense) with
Reciprocal Rank Fusion — directly mirrors the dissertation architecture.

Design rationale:
- BM25 captures exact keyword matches (strong for entity/numeric queries)
- Dense embeddings capture semantic similarity (strong for conceptual queries)
- RRF fuses both ranked lists without requiring score normalisation
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..types import Chunk, RetrievalResult
from ..config import RetrieverConfig


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _rrf(
    sparse_ranked: List[Tuple[int, float]],
    dense_ranked: List[Tuple[int, float]],
    sparse_weight: float,
    dense_weight: float,
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Weighted Reciprocal Rank Fusion.
    Each (index, score) list is already rank-ordered.
    Returns a merged list of (index, fused_score) sorted descending.
    """
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(sparse_ranked):
        scores[idx] = scores.get(idx, 0.0) + sparse_weight / (k + rank + 1)
    for rank, (idx, _) in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0.0) + dense_weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Loads .txt files from corpus_dir, chunks them, builds BM25 + FAISS
    indices, and retrieves via weighted RRF fusion.
    """

    def __init__(self, cfg: RetrieverConfig, corpus_dir: str = "data/corpus"):
        self.cfg = cfg
        self.corpus_dir = Path(corpus_dir)
        self._chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None
        # Embedder is loaded lazily in build() to avoid network calls on construction
        self._embedder: SentenceTransformer | None = None
        self._built = False

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Read corpus, chunk, build BM25 + FAISS. Call once at startup."""
        raw_docs = self._load_corpus()
        if not raw_docs:
            raise FileNotFoundError(
                f"No .txt files found in {self.corpus_dir}. "
                "Add documents before starting the server."
            )
        # Load embedding model here (deferred from __init__ to avoid
        # network calls during construction in tests and import time)
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.cfg.embedding_model)
        self._chunks = self._chunk_docs(raw_docs)
        self._build_bm25()
        self._build_faiss()
        self._built = True

    def _load_corpus(self) -> List[Tuple[str, str]]:
        """Return list of (filename, text) for every .txt in corpus_dir."""
        docs = []
        for path in sorted(self.corpus_dir.glob("*.txt")):
            docs.append((path.name, path.read_text(encoding="utf-8")))
        return docs

    def _chunk_docs(self, docs: List[Tuple[str, str]]) -> List[Chunk]:
        """Sliding-window character chunker."""
        chunks = []
        size = self.cfg.chunk_size
        overlap = self.cfg.chunk_overlap
        for source, text in docs:
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{source}::{chunk_idx}",
                            text=chunk_text,
                            source=source,
                            score=0.0,
                            retrieval_method="",
                        )
                    )
                    chunk_idx += 1
                start += size - overlap
        return chunks

    def _build_bm25(self) -> None:
        tokenised = [_tokenise(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenised)

    def _build_faiss(self) -> None:
        texts = [c.text for c in self._chunks]
        embeddings = self._embedder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # inner product = cosine on L2-normed vecs
        self._faiss_index.add(np.array(embeddings, dtype="float32"))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if not self._built:
            raise RuntimeError("Call build() before retrieve().")
        k = top_k or self.cfg.top_k

        sparse_ranked = self._sparse_retrieve(query, k * 3)
        dense_ranked = self._dense_retrieve(query, k * 3)
        fused = _rrf(
            sparse_ranked,
            dense_ranked,
            self.cfg.sparse_weight,
            self.cfg.dense_weight,
        )[:k]

        result_chunks = []
        for idx, fused_score in fused:
            c = self._chunks[idx]
            result_chunks.append(
                Chunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    score=round(fused_score, 4),
                    retrieval_method="hybrid_rrf",
                )
            )
        return RetrievalResult(query=query, chunks=result_chunks, method_used="hybrid_rrf")

    def _sparse_retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenise(query))
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_k_idx]

    def _dense_retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        distances, indices = self._faiss_index.search(q_emb, k)
        return [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if i >= 0]
