"""Configuration management — reads config/config.yaml and env vars."""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrieverConfig:
    top_k: int = 5
    sparse_weight: float = 0.4
    dense_weight: float = 0.6
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class GenerationConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0   # deterministic for reproducibility


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass
class Config:
    retriever: RetrieverConfig
    generation: GenerationConfig
    api: APIConfig
    corpus_dir: str = "data/corpus"

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Config":
        config_path = Path(path)
        raw: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}

        r = raw.get("retriever", {})
        g = raw.get("generation", {})
        a = raw.get("api", {})

        return cls(
            retriever=RetrieverConfig(
                top_k=r.get("top_k", 5),
                sparse_weight=r.get("sparse_weight", 0.4),
                dense_weight=r.get("dense_weight", 0.6),
                chunk_size=r.get("chunk_size", 512),
                chunk_overlap=r.get("chunk_overlap", 64),
                embedding_model=r.get("embedding_model", "all-MiniLM-L6-v2"),
            ),
            generation=GenerationConfig(
                model=os.getenv("CLAUDE_MODEL", g.get("model", "claude-sonnet-4-6")),
                max_tokens=g.get("max_tokens", 1024),
                temperature=g.get("temperature", 0.0),
            ),
            api=APIConfig(
                host=a.get("host", "0.0.0.0"),
                port=int(os.getenv("PORT", a.get("port", 8000))),
                reload=a.get("reload", False),
            ),
            corpus_dir=raw.get("corpus_dir", "data/corpus"),
        )

    @staticmethod
    def anthropic_key() -> str:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        return key
