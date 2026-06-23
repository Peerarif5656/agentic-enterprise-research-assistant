"""
MCP (Model Context Protocol) server.

Exposes two tools that the LangGraph agent calls via MCP:
  - retrieve_documents: hybrid BM25 + FAISS retrieval over the corpus
  - summarise_text: pass-through Claude summarisation for provided text

Why MCP here?
MCP gives the agent a standardised, discoverable tool interface. Any MCP-
compatible client (Claude Desktop, LangGraph, other agents) can connect to
this server and call its tools without custom integration code. This is the
same protocol pattern used in production enterprise deployments.

The server is started as a subprocess by the LangGraph agent, but can also
be run standalone for testing:
    python -m src.mcp_server.server
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path when running as __main__
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import AnswerGenerator
from src.config import Config
from src.types import RouteDecision

# ---------------------------------------------------------------------------
# Initialise shared components (loaded once at server start)
# ---------------------------------------------------------------------------
cfg = Config.load()
retriever = HybridRetriever(cfg.retriever, corpus_dir=cfg.corpus_dir)
retriever.build()

generator = AnswerGenerator(
    api_key=Config.anthropic_key(),
    cfg=cfg.generation,
)

# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------
app = Server("enterprise-rag-agent")


@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="retrieve_documents",
            description=(
                "Retrieve relevant document chunks from the enterprise corpus "
                "for a given question, using hybrid BM25 + dense vector search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user question to retrieve documents for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
        mcp_types.Tool(
            name="summarise_text",
            description=(
                "Summarise a provided block of text in response to a specific "
                "question or focus area using Claude."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarise.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question or focus guiding the summary.",
                    },
                },
                "required": ["text", "question"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent]:
    if name == "retrieve_documents":
        question = arguments["question"]
        top_k = int(arguments.get("top_k", cfg.retriever.top_k))
        retrieval = retriever.retrieve(question, top_k=top_k)
        answer = generator.generate(retrieval, route=RouteDecision.RETRIEVE)
        payload = {
            "answer": answer.answer,
            "citations": answer.citations,
            "confidence": answer.confidence,
            "chunks": [
                {"source": c.source, "score": c.score, "text": c.text[:300]}
                for c in retrieval.chunks
            ],
        }
        return [mcp_types.TextContent(type="text", text=json.dumps(payload))]

    elif name == "summarise_text":
        text = arguments["text"]
        question = arguments["question"]
        answer = generator.summarise(text, question)
        payload = {"answer": answer.answer, "confidence": answer.confidence}
        return [mcp_types.TextContent(type="text", text=json.dumps(payload))]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
