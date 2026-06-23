# Agentic Enterprise Research Assistant

**A production-ready agentic RAG system built with LangGraph, Claude, MCP, FastAPI, and Docker.**

This system answers enterprise questions by routing queries through a LangGraph state machine, retrieving evidence via hybrid BM25 + FAISS search, and generating citation-grounded answers using the Claude API — all exposed as a REST API and containerised for one-command deployment.

---

## Architecture

```
User Question (HTTP POST /query)
        │
        ▼
   FastAPI Layer
        │
        ▼
  LangGraph Agent
        │
   ┌────┴────┐
   │  Router │ ← Claude classifies query → retrieve | summarise | clarify
   └────┬────┘
        │
  ┌─────┴──────┐
  │            │
retrieve    summarise
  │            │
  ▼            ▼
Hybrid       Claude
Retrieval    Direct
(BM25+FAISS) Call
  │            │
  ▼            ▼
Citation-Grounded Answer (via Claude)
        │
        ▼
   REST Response (JSON: answer, citations, confidence, latency_ms)
```

### Components

| Component | Technology | Role |
|---|---|---|
| Agent Orchestration | LangGraph | State machine with conditional routing across retrieve / summarise / clarify branches |
| Sparse Retrieval | BM25 (rank-bm25) | Keyword-based retrieval, strong for entity and numeric queries |
| Dense Retrieval | FAISS + sentence-transformers | Semantic vector search, strong for conceptual queries |
| Fusion | Reciprocal Rank Fusion (RRF) | Merges sparse and dense ranked lists without score normalisation |
| Generation | Claude API (Anthropic) | Citation-grounded answer generation with inline `[n]` source markers |
| Tool Protocol | MCP (Model Context Protocol) | Standardised tool interface enabling any MCP-compatible client to call retrieve/summarise |
| API | FastAPI + uvicorn | REST endpoints with Pydantic validation and structured JSON responses |
| Deployment | Docker + docker-compose | One-command containerised deployment |

### Why hybrid retrieval?
Sparse (BM25) and dense (vector) retrieval have complementary failure modes. BM25 handles exact terminology; dense embeddings handle paraphrasing and conceptual similarity. RRF fusion consistently outperforms either alone on enterprise QA benchmarks — this design directly extends my MSc dissertation research on adaptive hybrid RAG for hallucination reduction.

### Why LangGraph?
LangGraph makes the agent's routing decisions explicit and auditable. Each node is a pure function; state flows through the graph unchanged except by the node that produced it. Conditional edges mean routing logic is concentrated in one `route_edge` function rather than scattered through application code. This is crucial for production systems where you need to debug why a query went to the wrong branch.

### Why MCP?
MCP gives the retrieval and summarisation capabilities a standardised, discoverable interface. Any MCP-compatible client — Claude Desktop, other LangGraph agents, future internal tools — can connect to the MCP server and call these tools without custom integration work. This is the tool protocol pattern now used across Anthropic's production systems.

---

## Project Structure

```
enterprise-rag-agent/
├── src/
│   ├── types.py              # Shared dataclasses: AgentState, Chunk, GeneratedAnswer, RouteDecision
│   ├── config.py             # Config loader (config.yaml + env var overrides)
│   ├── agent/
│   │   └── graph.py          # LangGraph state machine: nodes, edges, routing
│   ├── retrieval/
│   │   └── hybrid.py         # BM25 + FAISS + RRF hybrid retriever
│   ├── generation/
│   │   └── generator.py      # Citation-grounded generation via Claude API
│   ├── mcp_server/
│   │   └── server.py         # MCP server exposing retrieve + summarise tools
│   └── api/
│       └── app.py            # FastAPI REST API with lifespan index build
├── config/
│   └── config.yaml           # All tunable parameters (top_k, weights, model, chunk size)
├── data/
│   └── corpus/               # Drop .txt files here — loaded at startup
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   └── test_agent.py         # Unit + integration + API tests (pytest)
├── main.py                   # Uvicorn entrypoint
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Peerarif5656/agentic-enterprise-research-assistant.git
cd agentic-enterprise-research-assistant

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Add your documents

Drop `.txt` files into `data/corpus/`. Two sample documents are already included so the system runs out of the box.

### 4. Run the server

```bash
python main.py
```

API will be live at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

---

## Docker Deployment

```bash
# Build and run (one command)
cd docker
docker-compose up --build

# Or build manually
docker build -f docker/Dockerfile -t rag-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... \
  -v $(pwd)/data/corpus:/app/data/corpus:ro rag-agent
```

---

## API Reference

### `POST /query`

Run the full agent pipeline on a natural language question.

**Request:**
```json
{
  "question": "What is hybrid retrieval and why does it outperform single-method approaches?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is hybrid retrieval...",
  "answer": "Hybrid retrieval combines BM25 sparse search with FAISS dense vector search [1]. The two methods have complementary failure modes: BM25 handles exact terminology while dense embeddings handle semantic similarity [2]. Reciprocal Rank Fusion merges the ranked lists without requiring score normalisation [1].",
  "citations": ["enterprise_ai_guide.txt", "langgraph_reference.txt"],
  "route_taken": "retrieve",
  "retrieval_method": "hybrid_rrf",
  "confidence": 0.87,
  "latency_ms": 1243.5,
  "chunks": [
    {"source": "enterprise_ai_guide.txt", "score": 0.0231, "text": "Hybrid retrieval combines sparse..."}
  ]
}
```

### `GET /health`
Returns server status, active model, and corpus chunk count.

### `GET /corpus/stats`
Returns total chunks, total documents, and corpus directory path.

### `POST /corpus/reload`
Hot-reloads the corpus (add new documents without restarting the server).

---

## Configuration

All parameters in `config/config.yaml`:

```yaml
retriever:
  top_k: 5            # chunks returned per query
  sparse_weight: 0.4  # BM25 contribution to RRF
  dense_weight: 0.6   # FAISS contribution to RRF
  chunk_size: 512     # characters per chunk
  chunk_overlap: 64   # overlap between adjacent chunks
  embedding_model: "all-MiniLM-L6-v2"

generation:
  model: "claude-sonnet-4-6"
  max_tokens: 1024
  temperature: 0.0    # deterministic for reproducibility

api:
  host: "0.0.0.0"
  port: 8000
```

Override any value with an environment variable: `CLAUDE_MODEL`, `PORT`.

---

## Testing

```bash
# Unit + integration tests (no API key needed)
pytest -q -m "not api"

# All tests including API endpoints (needs ANTHROPIC_API_KEY)
pytest -q

# Verbose with coverage
pytest -v --tb=short
```

---

## Design Decisions

**Temperature 0 generation** — Deterministic outputs for reproducibility, critical in enterprise settings where consistent answers matter.

**RRF over score normalisation** — BM25 and cosine similarity scores are on different scales; RRF fuses ranked lists without requiring normalisation, avoiding the calibration problem.

**Chunking at character level** — Simpler than sentence-level and avoids dependency on language-specific tokenisers, making the system corpus-agnostic.

**Citation grounding** — The system prompt instructs Claude to attach `[n]` markers to every factual claim. This is the primary hallucination-reduction mechanism, directly derived from my MSc dissertation (RQ2: does citation grounding reduce hallucination?).

**MCP as tool protocol** — Rather than coupling the agent to specific tool implementations, MCP decouples the interface. This means the MCP server can be replaced or extended (e.g. adding a web search tool) without modifying the LangGraph graph.

---

## Related Work

This project extends ideas from my MSc dissertation:
*Design and Evaluation of an Adaptive Hybrid Retrieval-Augmented Generation Framework for Hallucination-Resilient Enterprise Question Answering* — Loughborough University London, 2026.

The dissertation evaluates six RAG configurations (sparse, dense, static hybrid, static hybrid + citations, adaptive, adaptive + citations) against FinanceBench, CUAD, and a custom Azure documentation corpus. This project takes the best-performing configuration (adaptive hybrid + citation grounding) and wraps it in a production-ready agentic API.

---

## Tech Stack

`Python 3.11` · `LangGraph` · `Anthropic Claude API` · `MCP` · `FastAPI` · `FAISS` · `BM25` · `sentence-transformers` · `Docker`
