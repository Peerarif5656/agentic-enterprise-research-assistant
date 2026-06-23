"""
LangGraph agentic orchestration layer.

Graph structure:
    START → router → [retrieve | summarise | clarify] → END

Nodes:
  - router:    Classifies the query into a RouteDecision using Claude
  - retrieve:  Calls hybrid retrieval then citation-grounded generation
  - summarise: Delegates to the summarise tool (for provided-text tasks)
  - clarify:   Returns a clarification request without calling the retriever

Each node is a pure function: AgentState → AgentState.
State flows through the graph; no node mutates shared mutable state.

Why LangGraph?
  - Explicit, auditable graph — each routing decision and tool call is logged
  - Conditional edges let us add more routes (e.g. web_search) without
    touching existing node code
  - Built-in checkpointing supports conversation memory across turns
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from ..types import AgentState, RouteDecision, RetrievalResult
from ..retrieval.hybrid import HybridRetriever
from ..generation.generator import AnswerGenerator
from ..config import Config
import anthropic


# ---------------------------------------------------------------------------
# Router — determines which branch the query takes
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a query router for an enterprise research assistant.
Classify the user query into exactly one of these three routes:

  retrieve  — The user is asking a question that can be answered by searching
               an internal document corpus (e.g. factual questions, definitions,
               policy questions, technical how-tos).

  summarise — The user has provided a block of text and wants it summarised or
               analysed.

  clarify   — The query is so ambiguous, vague, or off-topic that a useful
               answer cannot be produced without more information.

Respond with ONLY one word: retrieve, summarise, or clarify.
"""


def make_router_node(api_key: str, cfg: Config):
    client = anthropic.Anthropic(api_key=api_key)

    def router(state: AgentState) -> AgentState:
        response = client.messages.create(
            model=cfg.generation.model,
            max_tokens=10,
            temperature=0.0,
            system=_ROUTER_SYSTEM,
            messages=[{"role": "user", "content": state.question}],
        )
        raw = response.content[0].text.strip().lower()
        route_map = {
            "retrieve": RouteDecision.RETRIEVE,
            "summarise": RouteDecision.SUMMARISE,
            "clarify": RouteDecision.CLARIFY,
        }
        state.route = route_map.get(raw, RouteDecision.RETRIEVE)
        return state

    return router


# ---------------------------------------------------------------------------
# Retrieve node — hybrid retrieval + citation-grounded generation
# ---------------------------------------------------------------------------

def make_retrieve_node(retriever: HybridRetriever, generator: AnswerGenerator):
    def retrieve(state: AgentState) -> AgentState:
        try:
            result = retriever.retrieve(state.question)
            state.retrieval_result = result
            state.answer = generator.generate(result, route=RouteDecision.RETRIEVE)
        except Exception as e:
            state.error = f"Retrieval error: {e}"
        return state

    return retrieve


# ---------------------------------------------------------------------------
# Summarise node — direct Claude call, no retrieval
# ---------------------------------------------------------------------------

def make_summarise_node(generator: AnswerGenerator):
    def summarise(state: AgentState) -> AgentState:
        # The "text to summarise" is expected in the question field for now.
        # A production system would have a separate `document` field.
        state.answer = generator.summarise(state.question, state.question)
        return state

    return summarise


# ---------------------------------------------------------------------------
# Clarify node — returns a request for more information
# ---------------------------------------------------------------------------

def clarify(state: AgentState) -> AgentState:
    from ..types import GeneratedAnswer
    state.answer = GeneratedAnswer(
        question=state.question,
        answer=(
            "I need a little more context to answer this well. "
            "Could you clarify what you're looking for? "
            "For example, which document, system, or topic does this relate to?"
        ),
        citations=[],
        route_taken=RouteDecision.CLARIFY,
        retrieval_method="none",
        confidence=0.0,
    )
    return state


# ---------------------------------------------------------------------------
# Routing edge — directs flow after the router node
# ---------------------------------------------------------------------------

def route_edge(state: AgentState) -> str:
    return state.route.value if state.route else "retrieve"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    retriever: HybridRetriever,
    generator: AnswerGenerator,
    api_key: str,
    cfg: Config,
) -> StateGraph:
    """
    Builds and compiles the LangGraph state machine.

    Graph:
        START → router ──► retrieve  → END
                       ──► summarise → END
                       ──► clarify   → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", make_router_node(api_key, cfg))
    graph.add_node("retrieve", make_retrieve_node(retriever, generator))
    graph.add_node("summarise", make_summarise_node(generator))
    graph.add_node("clarify", clarify)

    # Wire edges
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_edge,
        {
            RouteDecision.RETRIEVE.value: "retrieve",
            RouteDecision.SUMMARISE.value: "summarise",
            RouteDecision.CLARIFY.value: "clarify",
        },
    )
    graph.add_edge("retrieve", END)
    graph.add_edge("summarise", END)
    graph.add_edge("clarify", END)

    return graph.compile()
