"""
Citation-grounded answer generation using the Anthropic Claude API.

The generator asks Claude to:
1. Answer the question using ONLY the retrieved chunks as evidence
2. Attach inline [n] citation markers to each factual claim
3. Return a confidence estimate based on how well the chunks cover the question

This directly addresses RQ2 from the dissertation — citation grounding
reduces hallucination by forcing the model to ground every claim in a
retrieved source rather than relying on parametric knowledge.
"""
from __future__ import annotations

import json
import re
from typing import List

import anthropic

from ..types import Chunk, GeneratedAnswer, RetrievalResult, RouteDecision
from ..config import GenerationConfig


_SYSTEM_PROMPT = """\
You are an enterprise research assistant. You answer questions using ONLY the
document chunks provided to you — you do not use any outside knowledge.

Rules:
1. Ground every factual claim in a retrieved chunk. Attach an inline citation
   marker [n] immediately after the claim, where n is the chunk number.
2. If the chunks do not contain enough information to answer, say so clearly
   and state what is missing — do not fabricate an answer.
3. Be concise. Enterprise users need clear, actionable answers.
4. At the end of your response, output a JSON block (and nothing else after it)
   in this exact format:
   ```json
   {"confidence": 0.85, "citations": ["source_a.txt", "source_b.txt"]}
   ```
   where confidence is 0.0–1.0 and citations lists only the source files
   that actually supported your answer.
"""


def _format_chunks(chunks: List[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] Source: {c.source}\n{c.text}")
    return "\n\n".join(lines)


def _parse_json_block(text: str) -> dict:
    """Extract the trailing JSON block from the model response."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fallback: try to find any JSON object at the end
    match = re.search(r"\{[^{}]*\"confidence\"[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {"confidence": 0.5, "citations": []}


def _strip_json_block(text: str) -> str:
    """Remove the trailing JSON metadata block from the visible answer."""
    text = re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
    return text.strip()


class AnswerGenerator:
    """Wraps the Anthropic Claude API for citation-grounded generation."""

    def __init__(self, api_key: str, cfg: GenerationConfig):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._cfg = cfg

    def generate(
        self,
        retrieval: RetrievalResult,
        route: RouteDecision,
    ) -> GeneratedAnswer:
        chunks_text = _format_chunks(retrieval.chunks)
        user_message = (
            f"Question: {retrieval.query}\n\n"
            f"Retrieved document chunks:\n{chunks_text}"
        )

        response = self._client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        metadata = _parse_json_block(raw_text)
        clean_answer = _strip_json_block(raw_text)

        return GeneratedAnswer(
            question=retrieval.query,
            answer=clean_answer,
            citations=metadata.get("citations", []),
            route_taken=route,
            retrieval_method=retrieval.method_used,
            confidence=float(metadata.get("confidence", 0.5)),
        )

    def summarise(self, text: str, question: str) -> GeneratedAnswer:
        """Summarise a provided document in response to a question."""
        response = self._client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system="You are a concise enterprise summarisation assistant.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question or focus area: {question}\n\n"
                        f"Document to summarise:\n{text}"
                    ),
                }
            ],
        )
        return GeneratedAnswer(
            question=question,
            answer=response.content[0].text,
            citations=[],
            route_taken=RouteDecision.SUMMARISE,
            retrieval_method="none",
            confidence=1.0,
        )
