"""Narrative model client implementations for Curious and News modes."""

from __future__ import annotations

from typing import Iterable, Protocol

from app.domain.dto import (
    CuriousNarrative,
    DocInsights,
    Mode,
    NarrativeResponse,
    NewsNarrative,
    RenderedPrompt,
    SemanticChunk,
    SlideBlock,
    SlideDeck,
)
from app.domain.interfaces import ModelClient


class LanguageModel(Protocol):
    """Protocol describing minimal LLM behavior required by model clients."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return generated text given system and user prompts."""


def _aggregate_chunks(chunks: Iterable[SemanticChunk], limit: int = 3) -> str:
    selected = []
    for chunk in chunks:
        if not chunk.text:
            continue
        selected.append(f"- {chunk.text.strip()}")
        if len(selected) >= limit:
            break
    return "\n".join(selected)


def _build_slide_deck(content_sections: list[str], template_key: str, language_code: str | None) -> SlideDeck:
    slides = [
        SlideBlock(
            placeholder_id=f"section_{idx+1}",
            text=section,
        )
        for idx, section in enumerate(content_sections)
    ] or [
        SlideBlock(placeholder_id="section_1", text="No content generated.")
    ]
    return SlideDeck(template_key=template_key, language_code=language_code, slides=slides)


class CuriousModelClient(ModelClient):
    """Curious mode model client focusing on explainability-heavy narratives."""

    mode: Mode = Mode.CURIOUS

    def __init__(self, language_model: LanguageModel, template_key: str = "curious_default") -> None:
        self._language_model = language_model
        self._template_key = template_key

    def generate(self, prompt: RenderedPrompt, insights: DocInsights) -> NarrativeResponse:
        user_prompt = self._compose_user_prompt(prompt.user, insights)
        raw_output = self._language_model.complete(prompt.system, user_prompt)
        sections = self._split_sections(raw_output)
        slide_deck = _build_slide_deck(sections, self._template_key, prompt.metadata.get("language"))
        explainability = self._build_explainability_notes(insights, sections)
        return CuriousNarrative(
            mode=self.mode,
            slide_deck=slide_deck,
            raw_output=raw_output,
            explainability_notes=explainability,
            reasoning_trace="\n".join(sections),
        )

    def _compose_user_prompt(self, base_prompt: str, insights: DocInsights) -> str:
        context = _aggregate_chunks(insights.semantic_chunks, limit=3)
        entities = ", ".join(
            {entity.name for entity_list in insights.entities.entities.values() for entity in entity_list}
        )
        segments = [base_prompt]
        if context:
            segments.append("Contextual Highlights:\n" + context)
        if entities:
            segments.append(f"Key Entities: {entities}")
        return "\n\n".join(segments)

    def _split_sections(self, output: str) -> list[str]:
        sections = [section.strip() for section in output.split("\n\n") if section.strip()]
        return sections or ["No explainable narrative generated."]

    def _build_explainability_notes(self, insights: DocInsights, sections: list[str]) -> list[str]:
        notes = []
        for idx, section in enumerate(sections):
            source_chunk = insights.semantic_chunks[idx].text if idx < len(insights.semantic_chunks) else ""
            notes.append(f"Section {idx+1}: {section[:120]} (Source excerpt: {source_chunk[:120]})")
        return notes


class NewsModelClient(ModelClient):
    """News mode model client focusing on concise headlines and bulletins."""

    mode: Mode = Mode.NEWS

    def __init__(self, language_model: LanguageModel, template_key: str = "news_default") -> None:
        self._language_model = language_model
        self._template_key = template_key

    def generate(self, prompt: RenderedPrompt, insights: DocInsights) -> NarrativeResponse:
        user_prompt = self._compose_user_prompt(prompt.user, insights)
        raw_output = self._language_model.complete(prompt.system, user_prompt)
        headlines, bullet_points = self._extract_news_sections(raw_output)
        slide_deck = _build_slide_deck(headlines + bullet_points, self._template_key, prompt.metadata.get("language"))
        return NewsNarrative(
            mode=self.mode,
            slide_deck=slide_deck,
            raw_output=raw_output,
            headlines=headlines,
            bullet_points=bullet_points,
        )

    def _compose_user_prompt(self, base_prompt: str, insights: DocInsights) -> str:
        context = _aggregate_chunks(insights.semantic_chunks, limit=2)
        meta = f"Entities: {', '.join(insights.entities.entities.keys())}" if insights.entities.entities else ""
        sections = [base_prompt]
        if context:
            sections.append("Context:\n" + context)
        if meta:
            sections.append(meta)
        return "\n\n".join(sections)

    def _extract_news_sections(self, output: str) -> tuple[list[str], list[str]]:
        lines = [line.strip("-• ").strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return (["Breaking Update Unavailable"], [])
        headline = lines[0]
        bullet_points = lines[1:] if len(lines) > 1 else []
        return ([headline], bullet_points)


__all__ = ["CuriousModelClient", "NewsModelClient", "LanguageModel"]

