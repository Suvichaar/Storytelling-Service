"""Service for constructing StructuredJobRequest objects."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.domain.dto import AttachmentDescriptor, IntakePayload, LanguageMetadata, StructuredJobRequest
from app.domain.interfaces import IngestionAggregator


class DefaultIngestionAggregator(IngestionAggregator):
    """Aggregate user input and metadata into a StructuredJobRequest."""

    def __init__(self, text_joiner: str = "\n\n") -> None:
        self._text_joiner = text_joiner

    def aggregate(self, payload: IntakePayload, language: LanguageMetadata) -> StructuredJobRequest:
        segments = self._collect_text_segments(payload, language)
        text_input = self._join_non_empty(segments)
        return StructuredJobRequest(
            text_input=text_input or None,
            url_list=[str(url) for url in payload.urls],
            attachments=self._normalize_attachments(payload.attachments),
            focus_keywords=list(payload.prompt_keywords),
        )

    def _collect_text_segments(self, payload: IntakePayload, language: LanguageMetadata) -> list[str]:
        segments: list[str] = []
        if payload.text_prompt:
            segments.append(payload.text_prompt.strip())
        if payload.notes:
            segments.append(payload.notes.strip())
        if payload.prompt_keywords:
            segments.append(" ".join(payload.prompt_keywords))
        if language.source_text_preview:
            segments.append(language.source_text_preview.strip())
        return segments

    def _normalize_attachments(self, attachments: Sequence[str]) -> list[AttachmentDescriptor]:
        descriptors: list[AttachmentDescriptor] = []
        for idx, attachment in enumerate(attachments):
            descriptor = AttachmentDescriptor(
                id=f"attachment-{idx+1}",
                uri=attachment,
                media_type=None,
                metadata={},
            )
            descriptors.append(descriptor)
        return descriptors

    def _join_non_empty(self, segments: Iterable[str]) -> str:
        cleaned = [segment for segment in segments if segment]
        return self._text_joiner.join(cleaned)

