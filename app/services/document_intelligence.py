"""Document intelligence pipeline composed of OCR and parser adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

from app.domain.dto import (
    AttachmentDescriptor,
    DocInsights,
    Entity,
    EntityMap,
    SemanticChunk,
    StructuredJobRequest,
)
from app.domain.interfaces import DocumentIntelligencePipeline


@dataclass
class OCRExtraction:
    """Result of running OCR on a single attachment."""

    attachment: AttachmentDescriptor
    text: str
    language: Optional[str] = None
    metadata: dict | None = None


@dataclass
class ParserResult:
    """Structured interpretation of OCR extraction."""

    chunks: List[SemanticChunk]
    entities: List[Entity]
    summary: Optional[str] = None


class OCRAdapter(Protocol):
    """Adapter interface for converting attachments into text."""

    def can_process(self, attachment: AttachmentDescriptor) -> bool:
        ...

    def extract(self, attachment: AttachmentDescriptor) -> Optional[OCRExtraction]:
        ...


class ParserAdapter(Protocol):
    """Adapter interface for turning OCR text into structured artifacts."""

    def supports(self, extraction: OCRExtraction) -> bool:
        ...

    def parse(self, extraction: OCRExtraction) -> ParserResult:
        ...


class DefaultDocumentIntelligencePipeline(DocumentIntelligencePipeline):
    """Coordinate OCR adapters and parser adapters to build DocInsights."""

    def __init__(
        self,
        ocr_adapters: Sequence[OCRAdapter],
        parser_adapters: Sequence[ParserAdapter],
    ) -> None:
        self._ocr_adapters = list(ocr_adapters)
        self._parser_adapters = list(parser_adapters)

    def run(self, job_request: StructuredJobRequest) -> DocInsights:
        insights = DocInsights()

        if job_request.text_input:
            insights.semantic_chunks.append(
                SemanticChunk(
                    id="payload:text",
                    text=job_request.text_input,
                    source_id="payload",
                    metadata={"source": "text_input"},
                )
            )

        for attachment in job_request.attachments:
            extraction = self._run_ocr(attachment)
            if extraction is None or not extraction.text.strip():
                continue

            parser = self._select_parser(extraction)
            if parser:
                result = parser.parse(extraction)
            else:
                result = self._default_parse(extraction)

            insights.semantic_chunks.extend(result.chunks)
            insights.entities.merge(result.entities)
            if result.summary:
                insights.summaries.append(result.summary)

        return insights

    def _run_ocr(self, attachment: AttachmentDescriptor) -> Optional[OCRExtraction]:
        for adapter in self._ocr_adapters:
            if adapter.can_process(attachment):
                return adapter.extract(attachment)
        return None

    def _select_parser(self, extraction: OCRExtraction) -> Optional[ParserAdapter]:
        for parser in self._parser_adapters:
            if parser.supports(extraction):
                return parser
        return None

    def _default_parse(self, extraction: OCRExtraction) -> ParserResult:
        chunk = SemanticChunk(
            id=f"{extraction.attachment.id}:chunk-1",
            text=extraction.text,
            source_id=extraction.attachment.id,
            metadata=extraction.metadata or {},
        )
        return ParserResult(
            chunks=[chunk],
            entities=[],
        )

