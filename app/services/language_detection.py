"""Language detection services and strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Tuple

from pydantic import ValidationError

from app.domain.dto import IntakePayload, LanguageMetadata
from app.domain.interfaces import LanguageDetectionService


DetectResult = Tuple[str, float]


class LanguageDetectionStrategy(Protocol):
    """Strategy abstraction for language detection engines."""

    def detect(self, text: str) -> DetectResult:
        """Return a tuple of (language_code, confidence)."""


class FastTextLanguageDetectionStrategy(LanguageDetectionStrategy):
    """Language detection backed by a FastText model."""

    def __init__(
        self,
        model_path: str,
        loader: Callable[[str], Any] | None = None,
        label_prefix: str = "__label__",
    ) -> None:
        """
        Parameters
        ----------
        model_path:
            Filesystem path to the FastText `.bin` model.
        loader:
            Optional callable used to load the model. Defaults to `fasttext.load_model`.
            Injected for easier testing/mocking.
        label_prefix:
            Prefix used by FastText labels (defaults to ``__label__``).
        """

        self._model = self._load_model(model_path, loader)
        self._label_prefix = label_prefix

    def detect(self, text: str) -> DetectResult:
        labels, confidences = self._model.predict(text, k=1)
        if not labels:
            return "unknown", 0.0
        label = labels[0]
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        language_code = label.removeprefix(self._label_prefix)
        confidence = float(confidences[0]) if confidences else 0.0
        return language_code, max(0.0, min(1.0, confidence))

    def _load_model(self, model_path: str, loader: Callable[[str], Any] | None) -> Any:
        if loader is None:
            try:
                import fasttext  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "fasttext is required for FastTextLanguageDetectionStrategy. Install it via pip."
                ) from exc
            loader = fasttext.load_model
        return loader(model_path)


@dataclass
class AggregatedText:
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class DefaultLanguageDetectionService(LanguageDetectionService):
    """Combine user input fields and run detection via the configured strategy."""

    def __init__(self, strategy: LanguageDetectionStrategy, default_language: str = "en") -> None:
        self._strategy = strategy
        self._default_language = default_language

    def detect(self, payload: IntakePayload) -> LanguageMetadata:
        aggregated = self._aggregate_text(payload)
        if aggregated.is_empty:
            return LanguageMetadata(language_code=self._default_language, confidence=0.0)

        language_code, confidence = self._strategy.detect(aggregated.text)
        language_code = language_code or self._default_language
        try:
            return LanguageMetadata(language_code=language_code, confidence=confidence, source_text_preview=self._preview(aggregated.text))
        except ValidationError as exc:
            raise ValueError(f"Invalid language detection result: {exc}") from exc

    def _aggregate_text(self, payload: IntakePayload) -> AggregatedText:
        segments: list[str] = []
        if payload.text_prompt:
            segments.append(payload.text_prompt)
        if payload.notes:
            segments.append(payload.notes)
        if payload.prompt_keywords:
            segments.append(" ".join(payload.prompt_keywords))
        if payload.urls:
            segments.append(" ".join(str(url) for url in payload.urls))
        return AggregatedText(" \n ".join(segments))

    def _preview(self, text: str, max_length: int = 200) -> str:
        return text[:max_length]

