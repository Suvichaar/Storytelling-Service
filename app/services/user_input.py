"""Service implementation for constructing IntakePayload objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from pydantic import HttpUrl, TypeAdapter, ValidationError

from app.domain.dto import IntakePayload
from app.domain.interfaces import UserInputService


RawInputs = dict[str, Any]
AttachmentNormalizer = Callable[[Any], str]


class DefaultUserInputService(UserInputService):
    """Convert raw user form data into a validated IntakePayload."""

    def __init__(self, attachment_normalizer: AttachmentNormalizer | None = None) -> None:
        """
        Parameters
        ----------
        attachment_normalizer:
            Optional callable that converts uploaded objects into storage keys.
        """

        self._attachment_normalizer = attachment_normalizer or (lambda value: str(value))

    def build_payload(self, **raw_inputs: Any) -> IntakePayload:
        """Transform raw inputs into a normalized IntakePayload instance."""

        candidate = {
            "text_prompt": raw_inputs.get("text_prompt"),
            "notes": raw_inputs.get("notes"),
            "urls": self._normalize_urls(raw_inputs.get("urls")),
            "attachments": self._normalize_attachments(raw_inputs.get("attachments")),
            "prompt_keywords": self._normalize_strings(raw_inputs.get("prompt_keywords")),
            "mode": raw_inputs.get("mode"),
            "template_key": raw_inputs.get("template_key"),
            "slide_count": self._maybe_as_int(raw_inputs.get("slide_count")),
            "category": raw_inputs.get("category"),
            "image_source": raw_inputs.get("image_source"),
            "voice_engine": raw_inputs.get("voice_engine"),
        }
        try:
            return IntakePayload(**candidate)
        except ValidationError as exc:  # pragma: no cover - rely on pydantic message
            raise ValueError(f"Invalid intake payload: {exc}") from exc

    def _normalize_strings(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            pieces = [piece.strip() for piece in values.split(",")]
            return [piece for piece in pieces if piece]
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            normalized: list[str] = []
            for item in values:
                normalized.extend(self._normalize_strings(item))
            return normalized
        return [str(values)]

    def _normalize_urls(self, values: Any) -> list[str]:
        adapter: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)
        normalized: list[str] = []
        for candidate in self._normalize_strings(values):
            try:
                normalized.append(str(adapter.validate_python(candidate)))
            except ValidationError:
                continue
        return normalized

    def _normalize_attachments(self, values: Any) -> list[str]:
        normalized: list[str] = []
        if values is None:
            return normalized
        if isinstance(values, (str, bytes)):
            normalized.append(self._attachment_normalizer(values))
            return normalized
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for item in values:
                normalized.append(self._attachment_normalizer(item))
        else:
            normalized.append(self._attachment_normalizer(values))
        return normalized

    def _maybe_as_int(self, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

