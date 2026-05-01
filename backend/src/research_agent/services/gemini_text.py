from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency guard
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency guard
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from research_agent.config import AppSettings


class GeminiTextService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self._settings.gemini_api_key) and genai is not None and types is not None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        if not self.available:
            if not self._settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini generation.")
            raise RuntimeError("Gemini SDK is not installed. Run pip install google-genai.")

        response = self._client_or_create().models.generate_content(
            model=self._settings.gemini_generation_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        text = getattr(response, "text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()

        raise RuntimeError("Gemini response did not include output text.")

    def _client_or_create(self) -> Any:
        if self._client is None:
            if genai is None:
                raise RuntimeError("Gemini SDK is not installed.")
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client
