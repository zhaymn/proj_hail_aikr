from __future__ import annotations

import re
import time

from research_agent.config import AppSettings
from research_agent.services.gemini_text import GeminiTextService
from research_agent.services.groq_text import GroqTextService
from research_agent.services.openrouter_text import OpenRouterTextService


class TextGenerationService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._groq = GroqTextService(settings)
        self._gemini = GeminiTextService(settings)
        self._openrouter = OpenRouterTextService(settings)
        self._cooldowns: dict[str, float] = {}
        self._last_provider: str = ""

    @property
    def last_provider(self) -> str:
        return self._last_provider

    @property
    def available(self) -> bool:
        provider = self._provider()
        if provider == "groq":
            return self._groq.available
        if provider == "gemini":
            return self._gemini.available
        if provider == "openrouter":
            return self._openrouter.available
        return self._groq.available or self._gemini.available or self._openrouter.available

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        provider = self._provider()
        if provider == "groq":
            return self._generate_with_preferred_provider(
                preferred="groq",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        if provider == "gemini":
            return self._generate_with_preferred_provider(
                preferred="gemini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        if provider == "openrouter":
            return self._generate_with_preferred_provider(
                preferred="openrouter",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        return self._generate_with_auto_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _generate_with_preferred_provider(
        self,
        *,
        preferred: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        attempts: list[str] = []
        cooldown_skipped: list[str] = []
        service = self._service_for(preferred)
        if service is None or not service.available:
            return self._generate_with_auto_fallback(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        try:
            generated = service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            self._last_provider = preferred
            return generated
        except Exception as error:  # pragma: no cover - provider-specific network errors
            retried = self._retry_preferred_provider_after_short_wait(
                provider=preferred,
                service=service,
                error=error,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if retried is not None:
                self._last_provider = preferred
                return retried
            attempts.append(f"{preferred}: {str(error)[:180]}")
            self._mark_provider_cooldown(preferred, error)

        for provider_name in self._provider_order():
            if provider_name == preferred:
                continue
            fallback_service = self._service_for(provider_name)
            if fallback_service is None or not fallback_service.available:
                continue
            if self._in_cooldown(provider_name):
                cooldown_skipped.append(provider_name)
                attempts.append(f"{provider_name}: cooling down")
                continue
            try:
                generated = fallback_service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                self._last_provider = provider_name
                return generated
            except Exception as error:  # pragma: no cover - provider-specific network errors
                retried = self._retry_preferred_provider_after_short_wait(
                    provider=provider_name,
                    service=fallback_service,
                    error=error,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                if retried is not None:
                    self._last_provider = provider_name
                    return retried
                attempts.append(f"{provider_name}: {str(error)[:180]}")
                self._mark_provider_cooldown(provider_name, error)

        # If fallbacks were skipped due to cooldown, do one final pass ignoring cooldown.
        if cooldown_skipped:
            seen: set[str] = set()
            for provider_name in cooldown_skipped:
                if provider_name in seen:
                    continue
                seen.add(provider_name)
                fallback_service = self._service_for(provider_name)
                if fallback_service is None or not fallback_service.available:
                    continue
                try:
                    generated = fallback_service.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    self._last_provider = provider_name
                    return generated
                except Exception as error:  # pragma: no cover - provider-specific network errors
                    attempts.append(f"{provider_name}: {str(error)[:180]}")
                    self._mark_provider_cooldown(provider_name, error)

        emergency = self._try_emergency_compacted_generation(
            providers=self._provider_order(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if emergency is not None:
            provider_name, generated = emergency
            self._last_provider = provider_name
            return generated

        raise RuntimeError(f"Generation failed after provider failover. {' | '.join(attempts)}")

    def _generate_with_auto_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        attempts: list[str] = []
        providers = self._provider_order()
        cooldown_skipped: list[str] = []

        attempted_any = False
        for name in providers:
            service = self._service_for(name)
            if service is None or not service.available:
                continue
            if self._in_cooldown(name):
                attempts.append(f"{name}: cooling down")
                cooldown_skipped.append(name)
                continue
            attempted_any = True
            try:
                generated = service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                self._last_provider = name
                return generated
            except Exception as error:  # pragma: no cover - provider-specific network errors
                retried = self._retry_preferred_provider_after_short_wait(
                    provider=name,
                    service=service,
                    error=error,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                if retried is not None:
                    self._last_provider = name
                    return retried
                attempts.append(f"{name}: {str(error)[:180]}")
                self._mark_provider_cooldown(name, error)
                continue

        # If providers were skipped for cooldown, retry one pass ignoring cooldown.
        # Also covers the case where every available provider was in cooldown.
        if cooldown_skipped or not attempted_any:
            seen_retry_providers: set[str] = set()
            for name in providers:
                if name in seen_retry_providers:
                    continue
                seen_retry_providers.add(name)
                service = self._service_for(name)
                if service is None or not service.available:
                    continue
                try:
                    generated = service.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    self._last_provider = name
                    return generated
                except Exception as error:  # pragma: no cover - provider-specific network errors
                    second_try = self._retry_preferred_provider_after_short_wait(
                        provider=name,
                        service=service,
                        error=error,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    if second_try is not None:
                        self._last_provider = name
                        return second_try
                    attempts.append(f"{name}: {str(error)[:180]}")
                    self._mark_provider_cooldown(name, error)

        emergency = self._try_emergency_compacted_generation(
            providers=providers,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if emergency is not None:
            provider_name, generated = emergency
            self._last_provider = provider_name
            return generated

        if attempts:
            joined = " | ".join(attempts)
            raise RuntimeError(f"All generation providers failed. {joined}")

        raise RuntimeError(
            "No generation provider configured. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY."
        )

    def _provider_order(self) -> list[str]:
        configured = [
            token.strip().lower()
            for token in (self._settings.generation_fallback_order or "").split(",")
            if token.strip()
        ]
        ordered: list[str] = []
        for name in configured + ["groq", "openrouter", "gemini"]:
            if name not in {"groq", "gemini", "openrouter"}:
                continue
            if name in ordered:
                continue
            ordered.append(name)
        return ordered

    def _service_for(self, provider: str) -> GroqTextService | GeminiTextService | OpenRouterTextService | None:
        if provider == "groq":
            return self._groq
        if provider == "gemini":
            return self._gemini
        if provider == "openrouter":
            return self._openrouter
        return None

    def _in_cooldown(self, provider: str) -> bool:
        until = float(self._cooldowns.get(provider, 0.0))
        return until > time.time()

    def _mark_provider_cooldown(self, provider: str, error: Exception) -> None:
        text = str(error or "")
        lower = text.lower()
        if not self._is_transient_provider_error(lower):
            return
        cooldown = self._extract_retry_seconds(text)
        if cooldown <= 0:
            cooldown = max(30, int(self._settings.generation_provider_cooldown_seconds))
        self._cooldowns[provider] = time.time() + float(cooldown)

    @staticmethod
    def _is_transient_provider_error(lower_text: str) -> bool:
        markers = (
            "rate_limit_exceeded",
            "rate limit",
            "quota",
            "resource_exhausted",
            "tokens per day",
            "429",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "connection failed",
            "connection reset",
            "overloaded",
        )
        return any(marker in lower_text for marker in markers)

    @staticmethod
    def _extract_retry_seconds(text: str) -> int:
        match = re.search(r"try again in\s+(\d+)\s*m(?:in)?\s*(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) * 60 + int(float(match.group(2)))
        match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return int(float(match.group(1)))
        match = re.search(r"retry[-\s]?after[:\s]+(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return int(float(match.group(1)))
        return 0

    def _provider(self) -> str:
        provider = (self._settings.generation_provider or "auto").strip().lower()
        if provider in {"auto", "groq", "gemini", "openrouter"}:
            return provider
        return "auto"

    def _retry_preferred_provider_after_short_wait(
        self,
        *,
        provider: str,
        service: GroqTextService | GeminiTextService | OpenRouterTextService,
        error: Exception,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str | None:
        lower = str(error or "").lower()
        if not self._is_transient_provider_error(lower):
            return None
        retry_seconds = self._extract_retry_seconds(str(error or ""))
        if retry_seconds <= 0 or retry_seconds > 12:
            return None
        time.sleep(min(12, retry_seconds + 0.3))
        reduced_tokens = max(192, int(max_output_tokens * 0.78))
        try:
            return service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=reduced_tokens,
            )
        except Exception:
            return None

    def _try_emergency_compacted_generation(
        self,
        *,
        providers: list[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> tuple[str, str] | None:
        compact_system = self._compact_prompt_text(system_prompt, max_chars=420)
        compact_user = self._compact_prompt_text(user_prompt, max_chars=1500)
        if compact_user:
            compact_user = (
                f"{compact_user}\n\n"
                "[Emergency generation mode: produce the best grounded answer possible with concise wording.]"
            )
        emergency_tokens = max(220, min(520, int(max_output_tokens * 0.45)))
        for name in providers:
            service = self._service_for(name)
            if service is None or not service.available:
                continue
            try:
                generated = service.generate(
                    system_prompt=compact_system,
                    user_prompt=compact_user,
                    temperature=min(temperature, 0.2),
                    max_output_tokens=emergency_tokens,
                )
                if str(generated or "").strip():
                    return name, generated
            except Exception:
                continue
        return None

    @staticmethod
    def _compact_prompt_text(text: str, *, max_chars: int) -> str:
        value = str(text or "").strip()
        if not value:
            return value
        if len(value) <= max_chars:
            return value
        head_chars = int(max_chars * 0.76)
        tail_chars = max(48, int(max_chars * 0.16))
        head = value[:head_chars].rstrip()
        tail = value[-tail_chars:].lstrip()
        compacted = f"{head}\n\n[...truncated for emergency generation...]\n\n{tail}".strip()
        if len(compacted) <= max_chars:
            return compacted
        return compacted[:max_chars].rstrip()
