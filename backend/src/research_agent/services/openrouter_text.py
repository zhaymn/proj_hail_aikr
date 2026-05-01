import json
import re
from urllib import error, request

from research_agent.config import AppSettings


class OpenRouterTextService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return bool(self._settings.openrouter_api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        if not self.available:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter generation.")

        payload = {
            "model": self._settings.openrouter_generation_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        last_error: RuntimeError | None = None
        for _ in range(4):
            try:
                response = self._post(payload)
                break
            except RuntimeError as exc:
                last_error = exc
                error_text = str(exc)
                updated = False

                current_tokens = int(payload.get("max_tokens", max_output_tokens))
                retry_tokens = self._retry_budget_tokens(
                    error_text=error_text,
                    requested_tokens=current_tokens,
                )
                if retry_tokens is not None and retry_tokens < current_tokens:
                    payload["max_tokens"] = retry_tokens
                    updated = True

                prompt_cap = self._retry_prompt_token_cap(error_text=error_text)
                if prompt_cap is not None:
                    if prompt_cap <= 20:
                        raise RuntimeError(
                            f"OpenRouter prompt-token cap is too low for generation ({prompt_cap}). "
                            "Check account limits or switch provider."
                        ) from exc
                    current_system_prompt = str((payload.get("messages") or [{}, {}])[0].get("content", ""))
                    compacted_system = self._compact_system_prompt_for_token_cap(
                        system_prompt=current_system_prompt,
                        prompt_token_cap=prompt_cap,
                    )
                    current_user_prompt = str((payload.get("messages") or [{}, {}])[1].get("content", ""))
                    compacted = self._compact_user_prompt_for_token_cap(
                        user_prompt=current_user_prompt,
                        prompt_token_cap=prompt_cap,
                    )
                    if compacted_system != current_system_prompt:
                        payload["messages"][0]["content"] = compacted_system
                        updated = True
                    if len(compacted) < len(current_user_prompt):
                        payload["messages"][1]["content"] = compacted
                        updated = True

                # Last-resort nudge when provider gives ambiguous limits.
                if not updated and current_tokens > 128:
                    payload["max_tokens"] = max(128, int(current_tokens * 0.7))
                    updated = True

                if not updated:
                    raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("OpenRouter request failed after adaptive retries.")
        text = self._extract_text(response)
        if not text:
            raise RuntimeError("OpenRouter response did not include output text.")
        return text

    @staticmethod
    def _retry_budget_tokens(*, error_text: str, requested_tokens: int) -> int | None:
        lower = (error_text or "").lower()
        budget_markers = (
            "requires more credits",
            "fewer max_tokens",
            "can only afford",
            "insufficient credits",
            "not enough credits",
            "credit limit",
            "insufficient_balance",
        )
        if not any(marker in lower for marker in budget_markers):
            return None

        requested = max(1, int(requested_tokens))
        affordable_match = re.search(r"can only afford\s+(\d+)", error_text, flags=re.IGNORECASE)
        affordable = int(affordable_match.group(1)) if affordable_match else 0
        fallback = int(requested * 0.6)
        candidate = affordable if affordable > 0 else fallback
        # Keep enough room for completion text while still shrinking meaningfully.
        min_retry = 64
        candidate = max(min_retry, candidate)
        if candidate >= requested:
            candidate = max(min_retry, requested - 64)
        if candidate >= requested:
            return None
        return candidate

    @staticmethod
    def _retry_prompt_token_cap(*, error_text: str) -> int | None:
        match = re.search(
            r"prompt tokens limit exceeded:\s*(\d+)\s*>\s*(\d+)",
            error_text or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        cap = int(match.group(2))
        if cap <= 0:
            return None
        return cap

    @staticmethod
    def _compact_system_prompt_for_token_cap(*, system_prompt: str, prompt_token_cap: int) -> str:
        text = str(system_prompt or "")
        if not text:
            return (
                "You are a concise research assistant. "
                "Use provided evidence only, avoid speculation, and cite claims as [n]."
            )
        if prompt_token_cap <= 80:
            return "Concise research assistant. Ground claims with [n]."
        if prompt_token_cap <= 220:
            return (
                "You are a concise research assistant. "
                "Use provided evidence only, avoid speculation, and cite claims as [n]."
            )
        if prompt_token_cap <= 420:
            max_chars = max(180, int(prompt_token_cap * 1.2))
            return text[:max_chars].rstrip()
        return text

    @staticmethod
    def _compact_user_prompt_for_token_cap(*, user_prompt: str, prompt_token_cap: int) -> str:
        text = str(user_prompt or "")
        if not text:
            return text
        # Rough token-to-char conversion for fallback compaction.
        if prompt_token_cap <= 80:
            max_chars = max(40, int(prompt_token_cap * 0.9))
        elif prompt_token_cap <= 220:
            max_chars = max(120, int(prompt_token_cap * 1.2))
        elif prompt_token_cap <= 420:
            max_chars = max(220, int(prompt_token_cap * 1.6))
        elif prompt_token_cap <= 900:
            max_chars = max(420, int(prompt_token_cap * 2.0))
        else:
            max_chars = max(420, min(2800, int(prompt_token_cap * 3)))
        if len(text) <= max_chars:
            return text
        head = text[: int(max_chars * 0.72)].rstrip()
        tail = text[-int(max_chars * 0.18) :].lstrip()
        marker = "\n\n[Context truncated to fit provider prompt-token budget.]\n\n"
        compacted = f"{head}{marker}{tail}".strip()
        if len(compacted) <= max_chars:
            return compacted
        return compacted[:max_chars].rstrip()

    def _post(self, payload: dict) -> dict:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "research-agent/0.1",
            },
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc

    @staticmethod
    def _extract_text(payload: dict) -> str:
        collected: list[str] = []
        for choice in payload.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                collected.append(content.strip())
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    collected.append(text.strip())
        return "\n".join(collected).strip()
