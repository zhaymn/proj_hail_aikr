import json
from urllib import error, request

from research_agent.config import AppSettings


class GroqTextService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return bool(self._settings.groq_api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        if not self.available:
            raise RuntimeError("GROQ_API_KEY is required for Groq generation.")

        payload = {
            "model": self._settings.generation_model,
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
        response = self._post(payload)
        text = self._extract_text(response)
        if not text:
            raise RuntimeError("Groq response did not include output text.")
        return text

    def _post(self, payload: dict) -> dict:
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._settings.groq_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Groq's edge may block default Python urllib fingerprints.
                "User-Agent": "research-agent/0.1",
            },
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Groq connection failed: {exc.reason}") from exc

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
