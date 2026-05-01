import json
from urllib import error, request

from research_agent.config import AppSettings


class XAITextService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return bool(self._settings.xai_api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        if not self.available:
            raise RuntimeError("XAI_API_KEY is required for Grok generation.")

        payload = {
            "model": self._settings.model_name,
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "store": False,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        response = self._post(payload)
        text = self._extract_text(response)
        if not text:
            raise RuntimeError("xAI response did not include output text.")
        return text

    def _post(self, payload: dict) -> dict:
        endpoint = "https://api.x.ai/v1/responses"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._settings.xai_api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"xAI request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"xAI connection failed: {exc.reason}") from exc

    @staticmethod
    def _extract_text(payload: dict) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        collected: list[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    collected.append(text.strip())
        return "\n".join(collected).strip()
