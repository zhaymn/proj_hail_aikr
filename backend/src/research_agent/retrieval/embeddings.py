import hashlib
import math
import re
from typing import Any

try:  # pragma: no cover - optional dependency guard
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency guard
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from research_agent.config import AppSettings


class GeminiEmbeddingService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self._settings.gemini_api_key) and genai is not None and types is not None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._embed([text], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            raise RuntimeError("Gemini embeddings did not return a query vector.")
        return embeddings[0]

    def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not self.available:
            if not self._settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for embedding generation.")
            raise RuntimeError("Gemini SDK is not installed. Run pip install google-genai.")

        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in self._batch(texts, self._settings.embedding_batch_size):
            result = self._client_or_create().models.embed_content(
                model=self._settings.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._settings.embedding_dimensions,
                ),
            )
            embeddings = getattr(result, "embeddings", None) or []
            batch_vectors = [list(item.values) for item in embeddings]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    "Gemini embeddings count did not match request batch size."
                )
            vectors.extend(batch_vectors)
        return vectors

    def _client_or_create(self) -> Any:
        if self._client is None:
            if genai is None:
                raise RuntimeError("Gemini SDK is not installed.")
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    @staticmethod
    def _batch(items: list[str], size: int) -> list[list[str]]:
        if size <= 0:
            return [items]
        return [items[index : index + size] for index in range(0, len(items), size)]


class LocalHashEmbeddingService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text, as_query=False) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text, as_query=True)

    def _embed_text(self, text: str, *, as_query: bool) -> list[float]:
        dims = max(64, int(self._settings.embedding_dimensions))
        vector = [0.0] * dims
        content = text or ""
        if not content.strip():
            return vector

        tokens = self._tokenize(content)
        prefix = "q" if as_query else "d"
        for token in tokens:
            digest = hashlib.sha256(f"{prefix}:{token}".encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dims
            sign = -1.0 if digest[4] & 1 else 1.0
            vector[index] += sign

        # L2 normalize for stable cosine behavior.
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())


class EmbeddingService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._gemini = GeminiEmbeddingService(settings)
        self._local = LocalHashEmbeddingService(settings)

    @property
    def available(self) -> bool:
        provider = (self._settings.embedding_provider or "auto").lower().strip()
        if provider == "gemini":
            return self._gemini.available
        return True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        provider = (self._settings.embedding_provider or "auto").lower().strip()
        if provider == "local":
            return self._local.embed_documents(texts)
        if provider == "gemini":
            return self._gemini.embed_documents(texts)
        return self._embed_with_auto_fallback(texts=texts, task="documents")

    def embed_query(self, text: str) -> list[float]:
        provider = (self._settings.embedding_provider or "auto").lower().strip()
        if provider == "local":
            return self._local.embed_query(text)
        if provider == "gemini":
            return self._gemini.embed_query(text)
        return self._embed_with_auto_fallback(texts=[text], task="query")[0]

    def _embed_with_auto_fallback(self, *, texts: list[str], task: str) -> list[list[float]]:
        if self._gemini.available:
            try:
                if task == "query":
                    return [self._gemini.embed_query(texts[0])]
                return self._gemini.embed_documents(texts)
            except Exception as error:
                if not self._is_quota_or_rate_error(error):
                    raise
        return self._local.embed_documents(texts)

    @staticmethod
    def _is_quota_or_rate_error(error: Exception) -> bool:
        message = str(error).lower()
        markers = [
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "429",
            "retry in",
        ]
        return any(marker in message for marker in markers)
