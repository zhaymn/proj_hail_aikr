from __future__ import annotations

import math
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from research_agent.config import AppSettings
from research_agent.retrieval.embeddings import EmbeddingService

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+){0,3}\s+)?[A-Z][A-Z0-9\s,:;()/_-]{3,}$")


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int | None
    is_heading: bool


class SemanticPaperChunker:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._embeddings = EmbeddingService(settings)

    def chunk_pages(
        self,
        pages: list[Document],
        *,
        paper_id: str,
        filename: str,
    ) -> list[Document]:
        units = self._build_units(pages)
        if not units:
            return []

        vectors = self._embeddings.embed_documents([unit.text for unit in units])
        break_points = self._find_break_points(units, vectors)

        chunk_groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        current_chars = 0
        max_chars = max(220, int(self._settings.chunk_size))

        for index, unit in enumerate(units):
            is_boundary = index in break_points and bool(current)
            would_overflow = bool(current) and (current_chars + len(unit.text) + 1 > max_chars)
            if is_boundary or would_overflow:
                chunk_groups.append(current)
                overlap_units = self._overlap_tail(current)
                current = overlap_units.copy()
                current_chars = self._joined_char_count(current)
                # If overlap itself is already too large, trim to keep chunking moving.
                while current and current_chars > max_chars * 0.75:
                    current = current[1:]
                    current_chars = self._joined_char_count(current)

            current.append(unit)
            current_chars = self._joined_char_count(current)

        if current:
            chunk_groups.append(current)

        prepared: list[Document] = []
        for group in chunk_groups:
            text = " ".join(item.text.strip() for item in group if item.text.strip()).strip()
            if len(text) < self._settings.paragraph_min_chars:
                continue
            chunk_index = len(prepared)
            page = group[0].page if group else None
            prepared.append(
                Document(
                    page_content=text,
                    metadata={
                        "paper_id": paper_id,
                        "filename": filename,
                        "chunk_id": f"{paper_id}:{chunk_index}",
                        "chunk_index": chunk_index,
                        "page": page,
                    },
                )
            )
        return prepared

    def _build_units(self, pages: list[Document]) -> list[_Unit]:
        units: list[_Unit] = []
        max_unit_chars = max(180, int(self._settings.semantic_unit_max_chars))

        for page_document in pages:
            raw_text = self._sanitize_page_text(page_document.page_content or "")
            if not raw_text:
                continue
            page = self._resolve_page(page_document)
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw_text) if part.strip()]
            for paragraph in paragraphs:
                compact = re.sub(r"\s+", " ", paragraph).strip()
                if not compact:
                    continue
                is_heading = self._looks_like_heading(compact)
                for unit_text in self._split_into_semantic_units(compact, max_unit_chars=max_unit_chars):
                    units.append(_Unit(text=unit_text, page=page, is_heading=is_heading))

        return units

    def _sanitize_page_text(self, text: str) -> str:
        raw = (text or "").replace("\r", "\n")
        raw = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", raw)
        raw = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", raw)
        raw = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
        lines = [line.strip() for line in raw.splitlines()]

        kept_lines: list[str] = []
        for line in lines:
            if not line:
                continue
            if self._is_noise_line(line):
                continue
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        lower = line.lower().strip()
        if not lower:
            return True
        if re.fullmatch(r"\d+(?:\s+\d+){0,80}", lower):
            return True
        if re.fullmatch(r"page\s+\d+\s*", lower):
            return True
        if lower in {"manuscript submitted to acm", "anonymous author(s)"}:
            return True
        if "permission to make digital or hard copies" in lower:
            return True
        if "copyrights for components" in lower:
            return True
        if "request permissions from" in lower:
            return True
        if "publication rights licensed to acm" in lower:
            return True
        if line.count(" ") <= 2 and sum(ch.isdigit() for ch in line) >= max(3, len(line) // 2):
            return True
        return False

    def _split_into_semantic_units(self, text: str, *, max_unit_chars: int) -> list[str]:
        if len(text) <= max_unit_chars:
            return [text]

        sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if not sentences:
            return [text[i : i + max_unit_chars] for i in range(0, len(text), max_unit_chars)]

        units: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_unit_chars:
                if current:
                    units.append(current)
                    current = ""
                units.extend(
                    sentence[i : i + max_unit_chars].strip()
                    for i in range(0, len(sentence), max_unit_chars)
                    if sentence[i : i + max_unit_chars].strip()
                )
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_unit_chars:
                current = candidate
            else:
                if current:
                    units.append(current)
                current = sentence
        if current:
            units.append(current)
        return units

    def _find_break_points(self, units: list[_Unit], vectors: list[list[float]]) -> set[int]:
        break_points: set[int] = set()
        if len(units) <= 1 or len(vectors) != len(units):
            return break_points

        similarities: list[float] = []
        for index in range(1, len(vectors)):
            similarities.append(self._cosine(vectors[index - 1], vectors[index]))

        if similarities:
            mean = sum(similarities) / len(similarities)
            variance = sum((value - mean) ** 2 for value in similarities) / len(similarities)
            std = math.sqrt(max(0.0, variance))
            similarity_threshold = max(
                float(self._settings.semantic_similarity_floor),
                min(0.85, mean - (0.45 * std)),
            )
        else:
            similarity_threshold = float(self._settings.semantic_similarity_floor)

        for index in range(1, len(units)):
            if units[index - 1].page != units[index].page:
                break_points.add(index)
                continue
            if units[index].is_heading:
                break_points.add(index)
                continue
            similarity = similarities[index - 1] if index - 1 < len(similarities) else 1.0
            if similarity < similarity_threshold:
                break_points.add(index)

        return break_points

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        overlap_chars = max(0, int(self._settings.chunk_overlap))
        if overlap_chars <= 0 or not units:
            return []

        selected: list[_Unit] = []
        total = 0
        for unit in reversed(units):
            selected.append(unit)
            total += len(unit.text)
            if total >= overlap_chars:
                break
        selected.reverse()
        return selected

    @staticmethod
    def _resolve_page(document: Document) -> int | None:
        raw_page = (document.metadata or {}).get("page")
        if isinstance(raw_page, int):
            return raw_page + 1
        return None

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        compact = re.sub(r"\s+", " ", text).strip()
        token_count = len(_TOKEN_RE.findall(compact))
        if token_count <= 12 and compact.isupper():
            return True
        if token_count <= 16 and bool(_HEADING_RE.match(compact)):
            return True
        return False

    @staticmethod
    def _joined_char_count(units: list[_Unit]) -> int:
        if not units:
            return 0
        return sum(len(unit.text) for unit in units) + max(0, len(units) - 1)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return numerator / (left_norm * right_norm)
