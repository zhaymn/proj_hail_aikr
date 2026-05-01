import json
import re
from datetime import UTC, datetime

from research_agent.config import AppSettings
from research_agent.schemas import PaperSummary, StyleProfileResponse
from research_agent.services.text_generation import TextGenerationService


class StyleMemoryService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._llm = TextGenerationService(settings)

    def get_profile(self) -> StyleProfileResponse:
        payload = self._load()
        return StyleProfileResponse(
            active=bool(payload.get("profile")),
            profile=payload.get("profile", ""),
            source_count=len(payload.get("paper_ids", [])),
            updated_at=payload.get("updated_at"),
        )

    def update_from_paper(self, paper: PaperSummary, paper_text: str) -> StyleProfileResponse:
        payload = self._load()
        existing = payload.get("profile", "")
        profile = self._extract_profile(existing, paper.filename, paper_text[:12000])

        paper_ids = set(payload.get("paper_ids", []))
        paper_ids.add(paper.paper_id)
        updated = {
            "profile": profile,
            "paper_ids": sorted(paper_ids),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._save(updated)
        return self.get_profile()

    def reset(self) -> bool:
        if not self._settings.style_profile_store.exists():
            return False
        self._settings.style_profile_store.unlink()
        return True

    def _extract_profile(self, existing_profile: str, filename: str, paper_text: str) -> str:
        if not self._llm.available:
            return self._heuristic_profile(existing_profile, filename, paper_text)

        try:
            return self._llm.generate(
                system_prompt="You extract concise academic writing-style memories for a research writing assistant.",
                user_prompt=(
                    "Update the user's persistent style profile in at most 180 words.\n\n"
                    f"Existing profile:\n{existing_profile or 'None yet.'}\n\n"
                    f"Paper source: {filename}\n\n"
                    f"Paper excerpt:\n{paper_text}\n\n"
                    "Capture tone, sentence rhythm, citation habits, section structure, "
                    "paragraph density, and stylistic quirks that a writer mode should imitate."
                ),
                temperature=0.0,
                max_output_tokens=260,
            )
        except Exception:
            return self._heuristic_profile(existing_profile, filename, paper_text)

    def _load(self) -> dict:
        if not self._settings.style_profile_store.exists():
            return {}
        return json.loads(self._settings.style_profile_store.read_text(encoding="utf-8"))

    def _save(self, payload: dict) -> None:
        self._settings.style_profile_store.parent.mkdir(parents=True, exist_ok=True)
        self._settings.style_profile_store.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _heuristic_profile(existing_profile: str, filename: str, paper_text: str) -> str:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", paper_text) if item.strip()]
        avg_sentence = 0.0
        if sentences:
            avg_sentence = round(sum(len(sentence.split()) for sentence in sentences) / len(sentences), 1)
        citation_style = "numeric bracket citations" if re.search(r"\[\d+(,\s*\d+)*\]", paper_text) else "author-year or prose citations"
        formal_markers = sum(paper_text.lower().count(word) for word in ["propose", "demonstrate", "evaluate", "results", "method"])
        new_profile = (
            f"Updated from {filename}. The prose is technical and formal, averaging about {avg_sentence} words per sentence. "
            f"It appears to prefer {citation_style}, uses dense research phrasing, and repeatedly leans on formal reporting verbs "
            f"(sampled {formal_markers} times in the excerpt). Match an academic, evidence-driven tone with clear sectioned progression."
        )
        if existing_profile:
            return f"{existing_profile}\n{new_profile}"
        return new_profile
