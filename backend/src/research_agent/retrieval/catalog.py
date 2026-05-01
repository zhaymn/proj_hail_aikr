import json
from pathlib import Path

from research_agent.config import AppSettings
from research_agent.schemas import PaperSummary


class PaperCatalog:
    def __init__(self, settings: AppSettings) -> None:
        self._path = settings.paper_catalog_path

    def list_papers(self) -> list[PaperSummary]:
        payload = self._load_payload()
        return [PaperSummary.model_validate(item) for item in payload]

    def get_paper(self, paper_id: str) -> PaperSummary | None:
        return next((paper for paper in self.list_papers() if paper.paper_id == paper_id), None)

    def get(self, paper_id: str) -> PaperSummary | None:
        return self.get_paper(paper_id)

    def upsert(self, paper: PaperSummary) -> None:
        papers = self.list_papers()
        by_id = {item.paper_id: item for item in papers}
        by_id[paper.paper_id] = paper
        self._save_payload([item.model_dump() for item in by_id.values()])

    def delete(self, paper_id: str) -> PaperSummary | None:
        papers = self.list_papers()
        removed = next((paper for paper in papers if paper.paper_id == paper_id), None)
        if removed is None:
            return None
        remaining = [paper for paper in papers if paper.paper_id != paper_id]
        self._save_payload([item.model_dump() for item in remaining])
        return removed

    def _load_payload(self) -> list[dict]:
        if not self._path.exists():
            return []
        content = self._path.read_text(encoding="utf-8")
        return json.loads(content)

    def _save_payload(self, payload: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
