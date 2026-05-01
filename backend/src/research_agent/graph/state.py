from typing import Any, TypedDict

from langchain_core.documents import Document
from research_agent.schemas import Mode


class GraphState(TypedDict, total=False):
    session_id: str
    mode: Mode
    message: str
    paper_ids: list[str]
    review_paper_id: str | None
    history: list[dict[str, str]]
    mode_instructions: str
    retrieved_documents: list[Document]
    draft_answer: str
    validated_answer: str
    validation_issues: list[str]
    answer: str
    citations: list[dict[str, Any]]
    attack_vectors: list[dict[str, Any]]
    active_vector_id: str
    debate_history: list[dict[str, Any]]
    debate_summary: str
    skeptic_position: str
    advocate_position: str
    resolution: str
    turn_count: int
    syntheses: dict[str, str]
    vector_verdicts: dict[str, str]
    vector_judgments: dict[str, dict[str, Any]]
    vector_reports: dict[str, dict[str, Any]]
    final_report: dict[str, Any]
    next_speaker: str
    intervention_mode: str
    vectors_remaining: list[str]
    debug: dict[str, Any]
