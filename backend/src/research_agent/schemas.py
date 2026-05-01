from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Mode(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COMPARATOR = "comparator"


class Citation(BaseModel):
    paper_id: str
    filename: str
    snippet: str
    chunk_id: str | None = None
    page: int | None = None


class HistoryMessage(BaseModel):
    role: str
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    mode: Mode
    message: str = Field(min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    review_paper_id: str | None = None
    intervention_mode: str | None = None
    history: list[HistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    mode: Mode
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class PaperSummary(BaseModel):
    paper_id: str
    filename: str
    stored_path: str
    text_path: str = ""
    chunk_count: int
    char_count: int = 0
    uploaded_at: str


class PaperListResponse(BaseModel):
    papers: list[PaperSummary] = Field(default_factory=list)


class PaperUploadResponse(BaseModel):
    papers: list[PaperSummary] = Field(default_factory=list)


class RetrievalPreviewRequest(BaseModel):
    query: str = Field(min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalPreviewHit(BaseModel):
    paper_id: str
    filename: str
    snippet: str
    score: float | None = None
    chunk_id: str | None = None


class RetrievalPreviewResponse(BaseModel):
    query: str
    hits: list[RetrievalPreviewHit] = Field(default_factory=list)


class DeletePaperResponse(BaseModel):
    deleted: bool
    paper: PaperSummary | None = None


class ClearPapersResponse(BaseModel):
    deleted_count: int
    cleared_style_profile: bool = False


class StyleProfileResponse(BaseModel):
    active: bool
    profile: str = ""
    source_count: int = 0
    updated_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    graph_ready: bool
    llm_available: bool = False
    indexed_papers: int = 0
