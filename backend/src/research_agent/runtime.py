import os
from contextlib import nullcontext
from copy import deepcopy
import re

from research_agent.config import AppSettings
from research_agent.graph.builder import REVIEWER_STATE_KEYS, build_graph, extract_reviewer_state
from research_agent.retrieval.dense import DenseRetriever
from research_agent.retrieval.ingestion import PaperIngestionService
from research_agent.schemas import (
    ChatRequest,
    ChatResponse,
    ClearPapersResponse,
    DeletePaperResponse,
    HealthResponse,
    PaperListResponse,
    PaperSummary,
    PaperUploadResponse,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
    StyleProfileResponse,
    Mode,
)
from research_agent.services.style_memory import StyleMemoryService


class ResearchAgentRuntime:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._langsmith_available = False
        try:
            import langsmith  # noqa: F401

            self._langsmith_available = True
        except Exception:
            self._langsmith_available = False
        self._configure_langsmith()
        self._papers = PaperIngestionService(settings)
        self._retriever = DenseRetriever(settings)
        self._styles = StyleMemoryService(settings)
        self._graph = build_graph()
        self._reviewer_sessions: dict[str, dict[str, dict]] = {}

    @property
    def graph_ready(self) -> bool:
        return self._graph is not None

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=self._settings.app_name,
            environment=self._settings.app_env,
            graph_ready=self.graph_ready,
            llm_available=bool(
                self._settings.groq_api_key
                or self._settings.gemini_api_key
                or self._settings.openrouter_api_key
            ),
            indexed_papers=len(self._papers.list_papers()),
        )

    def list_papers(self) -> PaperListResponse:
        return PaperListResponse(papers=self._papers.list_papers())

    def upload_papers(self, files: list[tuple[str, bytes]]) -> PaperUploadResponse:
        records = []
        for filename, content in files:
            paper = self._papers.ingest_pdf(filename, content)
            paper_text = self._papers.read_paper_text(paper.paper_id)
            self._styles.update_from_paper(paper, paper_text)
            records.append(paper)
        return PaperUploadResponse(papers=records)

    def delete_paper(self, paper_id: str) -> DeletePaperResponse:
        paper = self._papers.delete_paper(paper_id)
        if paper is not None:
            self._drop_reviewer_state_for_paper(paper_id)
        return DeletePaperResponse(deleted=paper is not None, paper=paper)

    def re_ingest_paper(self, paper_id: str) -> PaperSummary | None:
        return self._papers.re_ingest_paper(paper_id)

    def clear_papers(self) -> ClearPapersResponse:
        papers = self._papers.list_papers()
        deleted_count = 0
        for paper in papers:
            deleted = self._papers.delete_paper(paper.paper_id)
            if deleted is not None:
                deleted_count += 1
        cleared_style_profile = self._styles.reset()
        self._reviewer_sessions.clear()
        return ClearPapersResponse(
            deleted_count=deleted_count,
            cleared_style_profile=cleared_style_profile,
        )

    def style_profile(self) -> StyleProfileResponse:
        return self._styles.get_profile()

    def retrieval_preview(self, request: RetrievalPreviewRequest) -> RetrievalPreviewResponse:
        return RetrievalPreviewResponse(
            query=request.query,
            hits=self._retriever.search(
                query=request.query,
                paper_ids=request.paper_ids,
                top_k=request.top_k,
            ),
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        history_payload = [item.model_dump() for item in request.history]
        if request.mode == Mode.COMPARATOR:
            # Comparator runs stateless per turn to avoid cross-paper drift from prior turns.
            history_payload = []
        payload = {
            "session_id": request.session_id,
            "mode": request.mode,
            "message": request.message,
            "paper_ids": request.paper_ids,
            "review_paper_id": request.review_paper_id,
            "intervention_mode": request.intervention_mode,
            "history": history_payload,
            "debug": {},
        }
        if request.mode == Mode.REVIEWER and request.review_paper_id:
            payload.update(self._load_reviewer_state(request.session_id, request.review_paper_id))
            payload["intervention_mode"] = request.intervention_mode
        trace_context = nullcontext()
        if (
            self._settings.langsmith_tracing
            and self._settings.langsmith_api_key
            and self._langsmith_available
        ):
            import langsmith as ls

            client = ls.Client(
                api_key=self._settings.langsmith_api_key,
                api_url=self._settings.langsmith_endpoint,
            )
            trace_context = ls.tracing_context(
                client=client,
                project_name=self._settings.langsmith_project,
                enabled=True,
            )

        try:
            with trace_context:
                result = self._graph.invoke(payload)
        except Exception as error:
            return self._safe_chat_fallback(request=request, error=error)
        if request.mode == Mode.REVIEWER and request.review_paper_id:
            self._save_reviewer_state(
                session_id=request.session_id,
                paper_id=request.review_paper_id,
                state=result,
            )
        return ChatResponse(
            session_id=request.session_id,
            mode=request.mode,
            answer=result["answer"],
            citations=result.get("citations", []),
            debug=result.get("debug", {}),
        )

    def _safe_chat_fallback(self, *, request: ChatRequest, error: Exception) -> ChatResponse:
        paper_scope = request.paper_ids
        if request.mode == Mode.REVIEWER and request.review_paper_id:
            paper_scope = [request.review_paper_id]
        if request.mode == Mode.GLOBAL and not request.paper_ids:
            # For Global mode fallback, avoid forcing retrieval from all indexed papers.
            paper_scope = ["__global_fallback_no_paper_scope__"]
        hits = self._retriever.search(
            query=request.message,
            paper_ids=paper_scope,
            top_k=3,
        )
        citations = [
            {
                "paper_id": hit.paper_id,
                "filename": hit.filename,
                "snippet": hit.snippet,
                "chunk_id": hit.chunk_id,
                "page": None,
            }
            for hit in hits[:2]
        ]

        if request.mode == Mode.GLOBAL and not self._looks_paper_grounded_request(request.message):
            answer = (
                "Model generation is temporarily unavailable in Global mode. "
                "Please retry in a moment."
            )
            citations = []
        elif hits:
            top = self._compact_snippet(hits[0].snippet)
            if request.mode == Mode.REVIEWER:
                answer = (
                    "## Claim Trial Engine\n"
                    "Fallback trial response\n\n"
                    f"### Skeptic\n- Strongest grounded concern signal: {top} [1]\n\n"
                    "### Advocate\n- Partial defense exists, but scope wording should be tightened [1].\n\n"
                    "### Evidence-only Judge\n- Verdict: contested\n- Rationale: fallback context is too shallow for a decisive ruling [1].\n\n"
                    "### Rewrite Compiler Card\n"
                    "Target Section: contribution/claim paragraph\n"
                    "Patch Instruction: add one concrete metric comparator and explicit scope boundary.\n"
                )
            elif request.mode == Mode.COMPARATOR and len(hits) >= 2:
                second = self._compact_snippet(hits[1].snippet)
                answer = (
                    "## Papers Compared\n"
                    "- Paper 1\n- Paper 2\n\n"
                    "## Claim Matrix\n"
                    f"- Paper 1 claim signal: {top} [1]\n"
                    f"- Paper 2 claim signal: {second} [2]\n\n"
                    "## Conflict Map\n"
                    "- Agreements/contradictions are uncertain in runtime-safe fallback mode.\n\n"
                    "## Decision By Use Case\n"
                    "- Use this as a provisional grounding snapshot only; rerun for full comparator synthesis."
                )
            else:
                answer = f"Based on the uploaded papers: {top} [1]"
        else:
            answer = (
                "I could not retrieve enough grounded evidence for that request. "
                "Try a more specific paper-focused question."
            )

        debug = {
            "response_stage": "runtime_safe_fallback",
            "model_fallback": True,
            "error_type": type(error).__name__,
            "model_error": str(error)[:180],
        }
        return ChatResponse(
            session_id=request.session_id,
            mode=request.mode,
            answer=answer,
            citations=citations,
            debug=debug,
        )

    @staticmethod
    def _compact_snippet(text: str, max_chars: int = 420) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[: max_chars - 3].rstrip()}..."

    @staticmethod
    def _looks_paper_grounded_request(message: str) -> bool:
        lower = (message or "").lower()
        markers = (
            "paper",
            "uploaded",
            ".pdf",
            "according to",
            "in the paper",
            "authors",
            "dataset",
            "benchmark",
            "section",
            "table",
            "figure",
        )
        return any(marker in lower for marker in markers)

    def _configure_langsmith(self) -> None:
        if not self._settings.langsmith_tracing:
            return
        if self._settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self._settings.langsmith_api_key
        if self._settings.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = self._settings.langsmith_project
        if self._settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = self._settings.langsmith_endpoint
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    def _reviewer_key(self, session_id: str, paper_id: str) -> str:
        return f"{session_id}::{paper_id}"

    def _load_reviewer_state(self, session_id: str, paper_id: str) -> dict:
        store = self._reviewer_sessions.get(self._reviewer_key(session_id, paper_id), {})
        return deepcopy(store)

    def _save_reviewer_state(self, *, session_id: str, paper_id: str, state: dict) -> None:
        extracted = extract_reviewer_state(state)
        if not extracted:
            return
        key = self._reviewer_key(session_id, paper_id)
        existing = self._reviewer_sessions.get(key, {})
        merged = deepcopy(existing)
        for item_key in REVIEWER_STATE_KEYS:
            if item_key in extracted:
                merged[item_key] = extracted[item_key]
        self._reviewer_sessions[key] = merged

    def _drop_reviewer_state_for_paper(self, paper_id: str) -> None:
        suffix = f"::{paper_id}"
        to_remove = [key for key in self._reviewer_sessions.keys() if key.endswith(suffix)]
        for key in to_remove:
            self._reviewer_sessions.pop(key, None)
