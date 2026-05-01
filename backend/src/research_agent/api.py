from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from research_agent.config import get_settings
from research_agent.runtime import ResearchAgentRuntime
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
)

settings = get_settings()
runtime = ResearchAgentRuntime(settings)
workspace_root = Path(__file__).resolve().parents[3]

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Research agent backend powered by FastAPI and LangGraph.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(workspace_root / "landing.html")


@app.get("/dashboard")
@app.get("/dashboard/")
@app.get("/dashboard.html")
def serve_dashboard() -> FileResponse:
    return FileResponse(workspace_root / "dashboard.html")


@app.get("/research_agent.jsx")
def serve_react_bundle() -> FileResponse:
    return FileResponse(workspace_root / "research_agent.jsx")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return runtime.health()


@app.get(f"{settings.api_prefix}/papers", response_model=PaperListResponse)
def list_papers() -> PaperListResponse:
    return runtime.list_papers()


@app.post(f"{settings.api_prefix}/papers/upload", response_model=PaperUploadResponse)
async def upload_papers(files: list[UploadFile] = File(...)) -> PaperUploadResponse:
    prepared_files: list[tuple[str, bytes]] = []
    for file in files:
        filename = file.filename or "paper.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{filename} is not a PDF.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{filename} is empty.")
        prepared_files.append((filename, content))
    try:
        return runtime.upload_papers(prepared_files)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.delete(f"{settings.api_prefix}/papers", response_model=ClearPapersResponse)
def clear_papers() -> ClearPapersResponse:
    try:
        return runtime.clear_papers()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.delete(f"{settings.api_prefix}/papers/{{paper_id}}", response_model=DeletePaperResponse)
def delete_paper(paper_id: str) -> DeletePaperResponse:
    try:
        response = runtime.delete_paper(paper_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    if not response.deleted:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return response


@app.post(f"{settings.api_prefix}/papers/{{paper_id}}/re-ingest", response_model=PaperSummary)
def re_ingest_paper(paper_id: str) -> PaperSummary:
    try:
        paper = runtime.re_ingest_paper(paper_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@app.get(f"{settings.api_prefix}/style-profile", response_model=StyleProfileResponse)
def style_profile() -> StyleProfileResponse:
    return runtime.style_profile()


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return runtime.chat(request)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post(
    f"{settings.api_prefix}/retrieval/preview",
    response_model=RetrievalPreviewResponse,
)
def retrieval_preview(request: RetrievalPreviewRequest) -> RetrievalPreviewResponse:
    try:
        return runtime.retrieval_preview(request)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
