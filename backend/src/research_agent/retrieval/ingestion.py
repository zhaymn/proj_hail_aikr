from datetime import UTC, datetime
import json
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from research_agent.config import AppSettings
from research_agent.retrieval.catalog import PaperCatalog
from research_agent.retrieval.chunking import SemanticPaperChunker
from research_agent.retrieval.dense import DenseRetriever
from research_agent.schemas import PaperSummary


class PaperIngestionService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._catalog = PaperCatalog(settings)
        self._dense = DenseRetriever(settings)
        self._chunker = SemanticPaperChunker(settings)

    def list_papers(self) -> list[PaperSummary]:
        return self._catalog.list_papers()

    def get_paper(self, paper_id: str) -> PaperSummary | None:
        return self._catalog.get_paper(paper_id)

    def read_paper_text(self, paper_id: str) -> str:
        paper = self.get_paper(paper_id)
        if paper is None:
            raise FileNotFoundError(f"Unknown paper id: {paper_id}")
        text_path = Path(paper.text_path)
        if not text_path.exists():
            return ""
        return text_path.read_text(encoding="utf-8")

    def ingest_pdf(self, filename: str, content: bytes) -> PaperSummary:
        paper_id = uuid4().hex
        safe_name = Path(filename or "paper.pdf").name

        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._settings.paper_text_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._settings.uploads_dir / f"{paper_id}.pdf"
        target_path.write_bytes(content)

        pages = PyPDFLoader(str(target_path)).load()
        full_text = "\n\n".join(page.page_content.strip() for page in pages if page.page_content.strip())
        text_path = self._settings.paper_text_dir / f"{paper_id}.txt"
        text_path.write_text(full_text, encoding="utf-8")
        chunks = self._chunk_pages(pages, paper_id=paper_id, filename=safe_name)
        if not chunks:
            raise ValueError(f"No text could be extracted from {safe_name}.")

        self._write_chunk_manifest(paper_id=paper_id, chunks=chunks)
        self._upsert_dense_chunks(chunks)

        record = PaperSummary(
            paper_id=paper_id,
            filename=safe_name,
            stored_path=str(target_path),
            text_path=str(text_path),
            chunk_count=len(chunks),
            char_count=len(full_text),
            uploaded_at=datetime.now(UTC).isoformat(),
        )
        self._catalog.upsert(record)
        return record

    def delete_paper(self, paper_id: str) -> PaperSummary | None:
        paper = self._catalog.get(paper_id)
        if paper is None:
            return None

        self._dense.delete_paper(paper_id)
        deleted = self._catalog.delete(paper_id)
        if deleted is None:
            return None
        self._delete_chunk_manifest(paper_id)

        for path_str in [paper.stored_path, paper.text_path]:
            if not path_str:
                continue
            path = Path(path_str)
            if path.exists():
                path.unlink()
        return deleted

    def re_ingest_paper(self, paper_id: str) -> PaperSummary | None:
        paper = self._catalog.get(paper_id)
        if paper is None:
            return None

        # Delete old chunks
        self._dense.delete_paper(paper_id)

        # Read text
        text_path = Path(paper.text_path)
        if not text_path.exists():
            return None
        full_text = text_path.read_text(encoding="utf-8")

        # Create documents from text (simulate pages)
        pages = [Document(page_content=full_text, metadata={"source": paper.filename})]

        # Chunk
        chunks = self._chunk_pages(pages, paper_id=paper_id, filename=paper.filename)

        self._write_chunk_manifest(paper_id=paper_id, chunks=chunks)
        self._upsert_dense_chunks(chunks)

        # Update catalog with new counts
        updated_paper = paper.model_copy(update={
            "chunk_count": len(chunks),
            "char_count": len(full_text),
        })
        self._catalog.upsert(updated_paper)
        return updated_paper

    def _chunk_pages(
        self,
        pages: list[Document],
        *,
        paper_id: str,
        filename: str,
    ) -> list[Document]:
        return self._chunker.chunk_pages(
            pages,
            paper_id=paper_id,
            filename=filename,
        )

    def _write_chunk_manifest(self, *, paper_id: str, chunks: list[Document]) -> None:
        self._settings.chunk_manifest_dir.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, object]] = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            payload.append(
                {
                    "paper_id": str(metadata.get("paper_id", paper_id)),
                    "filename": str(metadata.get("filename", "unknown.pdf")),
                    "chunk_id": str(metadata.get("chunk_id", "")),
                    "chunk_index": metadata.get("chunk_index"),
                    "page": metadata.get("page"),
                    "text": chunk.page_content,
                }
            )

        manifest_path = self._settings.chunk_manifest_dir / f"{paper_id}.json"
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _delete_chunk_manifest(self, paper_id: str) -> None:
        manifest_path = self._settings.chunk_manifest_dir / f"{paper_id}.json"
        if manifest_path.exists():
            manifest_path.unlink()

    def _upsert_dense_chunks(self, chunks: list[Document]) -> None:
        try:
            self._dense.upsert_documents(chunks)
        except Exception as error:
            if self._is_recoverable_dense_error(error):
                # Sparse retrieval still works from local chunk manifests.
                return
            raise

    def _is_recoverable_dense_error(self, error: Exception) -> bool:
        provider = (self._settings.embedding_provider or "").lower().strip()
        message = str(error).lower()
        if "pinecone_api_key is required" in message:
            return True
        if provider != "local":
            return False
        marker_phrases = (
            "pinecone",
            "name or service not known",
            "temporary failure in name resolution",
            "connection error",
            "connection reset",
            "timed out",
            "service unavailable",
            "dns",
        )
        return any(marker in message for marker in marker_phrases)
