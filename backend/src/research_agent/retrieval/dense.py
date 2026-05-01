from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from research_agent.config import AppSettings
from research_agent.retrieval.embeddings import EmbeddingService
from research_agent.retrieval.sparse import SparseRetriever
from research_agent.schemas import RetrievalPreviewHit


class DenseRetriever:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = Pinecone(api_key=settings.pinecone_api_key) if settings.pinecone_api_key else None
        self._embeddings = EmbeddingService(settings)
        self._sparse = SparseRetriever(settings)

    def ensure_index(self) -> None:
        if self._client is None:
            raise RuntimeError("PINECONE_API_KEY is required before Pinecone operations can run.")
        if self._index_exists():
            return
        self._client.create_index(
            name=self._settings.pinecone_index_name,
            dimension=self._settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self._settings.pinecone_cloud,
                region=self._settings.pinecone_region,
            ),
            deletion_protection="disabled",
        )

    def search(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalPreviewHit]:
        pairs = self.retrieve(query=query, paper_ids=paper_ids, top_k=top_k)

        hits: list[RetrievalPreviewHit] = []
        for document, score in pairs:
            metadata = document.metadata or {}
            hits.append(
                RetrievalPreviewHit(
                    paper_id=str(metadata.get("paper_id", "")),
                    filename=str(metadata.get("filename", "unknown.pdf")),
                    snippet=document.page_content[:500],
                    score=float(score) if score is not None else None,
                    chunk_id=str(metadata.get("chunk_id")) if metadata.get("chunk_id") else None,
                )
            )
        return hits

    def retrieve(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[Document, float | None]]:
        target_top_k = max(1, top_k or self._settings.retrieval_top_k)
        dense_top_k = max(target_top_k, int(self._settings.hybrid_dense_top_k))
        sparse_top_k = max(target_top_k, int(self._settings.hybrid_sparse_top_k))

        dense_pairs = self._retrieve_dense(query=query, paper_ids=paper_ids, top_k=dense_top_k)
        sparse_pairs = self._sparse.retrieve(query=query, paper_ids=paper_ids, top_k=sparse_top_k)

        if dense_pairs and sparse_pairs:
            fused = self._fuse_rankings(dense_pairs=dense_pairs, sparse_pairs=sparse_pairs, limit=target_top_k)
            if fused:
                return fused

        if dense_pairs:
            return dense_pairs[:target_top_k]
        if sparse_pairs:
            return [(document, score) for document, score in sparse_pairs[:target_top_k]]
        return []

    def _retrieve_dense(
        self,
        *,
        query: str,
        paper_ids: list[str] | None,
        top_k: int,
    ) -> list[tuple[Document, float | None]]:
        if self._client is None or not self._index_exists():
            return []

        query_vector = self._embeddings.embed_query(query)
        pinecone_filter = None
        if paper_ids:
            pinecone_filter = {"paper_id": {"$in": paper_ids}}

        response = self.index().query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self._settings.pinecone_namespace,
            filter=pinecone_filter,
        )
        matches = getattr(response, "matches", []) or []
        pairs: list[tuple[Document, float | None]] = []
        for match in matches:
            metadata = getattr(match, "metadata", {}) or {}
            page_content = str(metadata.get("text", ""))
            if not page_content:
                continue
            document = Document(page_content=page_content, metadata=metadata)
            pairs.append((document, getattr(match, "score", None)))
        return pairs

    def _fuse_rankings(
        self,
        *,
        dense_pairs: list[tuple[Document, float | None]],
        sparse_pairs: list[tuple[Document, float]],
        limit: int,
    ) -> list[tuple[Document, float]]:
        rrf_k = max(1, int(self._settings.hybrid_rrf_k))
        dense_weight = max(0.0, float(self._settings.hybrid_dense_weight))
        sparse_weight = max(0.0, float(self._settings.hybrid_sparse_weight))
        provider = (self._settings.embedding_provider or "auto").lower().strip()
        if provider == "local":
            # Local hash embeddings are fast but weak semantically. Favor sparse ranking harder.
            dense_weight = min(dense_weight, 0.2)
            sparse_weight = max(sparse_weight, 0.8)
        if dense_weight == 0 and sparse_weight == 0:
            dense_weight = 0.5
            sparse_weight = 0.5

        merged: dict[str, dict[str, Any]] = {}

        for rank, (document, score) in enumerate(dense_pairs, start=1):
            key = self._document_key(document)
            state = merged.setdefault(
                key,
                {
                    "document": document,
                    "hybrid_score": 0.0,
                    "dense_score": score,
                    "sparse_score": None,
                    "dense_rank": None,
                    "sparse_rank": None,
                },
            )
            state["dense_rank"] = rank
            state["dense_score"] = score
            state["hybrid_score"] += dense_weight * (1.0 / (rrf_k + rank))

        for rank, (document, score) in enumerate(sparse_pairs, start=1):
            key = self._document_key(document)
            state = merged.setdefault(
                key,
                {
                    "document": document,
                    "hybrid_score": 0.0,
                    "dense_score": None,
                    "sparse_score": score,
                    "dense_rank": None,
                    "sparse_rank": None,
                },
            )
            state["sparse_rank"] = rank
            state["sparse_score"] = score
            state["hybrid_score"] += sparse_weight * (1.0 / (rrf_k + rank))

        ranked = sorted(merged.values(), key=lambda item: float(item["hybrid_score"]), reverse=True)
        fused: list[tuple[Document, float]] = []
        for item in ranked[:limit]:
            source_document = item["document"]
            metadata = dict(source_document.metadata or {})
            metadata["dense_score"] = item.get("dense_score")
            metadata["sparse_score"] = item.get("sparse_score")
            metadata["dense_rank"] = item.get("dense_rank")
            metadata["sparse_rank"] = item.get("sparse_rank")
            metadata["retrieval"] = "hybrid"
            fused_document = Document(page_content=source_document.page_content, metadata=metadata)
            fused.append((fused_document, float(item["hybrid_score"])))
        return fused

    def upsert_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self.ensure_index()
        embeddings = self._embeddings.embed_documents([document.page_content for document in documents])
        if len(embeddings) != len(documents):
            raise RuntimeError("Embedding count mismatch while preparing Pinecone vectors.")

        vectors: list[dict[str, Any]] = []
        for document, embedding in zip(documents, embeddings):
            metadata = self._sanitize_metadata(dict(document.metadata or {}))
            chunk_id = str(metadata.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            metadata["text"] = document.page_content
            vectors.append(
                {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )
        if not vectors:
            return

        for batch in self._batch(vectors, self._settings.pinecone_upsert_batch_size):
            self.index().upsert(vectors=batch, namespace=self._settings.pinecone_namespace)

    def delete_paper(self, paper_id: str) -> None:
        if self._client is None or not self._index_exists():
            return
        self.index().delete(
            filter={"paper_id": {"$eq": paper_id}},
            namespace=self._settings.pinecone_namespace,
        )

    def index(self):
        self.ensure_index()
        return self._client.Index(self._settings.pinecone_index_name)

    def _index_exists(self) -> bool:
        if self._client is None:
            return False
        try:
            if hasattr(self._client, "has_index"):
                return bool(self._client.has_index(self._settings.pinecone_index_name))
            indexes = self._client.list_indexes()
            if hasattr(indexes, "names"):
                return self._settings.pinecone_index_name in indexes.names()
            if isinstance(indexes, list):
                names = []
                for item in indexes:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict) and item.get("name"):
                        names.append(item["name"])
                return self._settings.pinecone_index_name in names
        except Exception:
            return False
        return False

    @staticmethod
    def _document_key(document: Document) -> str:
        metadata = document.metadata or {}
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        if chunk_id:
            return chunk_id
        paper_id = str(metadata.get("paper_id", "")).strip()
        page = str(metadata.get("page", "")).strip()
        return f"{paper_id}:{page}:{hash(document.page_content)}"

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
                continue
            if isinstance(value, list):
                cleaned[key] = [str(item) for item in value if item is not None]
                continue
            cleaned[key] = str(value)
        return cleaned

    @staticmethod
    def _batch(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
        if size <= 0:
            return [items]
        return [items[index : index + size] for index in range(0, len(items), size)]
