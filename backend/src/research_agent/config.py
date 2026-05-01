from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = BACKEND_ROOT / "storage"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Prefer checked-in backend/.env for local reliability.
        # Some shells keep empty API env vars in process scope, which can mask real .env values.
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )

    app_name: str = "Research Agent API"
    app_env: str = "development"
    api_prefix: str = "/api"

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    generation_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("GROQ_MODEL", "MODEL_NAME"),
    )
    openrouter_generation_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="OPENROUTER_MODEL",
    )
    generation_provider: str = Field(
        default="auto",
        alias="GENERATION_PROVIDER",
    )
    generation_fallback_order: str = Field(
        default="groq,openrouter,gemini",
        alias="GENERATION_FALLBACK_ORDER",
    )
    generation_provider_cooldown_seconds: int = Field(
        default=600,
        alias="GENERATION_PROVIDER_COOLDOWN_SECONDS",
    )
    gemini_generation_model: str = Field(
        default="gemini-2.0-flash",
        alias="GEMINI_MODEL",
    )
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="research-agent", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )

    storage_root: Path = STORAGE_ROOT
    uploads_dir: Path = STORAGE_ROOT / "uploads"
    paper_text_dir: Path = STORAGE_ROOT / "papers"
    style_profile_store: Path = STORAGE_ROOT / "style_profiles.json"
    paper_catalog_path: Path = STORAGE_ROOT / "papers.json"
    chunk_manifest_dir: Path = STORAGE_ROOT / "chunks"

    pinecone_index_name: str = "research-agent"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_provider: str = "local"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 96
    pinecone_upsert_batch_size: int = 64
    chunk_size: int = 900
    chunk_overlap: int = 150
    semantic_unit_max_chars: int = 360
    semantic_similarity_floor: float = 0.22
    retrieval_top_k: int = 8
    hybrid_dense_top_k: int = 24
    hybrid_sparse_top_k: int = 24
    hybrid_rrf_k: int = 20
    hybrid_dense_weight: float = 0.55
    hybrid_sparse_weight: float = 0.45
    rerank_top_n: int = 5
    reviewer_max_turns: int = 8
    reviewer_warning_turn: int = 6
    reviewer_turns_per_response: int = 1
    reviewer_attack_vector_count: int = 4
    conversation_window: int = 8
    paragraph_min_chars: int = 60
    max_review_chars: int = 14000


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
