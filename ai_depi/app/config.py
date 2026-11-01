from functools import lru_cache
import re
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

# Runtime defaults are overridden by environment variables and the local .env file.
# Keep the embedding model and Chroma collection identity aligned with the vectors
# already stored in that collection.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # Groq is the primary generator; Ollama is used only when the primary fails.
    llm_provider: str = "groq"
    llm_fallback_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    # Ingestion and query-time search must use the same provider/model and version.
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Persistent Chroma location and collection name determine which indexed corpus is used.
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "learning_materials_minilm"
    sample_data_dir: str = "sample_data"
    auto_ingest_pdfs: bool = True
    # One production strategy keeps endpoint retrieval behavior consistent.
    retrieval_strategy: Literal["hybrid_cross_encoder"] = "hybrid_cross_encoder"
    dense_candidate_k: int = Field(default=30, ge=1, le=1000)
    sparse_candidate_k: int = Field(default=30, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=10000)
    rerank_candidate_k: int = Field(default=30, ge=1, le=1000)
    # FINAL_TOP_K is preferred; TOP_K remains an environment compatibility alias.
    top_k: int = Field(default=5, ge=1, le=100, validation_alias=AliasChoices("FINAL_TOP_K", "TOP_K"))
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str = "auto"
    cross_encoder_batch_size: int = Field(default=16, ge=1, le=256)
    cross_encoder_score_offset: float = Field(default=5.0, ge=-20.0, le=20.0)
    max_context_chars: int = Field(default=12000, ge=1, le=1000000)  # Hard bound on evidence text included in a prompt.
    min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0)  # Sigmoid-normalized Cross-Encoder floor in [0, 1].
    # Chunk overlap keeps text near a boundary available to both neighboring chunks.
    chunk_chars: int = Field(default=1000, ge=1, le=100000)  # Larger chunks preserve context but yield fewer retrieval units.
    chunk_overlap_chars: int = Field(default=120, ge=0, le=99999)  # Shared text reduces loss of facts split at chunk edges.
    # Generation controls apply to provider calls; retries are for invalid structured output.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)  # Low randomness for consistent structured learning output.
    max_retries: int = Field(default=2, ge=0, le=5)  # Additional attempts after invalid JSON/schema/count output.
    claim_judge_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    request_timeout: float = Field(default=60.0, gt=0.0, le=600.0)  # Per-provider network request timeout in seconds.
    max_input_chars: int = Field(default=20000, ge=1, le=1000000)  # UTF-8 decoded request-body character limit.
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if self.chunk_overlap_chars >= self.chunk_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_chars")
        return self

    @field_validator("cross_encoder_device")
    @classmethod
    def validate_cross_encoder_device(cls, value):
        normalized = value.strip().casefold()
        if not re.fullmatch(r"(?:auto|cpu|mps|cuda(?::\d+)?)", normalized):
            raise ValueError("cross_encoder_device must be auto, cpu, mps, cuda, or cuda:<index>")
        return normalized

@lru_cache
def get_settings() -> Settings:
    return Settings()
