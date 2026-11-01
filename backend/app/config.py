"""Unified Configuration for Adaptive AI Personal Life & Study Coach."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
MATERIAL_DIR = BASE_DIR / "material"
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    app_name: str = "Adaptive AI Personal Life & Study Coach"
    version: str = "1.0.0"
    api_prefix: str = "/api"
    port: int = 8000
    host: str = "127.0.0.1"

    # LLM Provider Keys (Optional: when missing, smart deterministic fallbacks are used)
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # LLM Models
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"
    openai_model: str = "gpt-4o-mini"

    # Paths
    material_dir: Path = MATERIAL_DIR
    data_dir: Path = DATA_DIR
    db_file: Path = DATA_DIR / "adaptive_coach.db"
    store_file: Path = DATA_DIR / "state_store.json"

    # Adaptive Model Parameters
    risk_alert_threshold: float = 0.60
    alert_consecutive_days: int = 2
    mastery_threshold: float = 65.0
    heavy_lecture_hours: int = 5
    ema_alpha: float = 0.40

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
