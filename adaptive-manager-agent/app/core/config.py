from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Adaptive Manager Agent"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./adaptive_manager.db"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    risk_alert_threshold: float = 0.60
    alert_consecutive_days: int = 2
    mastery_threshold: float = 60.0
    heavy_lecture_hours: float = 5.0
    ema_alpha: float = 0.5
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

settings = Settings()
