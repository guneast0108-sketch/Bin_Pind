from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database — Alembic 마이그레이션용 (Session Mode Pooler, psycopg2 sync)
    # 직접 연결(5432)은 IPv6 전용이라 일부 환경에서 막힘 → Session Pooler 사용
    database_url: str = "postgresql://pind:pind_local@localhost:5433/pind"

    # Database — FastAPI 런타임용 (Transaction Mode Pooler, asyncpg)
    database_pool_url: str = "postgresql://pind:pind_local@localhost:5433/pind"

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_webhook_secret: str = ""

    # AI
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    google_places_api_key: str = ""

    # Cost caps
    max_video_duration_sec: int = 1800  # 30 min
    max_frames_per_video: int = 60
    max_cost_per_video_usd: float = 0.50

    # App
    debug: bool = False
    log_level: str = "INFO"


settings = Settings()
