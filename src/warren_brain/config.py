from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection: "openai" or "anthropic"
    llm_provider: str = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_base_url: str = ""

    # Anthropic / Claude
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # External APIs
    finnhub_api_key: str = ""
    massive_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # Storage — SQLite default; set DB_* vars to use PostgreSQL/Supabase
    database_url: str = "sqlite:///warren_brain.db"

    # PostgreSQL / Supabase connection parts (override database_url when db_host is set)
    db_host: str = ""
    db_user: str = "postgres"
    db_password: str = ""   # raw — quote_plus applied in database.py
    db_name: str = "postgres"
    db_port: int = 5432

    # Agent weights (must sum to 1.0)
    weight_fundamentals: float = 0.30
    weight_technicals: float = 0.20
    weight_sentiment: float = 0.15
    weight_thirteen_f: float = 0.15
    weight_ontology: float = 0.10
    weight_risk: float = 0.10

    # Data fetch settings
    price_history_years: int = 5
    news_lookback_days: int = 30


settings = Settings()
