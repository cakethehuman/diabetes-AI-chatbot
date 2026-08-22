from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LLM_API_KEY: str | None = None
    LLM_MODEL: str | None = None
    LLM_BASE_URL: str | None = None
    LLM_PROVIDER: str | None = None
    JINA_API_KEY: str | None = None
    JINA_MODEL: str | None = None
    JINA_DIMENSIONS : int | None = 1024
    AGENT_TOP_K: int | None = 5
    SEMSEARCH_DB: str | None = '.chroma_db'
    SEMSEARCH_COLLECTION: str | None = 'docs'
    SEMSEARCH_UPLOADS: str | None = 'data'
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    