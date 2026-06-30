from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str = "change-me"
    database_url: str = "postgresql+asyncpg://findwithin_user:findwithin_password@db:5432/findwithin"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 64
    top_k_default: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
