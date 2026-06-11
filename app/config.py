from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str = ""
    database_url: str = "sqlite:///./sales_agent.db"
    gemini_model: str = "gemini-2.5-flash"
    app_name: str = "Sales Assistant Agent"
    app_version: str = "1.0.0"
    # Flag threshold: confidence below this triggers human escalation
    flag_threshold: float = 0.65

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()