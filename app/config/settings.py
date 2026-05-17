from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = 'company_research_tool'
    X_API_KEY: str
    GITHUB_TOKEN: str
    QDRANT_BASE_URL: str
    QDRANT_API_KEY: str
    WEB_SEARCH_COST:float

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()