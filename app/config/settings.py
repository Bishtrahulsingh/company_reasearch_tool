from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = 'company_research_tool'
    X_API_KEY: str
    GITHUB_TOKEN: str
    QDRANT_BASE_URL: str
    QDRANT_API_KEY: str
    WEB_SEARCH_COST:float
    GROQ_API_KEY:str
    LANGFUSE_SECRET_KEY:str
    LANGFUSE_PUBLIC_KEY:str
    LANGFUSE_BASE_URL:str
    COLLECTION_NAME:str="company_research"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()