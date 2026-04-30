from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name:str = 'company_research_tool'
    X_API_KEY:str

    class Config:
        env_file = ".env",
        env_file_encoding = "utf-8"


settings = Settings()