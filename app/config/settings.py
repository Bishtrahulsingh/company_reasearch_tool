from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name:str = 'company_research_tool'


settings = Settings()