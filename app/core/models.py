import uuid
import datetime
from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    id:uuid.UUID
    text:str
    embedding:list[float]
    company:str
    source_url:str
    published_date:datetime.datetime
    score:float
    model_config = ConfigDict(str_strip_whitespace=True,extra='forbid',from_attributes=True)


class JobStatus(BaseModel):
    job_id:uuid.UUID
    status:str
    message:str
    model_config = ConfigDict(str_strip_whitespace=True,extra='forbid',from_attributes=True)
