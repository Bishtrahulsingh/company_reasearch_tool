import uuid
import datetime
from typing import Union, Dict, List

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    id:Union[uuid.UUID,str]
    text:str
    embedding:list[float]=[]
    sparse_embedding: Dict[str, List] = {}
    company:str
    source_url:str=''
    published_date:datetime.datetime=datetime.datetime.now()
    score:float=0.0
    model_config = ConfigDict(str_strip_whitespace=True,extra='forbid',from_attributes=True)


class JobStatus(BaseModel):
    job_id:str
    status:str
    message:str
    model_config = ConfigDict(str_strip_whitespace=True,extra='forbid',from_attributes=True)

class WebSearchResult(BaseModel):
    summary:str
    items:list
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid', from_attributes=True)
