import datetime
import hashlib
import uuid
from typing import Dict,List,Any
from app.core.models import Chunk

def get_fingerprint_id(text: str, company: str, source_url: str) -> str:
    fingerprint = f"{company}::{source_url}::{text[:200]}"
    md5 = hashlib.md5(fingerprint.encode('utf-8')).hexdigest()
    return str(uuid.UUID(md5))

def chunk_text(text:str,published_date:datetime.datetime,company:str,source_url:str, size:int=500, overlap:int=50)->List[Chunk]:
    chunks = []

    step = size-overlap

    if step <= 0:
        raise Exception('chunk size must be greater than 0')

    for i in range(0,len(text),step):
         start = i
         end = min(len(text),i+size)

         chunk:Chunk = Chunk(
             id = get_fingerprint_id(text[i:min(start+size,len(text))], company, source_url),
             text= text[start:end],
             embedding=[],
             company=company,
             source_url=source_url,
             published_date=published_date or datetime.datetime.now(datetime.timezone.utc),
             score=0.0
         )
         chunks.append(chunk)
    return chunks
