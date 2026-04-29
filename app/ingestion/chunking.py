import datetime
import uuid
from typing import Dict,List,Any

from app.core.models import Chunk


def chunk_text(text, size, overlap)->List[Chunk]:
    chunks = []

    step = size-overlap

    if step <= 0:
        raise Exception('chunk size must be greater than 0')

    for i in range(0,len(text),step):
         start = i
         end = min(len(text),i+size)

         chunk:Chunk = Chunk(
             id = str(uuid.uuid4()),
             text= text[start:end],
             embedding=[],
             company='',
             source_url='',
             published_date= datetime.datetime.now(datetime.timezone.utc),
             score=0.0
         )
         chunks.append(chunk)
    return chunks
