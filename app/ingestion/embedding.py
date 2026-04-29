from typing import List
from app.core.models import Chunk
from fastembed import TextEmbedding

embedding_model = TextEmbedding()

def embed_chunks(chunks,batch_size:int=18)->List[Chunk]:
    if not chunks:
        return []

    total_len = len(chunks)
    for i in range(0,total_len,batch_size):
        start = i
        end = min(i+batch_size,total_len)
        text = [chunk.text for chunk in chunks[start:end]]
        embeddings = list(embedding_model.embed(text))

        for j in range(start,end):
            chunks[j].embedding = embeddings[j-start]

    return chunks