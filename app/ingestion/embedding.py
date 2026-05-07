from typing import List
from app.core.models import Chunk
from fastembed import TextEmbedding, SparseTextEmbedding

dense_embedding_model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def embed_chunks(chunks:List[Chunk],batch_size:int=18)->List[Chunk]:
    if not chunks:
        return []

    total_len = len(chunks)
    for i in range(0,total_len,batch_size):
        start = i
        end = min(i+batch_size,total_len)
        text = [chunk.text for chunk in chunks[start:end]]
        embeddings = list(dense_embedding_model.embed(text))
        sparse_embedding = list(sparse_embedding_model.embed(text))

        for j in range(start,end):
            chunks[j].embedding = embeddings[j-start]
            sp = sparse_embedding[j-start]
            chunks[j].sparse_embedding = {
                'indices':sp.indices.tolist(),
                'values':sp.values.tolist()
            }

    return chunks