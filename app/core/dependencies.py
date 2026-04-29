from qdrant_client import QdrantClient

def get_qdrant_client():
    client = QdrantClient(":memory:")
    return client

