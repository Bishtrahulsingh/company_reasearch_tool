from qdrant_client import AsyncQdrantClient


def get_qdrant_client():
    client = AsyncQdrantClient(":memory:")
    return client

