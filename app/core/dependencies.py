from qdrant_client import AsyncQdrantClient

from app.config.settings import settings


def get_qdrant_client():

    # client = AsyncQdrantClient(
    #     url=settings.QDRANT_BASE_URL,
    #     api_key=settings.QDRANT_API_KEY,
    # )
    client = AsyncQdrantClient(":memory:")


    return client


