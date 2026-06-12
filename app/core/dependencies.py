from qdrant_client import AsyncQdrantClient
from app.config.settings import settings

_qdrant_client: AsyncQdrantClient | None = None

def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_BASE_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60
        )
    return _qdrant_client