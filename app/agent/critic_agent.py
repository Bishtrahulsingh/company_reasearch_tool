from pydantic import BaseModel

class CriticVerdict(BaseModel):
    approved: bool
    issues: list[str]
    confidence: float


