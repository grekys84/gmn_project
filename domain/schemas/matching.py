from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class MatchResult(BaseModel):
    reference_source: str | None
    similarity: float
    reference_nodes: int | None

class MatchResponse(BaseModel):
    id: Optional[str] = None
    similarity_percent: Optional[float] = None
    overlay_path: Optional[str] = None
    valid: bool
    processed_at: datetime