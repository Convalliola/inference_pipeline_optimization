from pydantic import BaseModel, Field


class EncodeRequest(BaseModel):
    text: str


class EncodeResponse(BaseModel):
    embedding: list[float]
    dim: int
    time_ms: float


class EncodeBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EncodeBatchResponse(BaseModel):
    embeddings: list[list[float]]
    count: int
    time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    embedding_dim: int
    model_load_time_s: float
