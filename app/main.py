import time

from fastapi import FastAPI

from app.model import encode_single, encode_batch, MODEL_NAME, EMBEDDING_DIM, model_load_time
from app.schemas import (
    EncodeRequest,
    EncodeResponse,
    EncodeBatchRequest,
    EncodeBatchResponse,
    HealthResponse,
)

app = FastAPI(title="rubert-mini-frida Inference Service")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_name=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        model_load_time_s=round(model_load_time, 3),
    )


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    start = time.perf_counter()
    embedding = encode_single(req.text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return EncodeResponse(
        embedding=embedding,
        dim=len(embedding),
        time_ms=round(elapsed_ms, 2),
    )


@app.post("/encode_batch", response_model=EncodeBatchResponse)
def encode_batch_endpoint(req: EncodeBatchRequest):
    start = time.perf_counter()
    embeddings = encode_batch(req.texts)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return EncodeBatchResponse(
        embeddings=embeddings,
        count=len(embeddings),
        time_ms=round(elapsed_ms, 2),
    )
