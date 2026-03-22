import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app_onnx.model import encode_batch, MODEL_NAME, EMBEDDING_DIM, model_load_time
from app.schemas import (
    EncodeRequest,
    EncodeResponse,
    EncodeBatchRequest,
    EncodeBatchResponse,
    HealthResponse,
)
from app_batched.batcher import DynamicBatcher

MAX_BATCH_SIZE = 32
MAX_WAIT_MS = 50.0

batcher = DynamicBatcher(max_batch_size=MAX_BATCH_SIZE, max_wait_ms=MAX_WAIT_MS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    batcher.start()
    yield
    await batcher.stop()


app = FastAPI(title="rubert-mini-frida ONNX + Dynamic Batching", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_name=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        model_load_time_s=round(model_load_time, 3),
    )


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest):
    start = time.perf_counter()
    embedding = await batcher.submit(req.text)
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
