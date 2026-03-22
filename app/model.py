import time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sergeyzh/rubert-mini-frida"
EMBEDDING_DIM = 312

_load_start = time.perf_counter()
model = SentenceTransformer(MODEL_NAME, device="cpu")
model_load_time = time.perf_counter() - _load_start


def encode_single(text: str) -> list[float]:
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def encode_batch(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
