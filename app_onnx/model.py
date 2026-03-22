import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_NAME = "sergeyzh/rubert-mini-frida"
EMBEDDING_DIM = 312
ONNX_PATH = str(Path(__file__).resolve().parent.parent / "models" / "model.onnx")

_load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.enable_mem_pattern = False
session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)

model_load_time = time.perf_counter() - _load_start


def _mean_pool(hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    sum_embeddings = np.sum(hidden_state * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms


def _encode(texts: list[str]) -> np.ndarray:
    inputs = tokenizer(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    outputs = session.run(None, ort_inputs)
    last_hidden_state = outputs[0]
    pooled = _mean_pool(last_hidden_state, inputs["attention_mask"])
    return _l2_normalize(pooled)


def encode_single(text: str) -> list[float]:
    embeddings = _encode([text])
    return embeddings[0].tolist()


def encode_batch(texts: list[str]) -> list[list[float]]:
    embeddings = _encode(texts)
    return embeddings.tolist()
