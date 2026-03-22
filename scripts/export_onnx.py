"""
Export rubert-mini-frida PyTorch model to ONNX format.

Usage:
    python -m scripts.export_onnx
"""

import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sergeyzh/rubert-mini-frida"
OUTPUT_DIR = Path("models")
OUTPUT_PATH = OUTPUT_DIR / "model.onnx"
OPSET_VERSION = 14


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    dummy_texts = [
        "Тестовое предложение для экспорта модели.",
        "Второе предложение для корректного определения динамических осей.",
    ]
    dummy_inputs = tokenizer(
        dummy_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"},
    }

    print(f"Exporting to ONNX (opset {OPSET_VERSION}), dummy batch_size={input_ids.shape[0]}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(OUTPUT_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )

    import onnx
    from onnx import shape_inference

    print("Running shape inference to propagate dynamic dimensions...")
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, str(OUTPUT_PATH))

    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")

    onnx_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"ONNX model saved to {OUTPUT_PATH} ({onnx_size_mb:.1f} MB)")

    for p in Path.home().rglob("model.safetensors"):
        if "rubert-mini-frida" in str(p):
            pytorch_size_mb = os.path.getsize(p) / (1024 * 1024)
            print(f"PyTorch weights: {pytorch_size_mb:.1f} MB")
            print(f"ONNX model:     {onnx_size_mb:.1f} MB")
            break
    else:
        print("(Could not locate cached PyTorch weights for size comparison)")


if __name__ == "__main__":
    main()
