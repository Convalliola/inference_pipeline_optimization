#### Как запускать бенчмарки:

**- Для бейзлайна:**
    1. Запуск сервера: python -m uvicorn app.main:app --port 8000
    2. Запуск бенчмарков: python -m benchmark.run_benchmark --port 8000 --output results.json

**- Для ONNX:**
    1. python -m scripts.export_onnx
    2. python -m uvicorn app_onnx.main:app --host 0.0.0.0 --port 8001
    3. python -m benchmark.run_benchmark --port 8001 --output results_onnx.json
    
**- Для батчинга:**
    1. python -m uvicorn app_batched.main:app --host 0.0.0.0 --port 8002
    2. python -m benchmark.run_benchmark --port 8002 --output results_batched.json