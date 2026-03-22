"""
Бенчамарковый скрипт для инференса модели rubert-mini-frida.
"""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil
import requests

SAMPLE_TEXT = "Привет, это тестовое предложение для бенчмарка модели эмбеддингов."
SAMPLE_TEXTS_8 = [
    "Привет, это тестовое предложение для бенчмарка модели эмбеддингов.",
    "Машинное обучение позволяет компьютерам учиться на данных.",
    "Сегодня хорошая погода для прогулки в парке.",
    "Нейронные сети используются в задачах обработки естественного языка.",
    "Москва — столица Российской Федерации.",
    "FastAPI — современный фреймворк для создания API на Python.",
    "Трансформеры произвели революцию в области NLP.",
    "Оптимизация инференса — важная задача для production систем.",
]
SAMPLE_TEXTS_32 = SAMPLE_TEXTS_8 * 4


def wait_for_server(base_url: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server at {base_url} did not become ready within {timeout}s")


def benchmark_single_latency(base_url: str, n: int = 100) -> dict:
    """отправляет n последовательных single-text запросов, измеряет latency """
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        r = requests.post(f"{base_url}/encode", json={"text": SAMPLE_TEXT})
        elapsed = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        latencies.append(elapsed)

    return {
        "scenario": "single_request_latency",
        "n_requests": n,
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(n * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(n * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def benchmark_batch_latency(base_url: str, batch_sizes: list[int] | None = None, n: int = 100) -> list[dict]:
    """отправляет n последовательных batch запросов для каждого batch size"""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]

    texts_map = {
        1: [SAMPLE_TEXT],
        8: SAMPLE_TEXTS_8,
        32: SAMPLE_TEXTS_32,
    }
    results = []
    for bs in batch_sizes:
        texts = texts_map.get(bs, [SAMPLE_TEXT] * bs)
        latencies = []
        for _ in range(n):
            start = time.perf_counter()
            r = requests.post(f"{base_url}/encode_batch", json={"texts": texts})
            elapsed = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            latencies.append(elapsed)

        results.append({
            "scenario": f"batch_latency_bs{bs}",
            "batch_size": bs,
            "n_requests": n,
            "mean_ms": round(statistics.mean(latencies), 2),
            "median_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(sorted(latencies)[int(n * 0.95)], 2),
            "p99_ms": round(sorted(latencies)[int(n * 0.99)], 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
        })
    return results


def benchmark_throughput(base_url: str, n_workers: int = 10, total_requests: int = 500) -> dict:
    """измеряет пропускную способность """
    latencies = []

    def _send():
        start = time.perf_counter()
        r = requests.post(f"{base_url}/encode", json={"text": SAMPLE_TEXT})
        elapsed = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        return elapsed

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_send) for _ in range(total_requests)]
        for f in as_completed(futures):
            latencies.append(f.result())
    wall_elapsed = time.perf_counter() - wall_start

    return {
        "scenario": "concurrent_throughput",
        "n_workers": n_workers,
        "total_requests": total_requests,
        "wall_time_s": round(wall_elapsed, 2),
        "throughput_rps": round(total_requests / wall_elapsed, 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(total_requests * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(total_requests * 0.99)], 2),
    }


def _find_uvicorn_process() -> psutil.Process | None:
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("uvicorn" in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ResourceMonitor:
    """Samples CPU and memory of a process in a background thread."""

    def __init__(self, process: psutil.Process, interval: float = 0.5):
        self._process = process
        self._interval = interval
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._vms_last: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._process.cpu_percent()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

        if not self._cpu_samples:
            return {"scenario": "resource_usage", "error": "no samples collected"}

        return {
            "scenario": "resource_usage",
            "pid": self._process.pid,
            "cpu_percent_mean": round(statistics.mean(self._cpu_samples), 1),
            "cpu_percent_max": round(max(self._cpu_samples), 1),
            "rss_mb_mean": round(statistics.mean(self._rss_samples), 1),
            "rss_mb_max": round(max(self._rss_samples), 1),
            "vms_mb": round(self._vms_last / (1024 * 1024), 1),
            "n_samples": len(self._cpu_samples),
        }

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                cpu = self._process.cpu_percent()
                mem = self._process.memory_info()
                self._cpu_samples.append(cpu)
                self._rss_samples.append(mem.rss / (1024 * 1024))
                self._vms_last = mem.vms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop_event.wait(self._interval)


def print_results(results: list[dict]):
    print("\n")
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("\n")
    for r in results:
        print(f"\n{r['scenario']}")
        for k, v in r.items():
            if k != "scenario":
                print(f"  {k}: {v}")
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the inference service")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--output", type=str, default="results.json", help="Output filename (default: results.json)")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    output_path = Path(__file__).parent / args.output

    print(f"Целевой сервер: {base_url}")
    print("Ожидание сервера...")
    health = wait_for_server(base_url)
    print(f"Сервер готов: {health}")

    all_results = []

    print("\n[1/4] Латентность одиночного запроса (100 запросов)...")
    single = benchmark_single_latency(base_url, n=100)
    all_results.append(single)

    print("[2/4] Латентность batch запросов (batch sizes: 1, 8, 32)...")
    batch_results = benchmark_batch_latency(base_url, n=100)
    all_results.extend(batch_results)

    print("[3/4] Пропускная способность (10 workers, 500 requests) + замер ресурсов...")
    uvicorn_proc = _find_uvicorn_process()
    monitor = None
    if uvicorn_proc:
        monitor = ResourceMonitor(uvicorn_proc, interval=0.5)
        monitor.start()

    throughput = benchmark_throughput(base_url, n_workers=10, total_requests=500)
    all_results.append(throughput)

    print("[4/4] Потребление ресурсов...")
    if monitor:
        resources = monitor.stop()
    else:
        resources = {"scenario": "resource_usage", "error": "uvicorn process not found"}
    all_results.append(resources)

    health["scenario"] = "server_info"
    all_results.insert(0, health)

    print_results(all_results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
