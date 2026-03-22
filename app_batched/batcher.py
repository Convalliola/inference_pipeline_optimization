import asyncio
import time

from app_onnx.model import _encode


class DynamicBatcher:
    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[tuple[str, asyncio.Future, float]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None):
        self._worker_task = asyncio.ensure_future(self._worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, text: str) -> list[float]:
        future: asyncio.Future[list[float]] = asyncio.get_running_loop().create_future()
        submit_time = time.perf_counter()
        await self._queue.put((text, future, submit_time))
        return await future

    async def _worker(self):
        while True:
            batch_texts: list[str] = []
            batch_futures: list[asyncio.Future] = []
            batch_submit_times: list[float] = []

            try:
                text, future, submit_time = await self._queue.get()
            except asyncio.CancelledError:
                return
            batch_texts.append(text)
            batch_futures.append(future)
            batch_submit_times.append(submit_time)

            deadline = time.perf_counter() + self.max_wait_ms / 1000.0
            while len(batch_texts) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    text, future, submit_time = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch_texts.append(text)
                    batch_futures.append(future)
                    batch_submit_times.append(submit_time)
                except asyncio.TimeoutError:
                    break

            try:
                embeddings = await asyncio.get_running_loop().run_in_executor(
                    None, _encode, batch_texts
                )
                for i, future in enumerate(batch_futures):
                    if not future.cancelled():
                        future.set_result(embeddings[i].tolist())
            except Exception as exc:
                for future in batch_futures:
                    if not future.cancelled() and not future.done():
                        future.set_exception(exc)
