"""
client.py — LoadGen SUT for MLPerf SDXL T2I benchmark.

Sends text prompts to the SDXL FastAPI server over HTTP, saves returned
PNG images to disk, and calls lg.QuerySamplesComplete.

Supports Offline (batched throughput) and SingleStream (one-at-a-time latency)
scenarios, matching the SUT_VLLM_patched.py pattern used for Llama benchmarks.
"""

import array
import logging
import os
import queue
import threading
import time
from pathlib import Path

import httpx
import mlperf_loadgen as lg
import numpy as np

from dataset import COCOCaptionDataset

log = logging.getLogger("sdxl-client")


class SUT:
    """
    System Under Test for Offline scenario (batched, throughput-optimised).

    issue_queries() enqueues batches; worker threads POST to /generate
    and save images to output_dir/images/<query_id>.png.
    """

    def __init__(
        self,
        dataset: COCOCaptionDataset,
        server_host: str,
        server_port: int,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        output_dir: str = "/data/mlperf_sdxl/output",
        batch_size: int = 1,
        workers: int = 4,
    ):
        self.dataset = dataset
        self.server_url = f"http://{server_host}:{server_port}/generate"
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = workers

        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.query_queue: queue.Queue = queue.Queue()
        self.worker_threads: list[threading.Thread] = []

        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

        # LoadGen QSL
        self.qsl = lg.ConstructQSL(
            self.dataset.total_sample_count,
            self.dataset.perf_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam,
        )

    def start(self):
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._process_queries, daemon=True)
            t.start()
            self.worker_threads.append(t)

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for t in self.worker_threads:
            t.join()

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery: {len(query_samples)} sample(s)")
        batch = list(query_samples)
        while batch:
            self.query_queue.put(batch[: self.batch_size])
            batch = batch[self.batch_size:]

    def flush_queries(self):
        pass

    def _process_queries(self):
        with httpx.Client(timeout=300.0) as client:
            while True:
                batch = self.query_queue.get()
                if batch is None:
                    break
                self._run_batch(client, batch)

    def _run_batch(self, client: httpx.Client, batch):
        t0 = time.time()
        responses = []
        for qs in batch:
            caption = self.dataset.get_caption(qs.index)
            payload = {
                "prompt": caption,
                "seed": qs.index,  # deterministic per sample
                "steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
            }
            try:
                resp = client.post(self.server_url, json=payload)
                resp.raise_for_status()
                png_bytes = resp.content
            except Exception as e:
                log.error(f"Request failed for index {qs.index}: {e}")
                png_bytes = b""

            # Save image to disk
            img_path = self.image_dir / f"{qs.id}.png"
            img_path.write_bytes(png_bytes)

            # Return minimal response token (image path length as proxy)
            resp_data = array.array("B", np.array([len(png_bytes)], dtype=np.int32).tobytes())
            bi = resp_data.buffer_info()
            responses.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)

        elapsed = time.time() - t0
        with self.sample_counter_lock:
            self.sample_counter += len(batch)
            log.info(
                f"Batch done: {len(batch)} sample(s) in {elapsed:.2f}s "
                f"| total={self.sample_counter}"
            )


class SUTServer(SUT):
    """
    System Under Test for SingleStream scenario (one query at a time).

    issue_queries() is called by LoadGen with exactly 1 sample;
    we process it synchronously before returning.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("batch_size", 1)
        super().__init__(**kwargs)

    def issue_queries(self, query_samples):
        # SingleStream: LoadGen sends one sample at a time
        self.query_queue.put(list(query_samples))
