"""
client.py — LoadGen SUT for MLPerf Wan2.2 T2V benchmark.

Sends text prompts to the Wan T2V FastAPI server over HTTP, saves returned
MP4 videos to disk, and calls lg.QuerySamplesComplete.

Supports Offline (throughput) and SingleStream (latency) scenarios.
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

log = logging.getLogger("wan-t2v-client")


class SUT:
    """System Under Test for Offline scenario."""

    def __init__(
        self,
        dataset: COCOCaptionDataset,
        server_host: str,
        server_port: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_frames: int = 81,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        output_dir: str = "/data/mlperf_t2v/output",
        batch_size: int = 1,
        workers: int = 2,
    ):
        self.dataset = dataset
        self.server_url = f"http://{server_host}:{server_port}/generate"
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_frames = num_frames
        self.fps = fps
        self.height = height
        self.width = width
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = workers

        self.video_dir = self.output_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.query_queue: queue.Queue = queue.Queue()
        self.worker_threads: list[threading.Thread] = []
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

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
        # T2V requests can take 60-300s each — generous timeout
        with httpx.Client(timeout=600.0) as client:
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
                "seed": qs.index,
                "num_frames": self.num_frames,
                "fps": self.fps,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "height": self.height,
                "width": self.width,
            }
            try:
                resp = client.post(self.server_url, json=payload)
                resp.raise_for_status()
                mp4_bytes = resp.content
            except Exception as e:
                log.error(f"Request failed for index {qs.index}: {e}")
                mp4_bytes = b""

            # Save video — filename = dataset index for accuracy alignment
            video_path = self.video_dir / f"{qs.index}.mp4"
            video_path.write_bytes(mp4_bytes)

            resp_data = array.array(
                "B", np.array([len(mp4_bytes)], dtype=np.int32).tobytes()
            )
            bi = resp_data.buffer_info()
            responses.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))

        lg.QuerySamplesComplete(responses)

        elapsed = time.time() - t0
        with self.sample_counter_lock:
            self.sample_counter += len(batch)
            log.info(
                f"Batch done: {len(batch)} video(s) in {elapsed:.1f}s "
                f"| total={self.sample_counter}"
            )


class SUTServer(SUT):
    """System Under Test for SingleStream scenario."""

    def __init__(self, **kwargs):
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("batch_size", 1)
        super().__init__(**kwargs)

    def issue_queries(self, query_samples):
        self.query_queue.put(list(query_samples))
