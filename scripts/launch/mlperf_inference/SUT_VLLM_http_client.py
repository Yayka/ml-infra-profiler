# Client-only SUT: SUTHTTPClient and SUTHTTPClientServer.
# Stripped of all torch/vLLM imports so the client container stays lightweight.
# Mounted as SUT_VLLM.py in the inference container when running in HTTP client mode.

import array
import json
import logging
import queue
import threading
import time

import numpy as np
import requests

import mlperf_loadgen as lg
from dataset import Dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-SUT-HTTPClient")


class SUTHTTPClient:
    """Offline-scenario SUT that forwards queries to a remote vLLM OpenAI API server."""

    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=24576,
        dataset_path=None,
        workers=1,
        max_output_tokens=1024,
        server_url=None,
        api_model_name=None,
    ):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size or 1
        self.dtype = dtype
        self.max_output_tokens = max_output_tokens
        self.server_url = server_url.rstrip("/")
        self.api_model_name = api_model_name or self.model_path

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            model_name=self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
        )
        # Load tokenizer directly — Dataset may not expose it as a public attribute
        self.tokenizer = (
            self.data_object.tokenizer
            if hasattr(self.data_object, "tokenizer")
            else AutoTokenizer.from_pretrained(self.model_path)
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

        self._wait_for_server()

    def _wait_for_server(self, retries=20, delay=5):
        url = f"{self.server_url}/v1/models"
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    log.info("Server ready")
                    return
            except requests.RequestException:
                pass
            log.info(f"Waiting for server ({attempt}/{retries})...")
            time.sleep(delay)
        raise RuntimeError(f"Server at {self.server_url} did not become ready after {retries} attempts")

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.join()

    def flush_queries(self):
        pass

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]
            pred_output_tokens = []

            for q in qitem:
                raw = self.data_object.input_ids[q.index]
                token_ids = np.squeeze(raw, axis=0).tolist() if hasattr(raw, 'shape') else list(raw)
                payload = {
                    "model": self.api_model_name,
                    "prompt": token_ids,
                    "max_tokens": self.max_output_tokens,
                    "min_tokens": 2,
                    "temperature": 1,
                    "top_p": 1,
                    "seed": 42,
                    "stream": False,
                }
                resp = requests.post(
                    f"{self.server_url}/v1/completions", json=payload, timeout=300
                )
                if not resp.ok:
                    log.error(f"Server returned {resp.status_code}: {resp.text}")
                resp.raise_for_status()
                text = resp.json()["choices"][0]["text"]
                output_token_ids = self.tokenizer.encode(text)
                pred_output_tokens.append(output_token_ids)

            for i in range(len(qitem)):
                token_ids = pred_output_tokens[i]
                output_array = np.array(token_ids, dtype=np.int32)
                n_tokens = output_array.shape[0]
                response_array = array.array("B", output_array.tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl


class SUTHTTPClientServer(SUTHTTPClient):
    """Server-scenario SUT that streams requests to a remote vLLM OpenAI API server."""

    def issue_queries(self, query_samples):
        self.query_queue.put(query_samples[0])

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            self._run_query_streaming(qitem)

    def _run_query_streaming(self, qitem):
        raw = self.data_object.input_ids[qitem.index]
        token_ids = np.squeeze(raw, axis=0).tolist() if hasattr(raw, 'shape') else list(raw)
        payload = {
            "model": self.api_model_name,
            "prompt": token_ids,
            "max_tokens": self.max_output_tokens,
            "min_tokens": 2,
            "temperature": 1,
            "top_p": 1,
            "seed": 42,
            "stream": True,
        }

        full_text = ""
        first = True
        with requests.post(
            f"{self.server_url}/v1/completions", json=payload, stream=True, timeout=300
        ) as resp:
            if not resp.ok:
                log.error(f"Server returned {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta_text = chunk["choices"][0].get("text", "")
                if delta_text and first:
                    first_token_ids = self.tokenizer.encode(delta_text)
                    response_data = array.array(
                        "B", np.array(first_token_ids, np.int32).tobytes()
                    )
                    bi = response_data.buffer_info()
                    lg.FirstTokenComplete([lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])
                    first = False
                full_text += delta_text

        output_token_ids = self.tokenizer.encode(full_text)
        output_array = np.array(output_token_ids, dtype=np.int32)
        n_tokens = output_array.shape[0]
        response_array = array.array("B", output_array.tobytes())
        bi = response_array.buffer_info()
        lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)])

        with self.sample_counter_lock:
            self.sample_counter += 1
            log.info(f"Samples run: {self.sample_counter}")
