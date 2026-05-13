import asyncio
import json
import os
import time
import numpy as np
import array
import requests
import torch
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt

import pickle
import threading
import queue

import logging
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama2-70B-SUT")


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=24576,
        dataset_path=None,
        workers=1,
        tensor_parallel_size=2,
        max_output_tokens=1024,
    ):
        self.model_path = model_path or "meta-llama/Llama-2-70b-chat-hf"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            model_name=self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.max_output_tokens = max_output_tokens
        self.load_model()
        gen_kwargs = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": self.max_output_tokens,
            "min_tokens": 2,
        }
        self.sampling_params = SamplingParams(**gen_kwargs)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

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

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()

            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]

            tik2 = time.time()
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor, sampling_params=self.sampling_params
            )
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=24576,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=2,
        max_output_tokens=1024,
    ):
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
            max_output_tokens=max_output_tokens,
        )
        self.request_id = 0
        self.first_token_queue = queue.Queue()

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    async def stream_output(self, qitem, results_generator):
        first = True
        async for request_output in results_generator:
            output_response = request_output
            if first:
                first_tokens = list(output_response.outputs[0].token_ids)
                response_data = array.array(
                    "B", np.array(first_tokens, np.int32).tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
                lg.FirstTokenComplete(response)
                first = False

        pred_output_tokens = list(output_response.outputs[0].token_ids)
        n_tokens = len(pred_output_tokens)
        response_array = array.array(
            "B", np.array(pred_output_tokens, np.int32).tobytes()
        )
        bi = response_array.buffer_info()
        response = [
            lg.QuerySampleResponse(
                qitem.id,
                bi[0],
                bi[1],
                n_tokens)]
        lg.QuerySamplesComplete(response)

    async def _run_query(self, qitem):
        """Run a single query entirely within the persistent engine event loop."""
        raw = self.data_object.input_ids[qitem.index]
        # dataset.py stores tensors of shape (1, seq_len); squeeze batch dim and convert to list
        token_ids = raw.squeeze(0).tolist()
        input_ids_tensor = TokensPrompt(prompt_token_ids=token_ids)
        results_generator = self.model.generate(
            prompt=input_ids_tensor,
            sampling_params=self.sampling_params,
            request_id=str(self.request_id),
        )
        self.request_id += 1
        await self.stream_output(qitem, results_generator)

    def process_queries(self):
        """Processor of the queued queries."""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            # Submit to the persistent engine loop and block until done.
            # Using run_coroutine_threadsafe avoids creating a new event loop
            # per request, which would destroy the AsyncLLMEngine background
            # task after the first query and hang on all subsequent ones.
            future = asyncio.run_coroutine_threadsafe(
                self._run_query(qitem), self._engine_loop)
            future.result()

    def issue_queries(self, query_samples):
        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)

        # Stop the persistent engine event loop
        self._engine_loop.call_soon_threadsafe(self._engine_loop.stop)
        self._engine_loop_thread.join()

    def load_model(self):
        log.info("Loading model")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size)
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)

        # Start a persistent event loop in a background thread so the
        # AsyncLLMEngine's run_engine_loop task survives across requests.
        self._engine_loop = asyncio.new_event_loop()
        self._engine_loop_thread = threading.Thread(
            target=self._engine_loop.run_forever,
            daemon=True,
            name="vllm-engine-loop",
        )
        self._engine_loop_thread.start()

        log.info("Loaded model")


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
        self.model_path = model_path or "meta-llama/Llama-2-70b-chat-hf"
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
                token_ids = raw.squeeze(0).tolist()
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
                output_token_ids = self.data_object.tokenizer.encode(text)
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
        token_ids = raw.squeeze(0).tolist()
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
                    first_token_ids = self.data_object.tokenizer.encode(delta_text)
                    response_data = array.array(
                        "B", np.array(first_token_ids, np.int32).tobytes()
                    )
                    bi = response_data.buffer_info()
                    lg.FirstTokenComplete([lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])
                    first = False
                full_text += delta_text

        output_token_ids = self.data_object.tokenizer.encode(full_text)
        output_array = np.array(output_token_ids, dtype=np.int32)
        n_tokens = output_array.shape[0]
        response_array = array.array("B", output_array.tobytes())
        bi = response_array.buffer_info()
        lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)])

        with self.sample_counter_lock:
            self.sample_counter += 1
            log.info(f"Samples run: {self.sample_counter}")
