"""
main.py — LoadGen orchestrator for MLPerf Wan2.2 T2V benchmark.

Mirrors main.py from mlperf_sdxl. Configures LoadGen, constructs SUT,
and runs SingleStream or Offline scenario.

Usage:
    python main.py --scenario SingleStream --config config/wan_t2v_2node.yaml
    python main.py --scenario Offline --max-query-count 5
"""

import argparse
import logging
import os
import sys
import time

import httpx
import mlperf_loadgen as lg
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from client import SUT, SUTServer
from dataset import COCOCaptionDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wan-t2v-main")

scenario_map = {
    "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
}

sut_map = {
    "singlestream": SUTServer,
    "offline": SUT,
}


def wait_for_server(url: str, timeout_s: int = 300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=10.0)
            if resp.status_code == 200:
                log.info(f"Server ready: {resp.json()}")
                return
        except Exception:
            pass
        log.info("Waiting for server (model load takes ~5-10 min for 14B)...")
        time.sleep(15)
    raise RuntimeError(f"Server not ready after {timeout_s}s: {url}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["SingleStream", "Offline"],
        default="SingleStream",
    )
    parser.add_argument("--config", type=str, default="config/wan_t2v_2node.yaml")
    parser.add_argument("--server-host", type=str, default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--annotations-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-sample-count", type=int, default=None)
    parser.add_argument("--max-query-count", type=int, default=0)
    parser.add_argument("--target-qps", type=float, default=None)
    parser.add_argument("--min-duration-ms", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--output-log-dir", type=str, default="output-logs")
    parser.add_argument("--enable-log-trace", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    args = get_args()
    cfg = load_config(args.config)

    server_host = (
        args.server_host
        or os.environ.get("SERVER_IP")
        or cfg.get("server", {}).get("host", "localhost")
    )
    server_port = args.server_port or cfg.get("server", {}).get("port", 8080)
    annotations_path = args.annotations_path or cfg.get("data", {}).get(
        "annotations_path", "/data/mlperf_t2v/annotations/captions_val2014.json"
    )
    output_dir = args.output_dir or cfg.get("data", {}).get(
        "output_dir", "/data/mlperf_t2v/output"
    )
    total_sample_count = args.total_sample_count or cfg.get("data", {}).get(
        "total_sample_count", 5000
    )
    target_qps = args.target_qps or cfg.get("loadgen", {}).get("target_qps", 0.1)
    min_duration_ms = args.min_duration_ms or cfg.get("loadgen", {}).get(
        "min_duration_ms", 60000
    )
    srv = cfg.get("server", {})

    scenario_key = args.scenario.lower()

    if not args.skip_server_check:
        wait_for_server(f"http://{server_host}:{server_port}/healthz")

    dataset = COCOCaptionDataset(
        annotations_path=annotations_path,
        total_sample_count=total_sample_count,
    )

    settings = lg.TestSettings()
    settings.scenario = scenario_map[scenario_key]
    settings.mode = (
        lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
    )

    if scenario_key == "offline":
        settings.offline_expected_qps = target_qps
    elif scenario_key == "singlestream":
        settings.single_stream_expected_latency_ns = int(1e9 / target_qps)

    settings.min_duration_ms = min_duration_ms

    if args.max_query_count > 0:
        settings.max_query_count = args.max_query_count
        settings.min_query_count = args.max_query_count

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    sut_cls = sut_map[scenario_key]
    sut = sut_cls(
        dataset=dataset,
        server_host=server_host,
        server_port=server_port,
        num_inference_steps=srv.get("num_inference_steps", 50),
        guidance_scale=srv.get("guidance_scale", 5.0),
        num_frames=srv.get("num_frames", 81),
        fps=srv.get("fps", 16),
        height=srv.get("height", 480),
        width=srv.get("width", 832),
        output_dir=output_dir,
        workers=args.workers,
    )

    sut.start()
    lg_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)

    log.info(
        f"Starting {args.scenario} | server={server_host}:{server_port} "
        f"| samples={total_sample_count}"
    )
    lg.StartTestWithLogSettings(lg_sut, sut.qsl, settings, log_settings, "")

    sut.stop()
    log.info("Benchmark complete")

    lg.DestroySUT(lg_sut)
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
