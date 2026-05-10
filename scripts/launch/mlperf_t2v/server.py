"""
server.py — FastAPI Wan2.2-T2V inference server for MLPerf T2V benchmark.

Uses the native Wan inference library (not diffusers) since Wan2.2-T2V-A14B
ships in Wan's own checkpoint format with separate high/low noise models.

Install the Wan library before running:
    pip install git+https://github.com/Wan-AI/Wan2.1.git

Usage:
    python server.py --config config/wan_t2v_2node.yaml
"""

import argparse
import io
import logging
import os
import time

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wan-t2v-server")

app = FastAPI()
wan_model = None
server_config: dict = {}


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 0
    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    height: int = 480
    width: int = 832


@app.on_event("startup")
async def load_model():
    global wan_model, server_config
    model_path = server_config.get("model_path", "/data/mlperf_t2v/models/Wan2.2-T2V-A14B")
    task = server_config.get("task", "t2v-14B")

    log.info(f"Loading Wan2.2-T2V from: {model_path} (task={task})")
    t0 = time.time()

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        if task not in WAN_CONFIGS:
            available = list(WAN_CONFIGS.keys())
            # pick the closest t2v key
            t2v_keys = [k for k in available if "t2v" in k.lower()]
            task = t2v_keys[0] if t2v_keys else available[0]
            log.warning(f"Task not found — using: {task}")

        cfg = WAN_CONFIGS[task]
        num_gpus = torch.cuda.device_count()
        log.info(f"Initialising Wan on {num_gpus} GPU(s)...")

        wan_model = wan.WANTxt2Vid(
            cfg,
            checkpoint_dir=model_path,
            device_id=0,
            rank=0,
        )
        log.info(f"Model loaded in {time.time() - t0:.1f}s")

    except ImportError:
        log.error(
            "Wan library not installed. Run:\n"
            "  pip install git+https://github.com/Wan-AI/Wan2.1.git"
        )
        raise


@app.get("/healthz")
async def healthz():
    if wan_model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "gpus": torch.cuda.device_count()}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if wan_model is None:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    t0 = time.time()
    try:
        video = wan_model.generate(
            req.prompt,
            size=(req.width, req.height),
            frame_num=req.num_frames,
            shift=server_config.get("shift", 5.0),
            sample_steps=req.num_inference_steps,
            guide_scale=req.guidance_scale,
            seed=req.seed,
            offload_model=False,
        )
    except Exception as e:
        log.error(f"Generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    latency_ms = (time.time() - t0) * 1000
    log.info(f"Generated {req.num_frames} frames in {latency_ms:.0f}ms | seed={req.seed}")

    mp4_bytes = _tensor_to_mp4(video, fps=req.fps)
    return Response(content=mp4_bytes, media_type="video/mp4")


def _tensor_to_mp4(video, fps: int) -> bytes:
    """Convert Wan output video tensor [C, F, H, W] or [F, H, W, C] to MP4 bytes."""
    import imageio
    import numpy as np

    # Wan returns tensor in [C, F, H, W] format, float in [-1, 1]
    if hasattr(video, "cpu"):
        arr = video.cpu().float().numpy()
    else:
        arr = np.array(video)

    # Normalise to [0, 255] uint8
    if arr.ndim == 4 and arr.shape[0] in (1, 3):
        # [C, F, H, W] → [F, H, W, C]
        arr = arr.transpose(1, 2, 3, 0)
    if arr.dtype != np.uint8:
        arr = ((arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

    buf = io.BytesIO()
    writer = imageio.get_writer(
        buf, format="mp4", fps=fps, codec="libx264",
        output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
    )
    for i in range(arr.shape[0]):
        writer.append_data(arr[i])
    writer.close()
    return buf.getvalue()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser.parse_args()


def main():
    global server_config
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        server_config["model_path"] = cfg.get("model", {}).get(
            "pretrained_path", "/data/mlperf_t2v/models/Wan2.2-T2V-A14B"
        )
        server_config["task"] = cfg.get("model", {}).get("task", "t2v-14B")
        server_config["shift"] = cfg.get("server", {}).get("shift", 5.0)
        host = cfg.get("server", {}).get("host", "0.0.0.0")
        port = cfg.get("server", {}).get("port", 8080)
    else:
        server_config["model_path"] = (
            args.model_path or "/data/mlperf_t2v/models/Wan2.2-T2V-A14B"
        )
        server_config["task"] = "t2v-14B"
        server_config["shift"] = 5.0
        host = args.host
        port = args.port

    server_config["model_path"] = os.environ.get(
        "WAN_MODEL_PATH", server_config["model_path"]
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
