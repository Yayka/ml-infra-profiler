"""
server.py — FastAPI Wan2.2-T2V inference server for MLPerf T2V benchmark.

Runs on the GPU node. Loads Wan2.2-T2V-A14B at startup and serves
POST /generate requests returning MP4 video bytes.

Usage:
    python server.py --config config/wan_t2v_2node.yaml
    python server.py --model-path /data/mlperf_t2v/models/Wan2.2-T2V-A14B --port 8080
"""

import argparse
import io
import logging
import os
import tempfile
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
pipeline = None
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
    global pipeline, server_config
    model_path = server_config.get("model_path", "Wan-AI/Wan2.2-T2V-A14B")
    log.info(f"Loading Wan2.2-T2V from: {model_path}")
    t0 = time.time()

    try:
        from diffusers import WanPipeline
        pipeline = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
    except ImportError:
        # Fallback: try AutoPipelineForText2Video
        from diffusers import AutoPipelineForText2Video
        pipeline = AutoPipelineForText2Video.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    num_gpus = torch.cuda.device_count()
    log.info(f"Model loaded in {time.time() - t0:.1f}s on {num_gpus} GPU(s)")


@app.get("/healthz")
async def healthz():
    if pipeline is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "gpus": torch.cuda.device_count()}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    generator = torch.Generator(device="cuda").manual_seed(req.seed)
    t0 = time.time()

    output = pipeline(
        prompt=req.prompt,
        num_frames=req.num_frames,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        height=req.height,
        width=req.width,
        generator=generator,
    )

    # output.frames is list of PIL images or a tensor [F, H, W, C]
    frames = output.frames[0] if isinstance(output.frames, list) else output.frames

    latency_ms = (time.time() - t0) * 1000
    log.info(f"Generated {req.num_frames} frames in {latency_ms:.0f}ms | seed={req.seed}")

    mp4_bytes = _frames_to_mp4(frames, fps=req.fps)
    return Response(content=mp4_bytes, media_type="video/mp4")


def _frames_to_mp4(frames, fps: int) -> bytes:
    """Convert list of PIL images or numpy array to MP4 bytes."""
    import imageio
    import numpy as np

    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                                output_params=["-crf", "23", "-pix_fmt", "yuv420p"])

    if hasattr(frames, "__len__") and hasattr(frames[0], "save"):
        # List of PIL images
        for frame in frames:
            writer.append_data(np.array(frame))
    else:
        # Tensor or numpy [F, H, W, C]
        arr = frames.cpu().numpy() if hasattr(frames, "cpu") else np.array(frames)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
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
            "pretrained_path", "Wan-AI/Wan2.2-T2V-A14B"
        )
        server_config["height"] = cfg.get("server", {}).get("height", 480)
        server_config["width"] = cfg.get("server", {}).get("width", 832)
        host = cfg.get("server", {}).get("host", "0.0.0.0")
        port = cfg.get("server", {}).get("port", 8080)
    else:
        server_config["model_path"] = args.model_path or "Wan-AI/Wan2.2-T2V-A14B"
        server_config["height"] = 480
        server_config["width"] = 832
        host = args.host
        port = args.port

    server_config["model_path"] = os.environ.get(
        "WAN_MODEL_PATH", server_config["model_path"]
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
