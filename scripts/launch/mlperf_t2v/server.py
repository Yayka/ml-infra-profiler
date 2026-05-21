"""
server.py — FastAPI Wan2.2-T2V-A14B-Diffusers inference server for MLPerf T2V benchmark.

Uses the diffusers WanPipeline with AutoencoderKLWan, matching the MLCommons
reference implementation at:
  github.com/mlcommons/inference/tree/master/text_to_video/wan-2.2-t2v-a14b

Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers (NOT the native Wan format)

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
pipeline = None
server_config: dict = {}

NEGATIVE_PROMPT = (
    "vivid colors, overexposed, static, blurry details, subtitles, style, "
    "work of art, painting, picture, still, overall grayish, worst quality, "
    "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
    "static image, cluttered background, three legs, many people in the background, "
    "walking backwards"
)


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 0
    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 20
    guidance_scale: float = 4.0
    guidance_scale_2: float = 3.0
    boundary_ratio: float = 0.875
    height: int = 720
    width: int = 1280
    negative_prompt: str = NEGATIVE_PROMPT


@app.on_event("startup")
async def load_model():
    global pipeline, server_config
    model_path = server_config.get("model_path", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    boundary_ratio = server_config.get("boundary_ratio", 0.875)

    log.info(f"Loading Wan2.2-T2V-Diffusers from: {model_path}")
    t0 = time.time()

    from diffusers import AutoencoderKLWan, WanPipeline

    # Try loading VAE from subfolder (some model variants), fall back to pipeline-only load
    vae_path = os.path.join(model_path, "vae")
    if os.path.isdir(vae_path):
        vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch.float32)
        pipeline = WanPipeline.from_pretrained(
            model_path,
            vae=vae,
            boundary_ratio=boundary_ratio,
            torch_dtype=torch.bfloat16,
        )
    else:
        # No separate vae/ dir — let WanPipeline load everything from model_index.json.
        pipeline = WanPipeline.from_pretrained(
            model_path,
            boundary_ratio=boundary_ratio,
            torch_dtype=torch.bfloat16,
        )
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    log.info(f"Model loaded in {time.time() - t0:.1f}s on {torch.cuda.device_count()} GPU(s)")


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
        negative_prompt=req.negative_prompt,
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        guidance_scale_2=req.guidance_scale_2,
        generator=generator,
    )
    frames = output.frames[0]  # list of PIL images

    latency_ms = (time.time() - t0) * 1000
    log.info(f"Generated {req.num_frames} frames in {latency_ms:.0f}ms | seed={req.seed}")

    mp4_bytes = _frames_to_mp4(frames, fps=req.fps)
    return Response(content=mp4_bytes, media_type="video/mp4")


def _frames_to_mp4(frames, fps: int) -> bytes:
    import imageio
    import numpy as np

    buf = io.BytesIO()
    writer = imageio.get_writer(
        buf, format="mp4", fps=fps, codec="libx264",
        output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        writer.append_data(np.array(frame))
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
        srv = cfg.get("server", {})
        server_config["model_path"] = cfg.get("model", {}).get(
            "pretrained_path", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        )
        server_config["boundary_ratio"] = srv.get("boundary_ratio", 0.875)
        host = srv.get("host", "0.0.0.0")
        port = srv.get("port", 8080)
        # CLI flags always override config
        if args.port != 8080:
            port = args.port
        if args.host != "0.0.0.0":
            host = args.host
    else:
        server_config["model_path"] = (
            args.model_path or "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        )
        server_config["boundary_ratio"] = 0.875
        host = args.host
        port = args.port

    server_config["model_path"] = os.environ.get(
        "WAN_MODEL_PATH", server_config["model_path"]
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
