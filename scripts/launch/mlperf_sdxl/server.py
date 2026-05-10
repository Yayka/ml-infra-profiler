"""
server.py — FastAPI SDXL inference server for MLPerf T2I benchmark.

Runs on the GPU node. Loads SDXL-base-1.0 at startup and serves
POST /generate requests with PNG image bytes.

Usage:
    python server.py --config config/sdxl_2node.yaml
    python server.py --model-path /data/mlperf_sdxl/models/sdxl-base-1.0 --port 8080
"""

import argparse
import io
import logging
import os
import time

import torch
import uvicorn
import yaml
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sdxl-server")

app = FastAPI()
pipeline: StableDiffusionXLPipeline | None = None
server_config: dict = {}


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = 0
    steps: int = 20
    guidance_scale: float = 7.5


@app.on_event("startup")
async def load_model():
    global pipeline, server_config
    model_path = server_config.get("model_path", "stabilityai/stable-diffusion-xl-base-1.0")
    log.info(f"Loading SDXL from: {model_path}")
    t0 = time.time()
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipeline = pipeline.to("cuda")
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
    image = pipeline(
        prompt=req.prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
        height=server_config.get("image_size", 1024),
        width=server_config.get("image_size", 1024),
    ).images[0]
    latency_ms = (time.time() - t0) * 1000
    log.info(f"Generated image in {latency_ms:.0f}ms | seed={req.seed}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


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
            "pretrained_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        server_config["image_size"] = cfg.get("server", {}).get("image_size", 1024)
        host = cfg.get("server", {}).get("host", "0.0.0.0")
        port = cfg.get("server", {}).get("port", 8080)
    else:
        server_config["model_path"] = (
            args.model_path or "stabilityai/stable-diffusion-xl-base-1.0"
        )
        server_config["image_size"] = 1024
        host = args.host
        port = args.port

    # Allow env-var override for model path
    server_config["model_path"] = os.environ.get(
        "SDXL_MODEL_PATH", server_config["model_path"]
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
