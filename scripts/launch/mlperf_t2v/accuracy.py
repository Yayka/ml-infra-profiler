"""
accuracy.py — CLIP score evaluation for MLPerf Wan2.2 T2V benchmark.

Reads generated MP4 videos from output_dir/videos/, samples frames,
and computes CLIP score between frames and their source captions.

FVD (Fréchet Video Distance) requires a reference video set — noted
but not computed here without ground-truth videos.

MLPerf v6.0 T2V targets (reference values, subject to official spec):
  CLIP score >= 18.0  (frame-level, ViT-H-14)

Usage:
    python accuracy.py \
        --output-dir /data/mlperf_t2v/output \
        --annotations /data/mlperf_t2v/annotations/captions_val2014.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

log = logging.getLogger("wan-t2v-accuracy")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--annotations",
        type=str,
        default="/data/mlperf_t2v/annotations/captions_val2014.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
        help="Frames to sample per video for CLIP evaluation",
    )
    parser.add_argument(
        "--clip-target",
        type=float,
        default=18.0,
        help="MLPerf CLIP score pass threshold",
    )
    return parser.parse_args()


def load_captions(annotations_path: str, count: int) -> list[str]:
    with open(annotations_path) as f:
        data = json.load(f)
    return [ann["caption"].strip() for ann in data["annotations"][:count]]


def sample_frames_from_video(video_path: Path, n_frames: int) -> list[Image.Image]:
    """Sample n_frames evenly from an MP4 file."""
    try:
        import imageio
        reader = imageio.get_reader(str(video_path), format="mp4")
        meta = reader.get_meta_data()
        total = meta.get("nframes", None)

        all_frames = list(reader)
        reader.close()

        if not all_frames:
            return []

        total = len(all_frames)
        indices = [int(i * total / n_frames) for i in range(n_frames)]
        indices = [min(i, total - 1) for i in indices]
        return [Image.fromarray(all_frames[i]) for i in indices]
    except Exception as e:
        log.warning(f"Failed to read {video_path.name}: {e}")
        return []


def compute_clip_score(
    video_paths: list[Path],
    captions: list[str],
    device: str,
    batch_size: int,
    frames_per_video: int,
) -> float:
    try:
        import open_clip
    except ImportError:
        log.error("open-clip-torch not installed. Run: pip install open-clip-torch")
        sys.exit(1)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model = model.float()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model.eval()

    all_scores = []
    n = min(len(video_paths), len(captions))
    log.info(f"Computing CLIP score for {n} video-caption pairs ({frames_per_video} frames each)...")

    img_batch = []
    cap_batch = []

    def flush_batch():
        if not img_batch:
            return
        tensors = torch.stack([preprocess(img) for img in img_batch]).to(device).float()
        tokens = tokenizer(cap_batch).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(tensors)
            txt_feat = model.encode_text(tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores = (img_feat * txt_feat).sum(dim=-1).clamp(min=0) * 100.0
            all_scores.extend(scores.cpu().float().tolist())
        img_batch.clear()
        cap_batch.clear()

    for i, (vpath, caption) in enumerate(zip(video_paths, captions)):
        frames = sample_frames_from_video(vpath, frames_per_video)
        if not frames:
            log.warning(f"Skipping {vpath.name} — no frames decoded")
            continue
        for frame in frames:
            img_batch.append(frame)
            cap_batch.append(caption)
            if len(img_batch) >= batch_size:
                flush_batch()

        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i + 1}/{n} videos...")

    flush_batch()

    return sum(all_scores) / len(all_scores) if all_scores else 0.0


def main():
    args = get_args()
    video_dir = Path(args.output_dir) / "videos"

    video_paths = sorted(video_dir.glob("*.mp4"), key=lambda p: int(p.stem))
    if not video_paths:
        log.error(f"No MP4 files found in {video_dir}")
        sys.exit(1)

    n = len(video_paths)
    log.info(f"Found {n} generated videos")

    captions = load_captions(args.annotations, max(int(p.stem) for p in video_paths) + 1)
    # Align captions to video indices
    aligned_captions = []
    valid_paths = []
    for vpath in video_paths:
        idx = int(vpath.stem)
        if idx < len(captions):
            aligned_captions.append(captions[idx])
            valid_paths.append(vpath)

    clip_score = compute_clip_score(
        valid_paths, aligned_captions, args.device, args.batch_size, args.frames_per_video
    )

    clip_pass = clip_score >= args.clip_target

    print("\n" + "=" * 50)
    print("MLPerf Wan2.2 T2V Accuracy Results")
    print("=" * 50)
    print(f"  Videos evaluated : {len(valid_paths)}")
    print(f"  Frames per video : {args.frames_per_video}")
    print(f"  CLIP score       : {clip_score:.4f}  (target >= {args.clip_target})")
    print(f"  FID/FVD          : N/A (requires reference video set)")
    print()
    print(f"  CLIP: {'PASS' if clip_pass else 'FAIL'}")
    print("=" * 50)

    results_path = Path(args.output_dir) / "accuracy_result.txt"
    with open(results_path, "w") as f:
        f.write(f"videos_evaluated={len(valid_paths)}\n")
        f.write(f"clip_score={clip_score:.6f}\n")
        f.write(f"clip_pass={clip_pass}\n")
    log.info(f"Results written to {results_path}")

    if not clip_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
