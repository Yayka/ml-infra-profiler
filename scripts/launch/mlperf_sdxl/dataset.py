"""
dataset.py — COCO 2014 val captions loader for MLPerf SDXL T2I benchmark.

Loads captions_val2014.json and exposes them as a flat list of prompts
along with the matching COCO image filenames (used for FID ground-truth).
"""

import json
import logging
import os

import mlperf_loadgen as lg

log = logging.getLogger("sdxl-dataset")


class COCOCaptionDataset:
    """
    Wraps COCO 2014 val captions for LoadGen.

    Each sample index maps to one caption string.
    Multiple captions per image are flattened — index → caption (not image).
    Only the first `total_sample_count` captions are used.
    """

    def __init__(self, annotations_path: str, total_sample_count: int = 5000):
        self.annotations_path = annotations_path
        self.total_sample_count = total_sample_count

        self.captions: list[str] = []
        self.image_filenames: list[str] = []  # for FID ground-truth matching

        self._load()

        self.perf_count = min(total_sample_count, len(self.captions))
        log.info(
            f"Dataset loaded: {len(self.captions)} captions "
            f"(using {self.perf_count} for benchmark)"
        )

    def _load(self):
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(
                f"COCO annotations not found: {self.annotations_path}\n"
                "Run: make prepare-mlperf-sdxl-data"
            )

        with open(self.annotations_path) as f:
            data = json.load(f)

        # Build image_id → filename lookup
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        # Flatten annotations, cap at total_sample_count
        for ann in data["annotations"][: self.total_sample_count]:
            self.captions.append(ann["caption"].strip())
            self.image_filenames.append(
                id_to_filename.get(ann["image_id"], f"unknown_{ann['image_id']}.jpg")
            )

    # ---------- LoadGen QSL callbacks ----------

    def LoadSamplesToRam(self, sample_list):
        pass  # captions are tiny — always in RAM

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def get_caption(self, index: int) -> str:
        return self.captions[index]

    def get_image_filename(self, index: int) -> str:
        return self.image_filenames[index]
