"""
dataset.py — COCO 2014 val captions loader for MLPerf Wan T2V benchmark.

Identical structure to the SDXL dataset loader — both benchmarks use
COCO 2014 val captions as the prompt source.
"""

import json
import logging
import os

import mlperf_loadgen as lg

log = logging.getLogger("wan-t2v-dataset")


class COCOCaptionDataset:
    """
    Wraps COCO 2014 val captions for LoadGen.
    Each sample index maps to one caption string.
    """

    def __init__(self, annotations_path: str, total_sample_count: int = 5000):
        self.annotations_path = annotations_path
        self.total_sample_count = total_sample_count
        self.captions: list[str] = []
        self.image_filenames: list[str] = []
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
                "Run: make prepare-mlperf-t2v-data"
            )
        with open(self.annotations_path) as f:
            data = json.load(f)
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"][: self.total_sample_count]:
            self.captions.append(ann["caption"].strip())
            self.image_filenames.append(
                id_to_filename.get(ann["image_id"], f"unknown_{ann['image_id']}.jpg")
            )

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def get_caption(self, index: int) -> str:
        return self.captions[index]
