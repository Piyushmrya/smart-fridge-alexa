import logging
import os
import time
from typing import List, Optional, Union
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from image_model.base import BaseDetector, DetectedObject
from image_model.vocabulary import FRIDGE_FOOD_ITEMS, get_food_prompt_chunks

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


def compute_box_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    b2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = b1_area + b2_area - inter
    return inter / union if union > 0 else 0.0


def apply_nms(detections: List[DetectedObject], iou_threshold: float = 0.5) -> List[DetectedObject]:
    """Non-maximum suppression to merge duplicate overlapping bounding boxes."""
    sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
    kept: List[DetectedObject] = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if compute_box_iou(best.bbox, d.bbox) < iou_threshold
        ]
    return kept


def clean_label(raw_label: str) -> str:
    """Clean repeated consecutive tokens (e.g. 'tomato tomato' -> 'tomato')."""
    words = raw_label.strip().split()
    deduped = []
    for word in words:
        if not deduped or deduped[-1] != word:
            deduped.append(word)
    cleaned = " ".join(deduped)
    if cleaned != raw_label:
        logger.debug(f"[GroundingDinoDetector] Normalized label '{raw_label}' -> '{cleaned}'")
    return cleaned


class GroundingDinoDetector(BaseDetector):
    """
    Grounding DINO zero-shot object detector implementing BaseDetector.
    Supports large vocabularies (~200+ food items) via automatic prompt chunking
    to avoid BERT 512-token sequence length limitations.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        prompt: Union[str, List[str]] = FRIDGE_FOOD_ITEMS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        chunk_size: int = 40,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.confidence_threshold = confidence_threshold
        self.chunk_size = chunk_size

        # Prepare prompt chunks safely within 512 tokens
        if isinstance(prompt, list):
            self.prompt_chunks = get_food_prompt_chunks(prompt, chunk_size=self.chunk_size)
            self.total_vocab_size = len(prompt)
        elif isinstance(prompt, str):
            # If string contains multiple dot-separated items
            items = [item.strip() for item in prompt.split(" . ") if item.strip()]
            if len(items) > self.chunk_size:
                self.prompt_chunks = get_food_prompt_chunks(items, chunk_size=self.chunk_size)
                self.total_vocab_size = len(items)
            else:
                self.prompt_chunks = [prompt]
                self.total_vocab_size = len(items) if items else 1
        else:
            raise TypeError("prompt must be a string or list of strings")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = None
        self.model = None
        logger.info(
            f"[GroundingDinoDetector] Initialized detector: vocab_size={self.total_vocab_size} items, "
            f"prompt_chunks={len(self.prompt_chunks)}, target_device='{self.device}', "
            f"confidence_threshold={self.confidence_threshold}"
        )

    def load_model(self) -> None:
        """Loads Grounding DINO processor and model weights into memory/device."""
        if self.processor is not None and self.model is not None:
            logger.debug("[GroundingDinoDetector] Model weights and processor are already in memory. Skipping reload.")
            return

        start_time = time.perf_counter()
        logger.info(
            f"[GroundingDinoDetector] Loading processor & model weights: '{self.model_id}' "
            f"(cache_dir='{self.cache_dir}') to device '{self.device}'..."
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self.model = model.to(self.device)
            self.model.eval()

            elapsed = time.perf_counter() - start_time
            vram_info = ""
            if self.device == "cuda" and torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                vram_info = f", VRAM used: {vram_gb:.2f} GB"

            logger.info(
                f"[GroundingDinoDetector] Model loaded successfully on {self.device}{vram_info} "
                f"(took {elapsed:.2f}s)"
            )
        except Exception as exc:
            logger.exception(
                f"[GroundingDinoDetector] Error occurred while loading model '{self.model_id}': {exc}"
            )
            raise

    def detect(self, image: Image.Image) -> List[DetectedObject]:
        """
        Runs object detection on the PIL image across all prompt vocabulary chunks,
        applies Non-Maximum Suppression (NMS), and returns deduplicated DetectedObject list.
        """
        if self.processor is None or self.model is None:
            logger.info("[GroundingDinoDetector] Model not yet loaded. Performing just-in-time load...")
            self.load_model()

        inference_start = time.perf_counter()
        rgb_image = image.convert("RGB")
        img_w, img_h = rgb_image.size
        logger.info(
            f"[GroundingDinoDetector] Starting detection on frame ({img_w}x{img_h}) across "
            f"{len(self.prompt_chunks)} prompt chunk(s) (total vocab: {self.total_vocab_size} items)..."
        )

        all_detections: List[DetectedObject] = []

        try:
            for chunk_idx, chunk_prompt in enumerate(self.prompt_chunks):
                chunk_start = time.perf_counter()

                inputs = self.processor(
                    images=rgb_image, text=chunk_prompt, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=self.confidence_threshold,
                    text_threshold=self.confidence_threshold,
                    target_sizes=[rgb_image.size[::-1]],  # (height, width)
                )[0]

                boxes = results["boxes"]
                scores = results["scores"]
                if "text_labels" in results:
                    labels = results["text_labels"]
                else:
                    labels = results["labels"]

                chunk_dets: List[DetectedObject] = []
                for box, label, score in zip(boxes, labels, scores):
                    confidence = round(float(score.item()), 4)
                    bbox = [round(float(coord.item()), 2) for coord in box]
                    cleaned_label = clean_label(str(label))

                    chunk_dets.append(
                        DetectedObject(
                            label=cleaned_label,
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )

                chunk_ms = (time.perf_counter() - chunk_start) * 1000
                logger.debug(
                    f"  [Chunk {chunk_idx + 1}/{len(self.prompt_chunks)}] "
                    f"Found {len(chunk_dets)} detection(s) in {chunk_ms:.2f}ms"
                )
                all_detections.extend(chunk_dets)

                # Explicitly clean up CUDA intermediate activations to prevent VRAM fragmentation
                del inputs
                del outputs
                del results
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Apply NMS to remove redundant overlapping bounding boxes
            raw_total = len(all_detections)
            nms_detections = apply_nms(all_detections, iou_threshold=0.5)

            total_elapsed_ms = (time.perf_counter() - inference_start) * 1000
            unique_labels = sorted(list(set(d.label for d in nms_detections)))

            logger.info(
                f"[GroundingDinoDetector] Detection finished: {raw_total} raw detections -> "
                f"{len(nms_detections)} after NMS deduplication across {len(unique_labels)} unique food classes: "
                f"{unique_labels}. Total inference time: {total_elapsed_ms:.2f}ms"
            )

            return nms_detections

        except Exception as exc:
            logger.exception(f"[GroundingDinoDetector] Error during model detection inference: {exc}")
            raise
