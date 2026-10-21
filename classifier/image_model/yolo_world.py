import logging
import os
import time
from typing import List, Optional, Union
from PIL import Image

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.grounding_dino import clean_label, apply_nms
from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_ID = "yolov8s-worldv2.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_IOU_THRESHOLD = 0.5


class YoloWorldDetector(BaseDetector):
    """
    YOLO-World real-time open-vocabulary object detector.
    Precomputes text embeddings offline for large vocabularies (200+ items),
    completely bypassing sequence token limits and eliminating prompt chunking.
    """

    @Profiler
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        vocabulary: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: Optional[str] = None,
    ):
        # If model_id is a filename, check if it exists in local models cache directory
        if not os.path.isabs(model_id):
            cached_candidate = os.path.join(DEFAULT_MODELS_DIR, model_id)
            if os.path.exists(cached_candidate):
                model_id = cached_candidate

        self.model_id = model_id
        self.vocabulary = vocabulary if vocabulary is not None else list(FRIDGE_FOOD_ITEMS)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        if device is None or device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        logger.info(
            f"[YoloWorldDetector] Initialized detector: model_id='{self.model_id}', "
            f"vocab_size={len(self.vocabulary)} items, device='{self.device}', "
            f"confidence_threshold={self.confidence_threshold}, iou_threshold={self.iou_threshold}"
        )

    @Profiler
    def load_model(self) -> None:
        """Loads YOLO-World model weights and sets the custom offline vocabulary."""
        if self.model is not None:
            logger.debug("[YoloWorldDetector] Model already loaded. Skipping reload.")
            return

        from ultralytics import YOLO

        start_time = time.perf_counter()
        logger.info(
            f"[YoloWorldDetector] Loading YOLO-World model: '{self.model_id}' on device '{self.device}'..."
        )

        try:
            self.model = YOLO(self.model_id)
            # Embed the vocabulary into offline text embeddings
            logger.info(f"[YoloWorldDetector] Precomputing embeddings for {len(self.vocabulary)} categories...")
            self.model.set_classes(self.vocabulary)

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[YoloWorldDetector] Model loaded and vocabulary embedded successfully (took {elapsed:.2f}s)"
            )
        except Exception as exc:
            logger.exception(f"[YoloWorldDetector] Failed to load model '{self.model_id}': {exc}")
            raise

    @Profiler
    def detect(self, image: Image.Image) -> List[DetectedObject]:
        """
        Runs single-pass open-vocabulary detection on the PIL image across the entire vocabulary.
        """
        if self.model is None:
            logger.info("[YoloWorldDetector] Model not loaded yet. Performing just-in-time load...")
            self.load_model()

        inference_start = time.perf_counter()
        rgb_image = image.convert("RGB")

        try:
            results = self.model.predict(
                rgb_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )[0]

            detections: List[DetectedObject] = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_idx = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                    if 0 <= cls_idx < len(self.vocabulary):
                        raw_label = self.vocabulary[cls_idx]
                    else:
                        raw_label = str(cls_idx)

                    label = clean_label(raw_label)
                    detections.append(
                        DetectedObject(
                            label=label,
                            confidence=round(score, 4),
                            bbox=xyxy,
                        )
                    )

            # Apply NMS for additional cross-class box deduplication
            deduped = apply_nms(detections, iou_threshold=self.iou_threshold)

            elapsed_ms = (time.perf_counter() - inference_start) * 1000
            unique_labels = sorted(list(set(d.label for d in deduped)))
            logger.info(
                f"[YoloWorldDetector] Detection finished: {len(detections)} raw -> {len(deduped)} after NMS "
                f"across {len(unique_labels)} classes: {unique_labels}. Total inference time: {elapsed_ms:.2f}ms"
            )
            return deduped

        except Exception as exc:
            logger.exception(f"[YoloWorldDetector] Error during YOLO-World inference: {exc}")
            raise
