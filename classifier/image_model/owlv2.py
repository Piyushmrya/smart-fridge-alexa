import logging
import os
import time
from typing import List, Optional
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.grounding_dino import clean_label, apply_nms
from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_CONFIDENCE_THRESHOLD = 0.15


class OwlV2Detector(BaseDetector):
    """
    Google OWLv2 (Open-Vocabulary Object Localization v2) detector.
    Uses CLIP ViT vision-language representations for zero-shot text-conditioned detection.
    """

    @Profiler
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        vocabulary: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        query_batch_size: int = 40,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.vocabulary = vocabulary if vocabulary is not None else list(FRIDGE_FOOD_ITEMS)
        self.confidence_threshold = confidence_threshold
        self.query_batch_size = query_batch_size

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = None
        self.model = None
        logger.info(
            f"[OwlV2Detector] Initialized detector: model_id='{self.model_id}', "
            f"vocab_size={len(self.vocabulary)} items, device='{self.device}', "
            f"confidence_threshold={self.confidence_threshold}"
        )

    @Profiler
    def load_model(self) -> None:
        """Loads OWLv2 processor and model weights."""
        if self.processor is not None and self.model is not None:
            logger.debug("[OwlV2Detector] Model weights and processor are already in memory. Skipping reload.")
            return

        start_time = time.perf_counter()
        logger.info(
            f"[OwlV2Detector] Loading processor & model weights: '{self.model_id}' "
            f"(cache_dir='{self.cache_dir}') to device '{self.device}'..."
        )

        try:
            self.processor = Owlv2Processor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model = Owlv2ForObjectDetection.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            self.model = model.to(self.device)
            self.model.eval()

            elapsed = time.perf_counter() - start_time
            logger.info(f"[OwlV2Detector] Model loaded successfully on {self.device} (took {elapsed:.2f}s)")
        except Exception as exc:
            logger.exception(f"[OwlV2Detector] Error loading model '{self.model_id}': {exc}")
            raise

    @Profiler
    def detect(self, image: Image.Image) -> List[DetectedObject]:
        """
        Runs OWLv2 zero-shot detection on the PIL image across the vocabulary.
        """
        if self.processor is None or self.model is None:
            logger.info("[OwlV2Detector] Model not yet loaded. Performing just-in-time load...")
            self.load_model()

        inference_start = time.perf_counter()
        rgb_image = image.convert("RGB")
        target_sizes = torch.tensor([rgb_image.size[::-1]], device=self.device)  # (h, w)

        all_detections: List[DetectedObject] = []

        try:
            # Batch queries to prevent memory spikes
            vocab = self.vocabulary
            batch_size = self.query_batch_size
            query_batches = [vocab[i : i + batch_size] for i in range(0, len(vocab), batch_size)]

            for batch in query_batches:
                # Format text queries with standard prompt template
                text_queries = [[f"a photo of a {item}" for item in batch]]

                inputs = self.processor(
                    text=text_queries, images=rgb_image, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                results = self.processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.confidence_threshold,
                )[0]

                boxes = results["boxes"]
                scores = results["scores"]
                labels = results["labels"]

                for box, score, label_idx in zip(boxes, scores, labels):
                    idx = int(label_idx.item())
                    if 0 <= idx < len(batch):
                        raw_label = batch[idx]
                    else:
                        raw_label = str(idx)

                    cleaned = clean_label(raw_label)
                    bbox = [round(float(coord.item()), 2) for coord in box]
                    all_detections.append(
                        DetectedObject(
                            label=cleaned,
                            confidence=round(float(score.item()), 4),
                            bbox=bbox,
                        )
                    )

            # Apply NMS to merge overlapping predictions
            nms_detections = apply_nms(all_detections, iou_threshold=0.5)

            total_elapsed_ms = (time.perf_counter() - inference_start) * 1000
            unique_labels = sorted(list(set(d.label for d in nms_detections)))
            logger.info(
                f"[OwlV2Detector] Detection finished: {len(all_detections)} raw -> "
                f"{len(nms_detections)} after NMS across {len(unique_labels)} classes. "
                f"Time: {total_elapsed_ms:.2f}ms"
            )

            return nms_detections

        except Exception as exc:
            logger.exception(f"[OwlV2Detector] Error during OWLv2 inference: {exc}")
            raise
