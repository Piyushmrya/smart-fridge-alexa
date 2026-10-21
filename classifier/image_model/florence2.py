import logging
import os
import time
from typing import List, Optional
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.grounding_dino import clean_label, apply_nms
from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "microsoft/Florence-2-base"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class Florence2Detector(BaseDetector):
    """
    Microsoft Florence-2 unified vision foundation model.
    Performs open-vocabulary phrase grounding and object detection.
    Note: Download size is ~460 MB.
    """

    @Profiler
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        vocabulary: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        task_prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.vocabulary = vocabulary if vocabulary is not None else list(FRIDGE_FOOD_ITEMS)
        self.confidence_threshold = confidence_threshold
        self.task_prompt = task_prompt

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = None
        self.model = None
        logger.info(
            f"[Florence2Detector] Initialized detector: model_id='{self.model_id}', "
            f"vocab_size={len(self.vocabulary)} items, device='{self.device}', "
            f"task_prompt='{self.task_prompt}'"
        )

    @Profiler
    def load_model(self) -> None:
        """Loads Florence-2 processor and model weights (lazy loaded)."""
        if self.processor is not None and self.model is not None:
            logger.debug("[Florence2Detector] Model already loaded. Skipping reload.")
            return

        start_time = time.perf_counter()
        logger.info(
            f"[Florence2Detector] Loading processor & model weights: '{self.model_id}' "
            f"(cache_dir='{self.cache_dir}') to device '{self.device}'..."
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self.model = model.to(self.device)
            self.model.eval()

            elapsed = time.perf_counter() - start_time
            logger.info(f"[Florence2Detector] Model loaded successfully on {self.device} (took {elapsed:.2f}s)")
        except Exception as exc:
            logger.exception(f"[Florence2Detector] Error loading model '{self.model_id}': {exc}")
            raise

    @Profiler
    def detect(self, image: Image.Image) -> List[DetectedObject]:
        """
        Runs Florence-2 phrase grounding across the target vocabulary.
        """
        if self.processor is None or self.model is None:
            logger.info("[Florence2Detector] Model not yet loaded. Performing just-in-time load...")
            self.load_model()

        inference_start = time.perf_counter()
        rgb_image = image.convert("RGB")
        w, h = rgb_image.size

        # In Florence-2, text prompt specifies task + caption/phrases
        caption_text = " . ".join(self.vocabulary[:60])  # Florence-2 works best with concise phrase list
        full_prompt = f"{self.task_prompt} {caption_text}"

        all_detections: List[DetectedObject] = []

        try:
            inputs = self.processor(
                text=full_prompt, images=rgb_image, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                    use_cache=False,
                )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(w, h),
            )

            # Extract boxes and labels from parsed result
            task_result = parsed_answer.get(self.task_prompt, {})
            bboxes = task_result.get("bboxes", [])
            labels = task_result.get("labels", [])

            for box, label in zip(bboxes, labels):
                cleaned = clean_label(str(label))
                bbox = [round(float(coord), 2) for coord in box]
                all_detections.append(
                    DetectedObject(
                        label=cleaned,
                        confidence=0.80,  # Florence-2 outputs discrete grounded spans
                        bbox=bbox,
                    )
                )

            deduped = apply_nms(all_detections, iou_threshold=0.5)
            total_elapsed_ms = (time.perf_counter() - inference_start) * 1000
            unique_labels = sorted(list(set(d.label for d in deduped)))
            logger.info(
                f"[Florence2Detector] Detection finished: {len(all_detections)} raw -> "
                f"{len(deduped)} after NMS across {len(unique_labels)} classes. "
                f"Time: {total_elapsed_ms:.2f}ms"
            )

            return deduped

        except Exception as exc:
            logger.exception(f"[Florence2Detector] Error during Florence-2 inference: {exc}")
            raise
