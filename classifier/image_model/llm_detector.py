"""LLM-based object detector for smart refrigerator images.

Model-agnostic vision detector that delegates API invocation, multi-provider dispatch
(Anthropic, OpenAI), token/thinking tracking, and cost calculation to the common LLMClient.
"""

import base64
import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.grounding_dino import clean_label, apply_nms
from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS
from classifier.llm import LLMClient, LLMResponse, get_llm_client
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-5-5")
DEFAULT_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_MAX_IMAGE_DIM = 1024
DEFAULT_MAX_TOKENS = 1500


def _parse_json_from_response(text: str) -> Dict[str, Any]:
    """Extracts and parses JSON object from model response text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    start_idx = text.find("{")
    if start_idx != -1:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback to regex search
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError(f"Could not parse valid JSON from LLM response:\n{text[:200]}")


class LLMVisionDetector(BaseDetector):
    """
    Object detector powered by multimodal LLMs (Claude, GPT-4o, etc.).
    Detects fridge food items with bounding boxes and normalized confidence scores.
    """

    @Profiler
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        vocabulary: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_image_dim: int = DEFAULT_MAX_IMAGE_DIM,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.1,
        apply_dedup: bool = True,
        iou_threshold: float = 0.5,
        coord_format: str = "pixel",
        llm_client: Optional[LLMClient] = None,
    ):
        self.model_id = model_id
        self.provider = provider
        self._api_key = api_key
        self.vocabulary = vocabulary if vocabulary is not None else list(FRIDGE_FOOD_ITEMS)
        self.confidence_threshold = confidence_threshold
        self.max_image_dim = max_image_dim
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.apply_dedup = apply_dedup
        self.iou_threshold = iou_threshold
        self.coord_format = coord_format
        self.llm_client = llm_client or get_llm_client()

        logger.info(
            f"[LLMVisionDetector] Initialized detector: model_id='{self.model_id}', "
            f"provider='{self.provider or 'auto'}', vocab_size={len(self.vocabulary)} items, "
            f"confidence_threshold={self.confidence_threshold}"
        )

    @Profiler
    def load_model(self) -> None:
        """Verifies configuration and client readiness."""
        logger.info(
            f"[LLMVisionDetector] Ready with model '{self.model_id}' (provider: {self.provider or 'auto'})"
        )

    def _prepare_image_payload(self, image: Image.Image) -> Tuple[str, int, int]:
        """Converts PIL image to JPEG base64 string, scaling down if needed."""
        orig_w, orig_h = image.size
        img = image.convert("RGB")

        # Scale down large images to reduce API latency and payload size
        if max(orig_w, orig_h) > self.max_image_dim:
            img_to_send = img.copy()
            img_to_send.thumbnail((self.max_image_dim, self.max_image_dim), Image.Resampling.LANCZOS)
        else:
            img_to_send = img

        buf = io.BytesIO()
        img_to_send.save(buf, format="JPEG", quality=85)
        base64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return base64_data, orig_w, orig_h

    def _normalize_box(
        self, raw_box: List[Any], orig_w: int, orig_h: int
    ) -> Optional[List[float]]:
        """Converts bounding box coordinates into absolute pixel coordinates [x1, y1, x2, y2]."""
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            return None

        try:
            c1, c2, c3, c4 = [float(v) for v in raw_box]
        except (ValueError, TypeError):
            return None

        # Check if coordinates are normalized 0-1.0
        if max(c1, c2, c3, c4) <= 1.0:
            ymin, xmin, ymax, xmax = c1, c2, c3, c4
            x1 = xmin * orig_w
            y1 = ymin * orig_h
            x2 = xmax * orig_w
            y2 = ymax * orig_h
        elif self.coord_format == "normalized_1000" and max(c1, c2, c3, c4) <= 1000.0:
            ymin, xmin, ymax, xmax = c1, c2, c3, c4
            x1 = (xmin / 1000.0) * orig_w
            y1 = (ymin / 1000.0) * orig_h
            x2 = (xmax / 1000.0) * orig_w
            y2 = (ymax / 1000.0) * orig_h
        else:
            # Pixel coordinates: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = c1, c2, c3, c4
            x1 = xmin
            y1 = ymin
            x2 = xmax
            y2 = ymax

        # Clamp coordinates to image boundaries
        x1 = max(0.0, min(float(orig_w), x1))
        y1 = max(0.0, min(float(orig_h), y1))
        x2 = max(0.0, min(float(orig_w), x2))
        y2 = max(0.0, min(float(orig_h), y2))

        # Validate box has positive area
        if x2 <= x1 or y2 <= y1:
            return None

        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    @Profiler
    def detect(self, image: Image.Image) -> List[DetectedObject]:
        """Detects food items in the provided fridge image via common LLMClient."""
        b64_image, orig_w, orig_h = self._prepare_image_payload(image)

        vocab_str = ", ".join(self.vocabulary[:150])
        system_prompt = (
            "You are an expert food and grocery detector for a smart refrigerator. "
            "Identify all food, drink, produce, ingredient, and grocery items inside the refrigerator image. "
            f"Image dimensions: width={orig_w}px, height={orig_h}px.\n"
            "Prefer standard item names matching this vocabulary where applicable:\n"
            f"{vocab_str}\n\n"
            "Respond ONLY with a valid JSON object without markdown fences, in this exact format:\n"
            "{\n"
            '  "detections": [\n'
            '    {\n'
            '      "label": "<food item name in lowercase>",\n'
            '      "confidence": <float between 0.0 and 1.0>,\n'
            '      "box_2d": [<ymin>, <xmin>, <ymax>, <xmax>]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            f"where box_2d coordinates are [ymin, xmin, ymax, xmax] in pixel coordinates within width={orig_w} and height={orig_h}."
        )

        user_prompt = "Detect all food and grocery items visible in this refrigerator image with their bounding boxes."

        # Delegate execution, provider formatting, timing, and cost tracking completely to LLMClient
        llm_response = self.llm_client.complete_vision(
            prompt=user_prompt,
            image_b64=b64_image,
            system_prompt=system_prompt,
            model_id=self.model_id,
            provider=self.provider,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            op="detect_fridge_items",
            api_key=self._api_key,
        )

        parsed = _parse_json_from_response(llm_response.text)
        raw_detections = parsed.get("detections", [])

        detected_objects: List[DetectedObject] = []
        for det in raw_detections:
            if not isinstance(det, dict):
                continue
            label = clean_label(str(det.get("label", "")))
            if not label:
                continue

            try:
                confidence = float(det.get("confidence", 0.90))
            except (ValueError, TypeError):
                confidence = 0.90

            if confidence < self.confidence_threshold:
                continue

            box = self._normalize_box(det.get("box_2d", []), orig_w, orig_h)
            if box is None:
                continue

            detected_objects.append(
                DetectedObject(label=label, confidence=round(confidence, 3), bbox=box)
            )

        # Apply NMS deduplication if enabled
        if self.apply_dedup and detected_objects:
            detected_objects = apply_nms(
                detected_objects, iou_threshold=self.iou_threshold
            )

        logger.info(
            f"[LLMVisionDetector] Detected {len(detected_objects)} items "
            f"(tokens: in={llm_response.input_tokens}, out={llm_response.output_tokens}, "
            f"thinking={llm_response.thinking_tokens}, cost={llm_response.cost_usd})"
        )
        return detected_objects
