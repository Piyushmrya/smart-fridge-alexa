import json
import logging
import os
import shutil
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from classifier.image_model.base import BaseDetector
from classifier.image_model.grounding_dino import GroundingDinoDetector
from classifier.image_model.yolo_world import YoloWorldDetector
from classifier.image_model.owlv2 import OwlV2Detector
from classifier.image_model.florence2 import Florence2Detector
from classifier.image_model.claude_detector import ClaudeVisionDetector
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_config.json"
)

# Approximate disk sizes for models if a download is required
MODEL_APPROX_SIZES = {
    "yolo_world": "25 - 60 MB",
    "grounding_dino": "689 MB",
    "owlv2": "600 MB",
    "florence2": "460 MB",
    "claude": "Cloud API (0 MB)",
    "claude_vision": "Cloud API (0 MB)",
    "llm": "Cloud API (0 MB)",
}


def get_available_disk_space_gb(path: str = "/") -> float:
    """Returns available disk space in gigabytes at the given path."""
    try:
        total, used, free = shutil.disk_usage(path)
        return round(free / (1024**3), 2)
    except Exception:
        return -1.0


class DetectorConfig(BaseModel):
    """Configuration model for vision detectors loaded from JSON."""

    detector_type: Literal[
        "yolo_world", "grounding_dino", "owlv2", "florence2", "claude", "claude_vision", "llm"
    ] = Field(
        "yolo_world",
        description="Vision model type to use ('yolo_world', 'grounding_dino', 'owlv2', 'florence2', 'claude')",
    )
    model_id: Optional[str] = Field(
        None, description="Hugging Face repo ID or local checkpoint path"
    )
    confidence_threshold: Optional[float] = Field(
        None, description="Detection confidence threshold (0.0 to 1.0)"
    )
    device: Optional[str] = Field("auto", description="'cuda', 'cpu', or 'auto'")
    cache_dir: Optional[str] = Field(None, description="Custom directory for model weights cache")
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific options"
    )


@Profiler
def load_detector_from_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> BaseDetector:
    """
    Factory function that loads and instantiates the configured vision detector.
    Reads configuration from JSON file (or dictionary).

    Args:
        config_path: Path to the JSON configuration file (defaults to MODEL_CONFIG_PATH env var or classifier/model_config.json).
        config_dict: Optional pre-parsed dictionary to override file loading.

    Returns:
        Configured BaseDetector instance.
    """
    if config_dict is not None:
        raw_config = config_dict
        source_desc = "in-memory dictionary"
    else:
        if config_path is None:
            config_path = os.environ.get("MODEL_CONFIG_PATH", DEFAULT_CONFIG_FILE)

        source_desc = config_path
        if not os.path.exists(config_path):
            logger.warning(
                f"[DetectorFactory] Config file '{config_path}' not found! "
                f"Falling back to default 'yolo_world' detector."
            )
            raw_config = {"detector_type": "yolo_world"}
        else:
            logger.info(f"[DetectorFactory] Reading vision detector configuration from: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)

    cfg = DetectorConfig(**raw_config)

    # Disk space check
    free_gb = get_available_disk_space_gb(os.path.dirname(DEFAULT_CONFIG_FILE))
    approx_size = MODEL_APPROX_SIZES.get(cfg.detector_type, "unknown")
    logger.info(
        f"[DetectorFactory] Selected detector: '{cfg.detector_type}' from {source_desc}. "
        f"(Model size: ~{approx_size}, Available disk space: {free_gb:.2f} GB)"
    )

    det_type = cfg.detector_type.lower()

    if det_type == "yolo_world":
        kwargs: Dict[str, Any] = {}
        if cfg.model_id:
            kwargs["model_id"] = cfg.model_id
        if cfg.confidence_threshold is not None:
            kwargs["confidence_threshold"] = cfg.confidence_threshold
        if cfg.device:
            kwargs["device"] = cfg.device
        if "iou_threshold" in cfg.options:
            kwargs["iou_threshold"] = cfg.options["iou_threshold"]
        return YoloWorldDetector(**kwargs)

    elif det_type == "grounding_dino":
        kwargs = {}
        if cfg.model_id:
            kwargs["model_id"] = cfg.model_id
        if cfg.cache_dir:
            kwargs["cache_dir"] = cfg.cache_dir
        if cfg.confidence_threshold is not None:
            kwargs["confidence_threshold"] = cfg.confidence_threshold
        if cfg.device:
            kwargs["device"] = cfg.device
        if "chunk_size" in cfg.options:
            kwargs["chunk_size"] = cfg.options["chunk_size"]
        return GroundingDinoDetector(**kwargs)

    elif det_type == "owlv2":
        kwargs = {}
        if cfg.model_id:
            kwargs["model_id"] = cfg.model_id
        if cfg.cache_dir:
            kwargs["cache_dir"] = cfg.cache_dir
        if cfg.confidence_threshold is not None:
            kwargs["confidence_threshold"] = cfg.confidence_threshold
        if cfg.device:
            kwargs["device"] = cfg.device
        if "query_batch_size" in cfg.options:
            kwargs["query_batch_size"] = cfg.options["query_batch_size"]
        return OwlV2Detector(**kwargs)

    elif det_type == "florence2":
        kwargs = {}
        if cfg.model_id:
            kwargs["model_id"] = cfg.model_id
        if cfg.cache_dir:
            kwargs["cache_dir"] = cfg.cache_dir
        if cfg.confidence_threshold is not None:
            kwargs["confidence_threshold"] = cfg.confidence_threshold
        if cfg.device:
            kwargs["device"] = cfg.device
        if "task_prompt" in cfg.options:
            kwargs["task_prompt"] = cfg.options["task_prompt"]
        return Florence2Detector(**kwargs)

    elif det_type in ("claude", "claude_vision", "llm"):
        kwargs = {}
        if cfg.model_id:
            kwargs["model_id"] = cfg.model_id
        if cfg.confidence_threshold is not None:
            kwargs["confidence_threshold"] = cfg.confidence_threshold
        for opt_key in (
            "api_key",
            "max_image_dim",
            "max_tokens",
            "temperature",
            "apply_dedup",
            "iou_threshold",
        ):
            if opt_key in cfg.options:
                kwargs[opt_key] = cfg.options[opt_key]
        return ClaudeVisionDetector(**kwargs)

    else:
        raise ValueError(
            f"Unsupported detector_type '{cfg.detector_type}'. "
            f"Must be one of: 'yolo_world', 'grounding_dino', 'owlv2', 'florence2', 'claude'."
        )
