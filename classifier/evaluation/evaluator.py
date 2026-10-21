import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.factory import load_detector_from_config
from classifier.evaluation.config import DatasetConfig, EvalConfig, MatchingConfig, ModelEvalConfig
from classifier.evaluation.metrics import (
    DatasetEvaluationResult,
    ImageEvaluationResult,
    aggregate_dataset_metrics,
    match_image_boxes,
)
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and standardizes ground truth bounding box annotations from JSON."""

    @classmethod
    def load(cls, config: DatasetConfig) -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
        """
        Loads dataset images and ground truth annotations.

        Returns:
            Dict mapping image_key -> (absolute_image_path, list of GT dicts with 'label' and 'bbox')
        """
        annot_path = Path(os.path.expanduser(config.annotations_path)).resolve()
        img_dir = Path(os.path.expanduser(config.images_dir)).resolve()

        if not annot_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annot_path}")
        if not img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {img_dir}")

        logger.info(f"[DatasetLoader] Loading annotations from: {annot_path}")
        with open(annot_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Standardize different annotation formats
        standardized_gt: Dict[str, List[Dict[str, Any]]] = {}

        if isinstance(raw_data, dict):
            # Format A: annotations.json format {"images": {"img.jpg": {"boxes": [...]}}}
            if "images" in raw_data and isinstance(raw_data["images"], dict):
                for img_key, img_info in raw_data["images"].items():
                    boxes = img_info.get("boxes", []) if isinstance(img_info, dict) else []
                    standardized_gt[img_key] = [
                        {"label": b["label"], "bbox": b["bbox"]}
                        for b in boxes if "label" in b and "bbox" in b and len(b["bbox"]) == 4
                    ]
            else:
                # Format B: ground_truth_bounding_boxes.json format {"img.jpg": {"detections": [...]}}
                for img_key, img_val in raw_data.items():
                    if isinstance(img_val, dict) and "detections" in img_val:
                        boxes = img_val["detections"]
                    elif isinstance(img_val, list):
                        boxes = img_val
                    else:
                        boxes = []

                    standardized_gt[img_key] = [
                        {"label": b["label"], "bbox": b["bbox"]}
                        for b in boxes if "label" in b and "bbox" in b and len(b["bbox"]) == 4
                    ]

        # Match annotation image keys to files on disk in img_dir
        dataset: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
        disk_files = {p.name: p for p in img_dir.iterdir() if p.is_file()}

        for img_name, boxes in standardized_gt.items():
            clean_name = os.path.basename(img_name)
            matched_path = None

            if clean_name in disk_files:
                matched_path = str(disk_files[clean_name])
            elif (img_dir / img_name).exists():
                matched_path = str(img_dir / img_name)

            if matched_path:
                dataset[clean_name] = (matched_path, boxes)
            else:
                logger.warning(
                    f"[DatasetLoader] Image file '{img_name}' listed in annotations not found in {img_dir}. Skipping."
                )

        total_boxes = sum(len(b) for _, b in dataset.values())
        logger.info(
            f"[DatasetLoader] Successfully loaded {len(dataset)} images with {total_boxes} ground truth boxes."
        )
        return dataset


class ModelEvaluator:
    """Evaluates a single model against a loaded ground truth dataset."""

    def __init__(
        self,
        model_config: ModelEvalConfig,
        matching_config: MatchingConfig,
        detector: Optional[BaseDetector] = None,
    ):
        self.model_config = model_config
        self.matching_config = matching_config

        if detector is not None:
            self.detector = detector
        else:
            config_dict = {
                "detector_type": model_config.detector_type,
                "model_id": model_config.model_id,
                "confidence_threshold": model_config.confidence_threshold,
                "device": model_config.device,
                "options": model_config.options,
            }
            logger.info(f"[ModelEvaluator] Initializing detector '{model_config.name}' via factory...")
            self.detector = load_detector_from_config(config_dict=config_dict)
            try:
                logger.info(f"[ModelEvaluator] Pre-warming detector '{model_config.name}' for accurate inference profiling...")
                self.detector.load_model()
            except Exception as e:
                logger.warning(f"[ModelEvaluator] Pre-warming failed (will fall back to JIT): {e}")

    @Profiler
    def evaluate(
        self, dataset: Dict[str, Tuple[str, List[Dict[str, Any]]]]
    ) -> DatasetEvaluationResult:
        """Runs evaluation over all images in the dataset and computes precision/recall/F1."""
        image_results: List[ImageEvaluationResult] = []
        total_inference_start = time.perf_counter()

        logger.info(
            f"[ModelEvaluator] >>> Starting evaluation for '{self.model_config.name}' "
            f"across {len(dataset)} images (IoU threshold: {self.matching_config.iou_threshold:.0%})..."
        )

        for img_name, (img_path, gt_boxes) in dataset.items():
            t0 = time.perf_counter()
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                predictions = self.detector.detect(rgb_img)

            img_ms = (time.perf_counter() - t0) * 1000

            img_result = match_image_boxes(
                predictions=predictions,
                ground_truths=gt_boxes,
                image_name=img_name,
                iou_threshold=self.matching_config.iou_threshold,
                case_sensitive=self.matching_config.case_sensitive,
                normalize_labels=self.matching_config.normalize_labels,
            )
            image_results.append(img_result)
            logger.debug(
                f"  [{img_name}] {img_result.matches_count} match(es) / {img_result.extracted} pred(s), "
                f"GT: {img_result.total_in_gt} | Prec: {img_result.precision:.1%}, Rec: {img_result.recall:.1%} "
                f"({img_ms:.1f}ms)"
            )

        total_time_ms = (time.perf_counter() - total_inference_start) * 1000

        result = aggregate_dataset_metrics(
            model_name=self.model_config.name,
            image_results=image_results,
            total_time_ms=total_time_ms,
            case_sensitive=self.matching_config.case_sensitive,
            normalize_labels=self.matching_config.normalize_labels,
        )

        logger.info(
            f"[ModelEvaluator] <<< Finished '{self.model_config.name}': "
            f"Matches(TP)={result.total_matches}, Extracted={result.total_extracted}, GT={result.total_in_gt} | "
            f"Precision={result.precision:.1%}, Recall={result.recall:.1%}, F1={result.f1_score:.1%} "
            f"({result.latency_ms_per_image:.1f} ms/image, {result.fps:.1f} FPS)"
        )
        return result


class BenchmarkRunner:
    """Orchestrates running multiple models and compiling comparative evaluation results."""

    def __init__(self, config: EvalConfig):
        self.config = config

    @Profiler
    def run(self) -> Dict[str, DatasetEvaluationResult]:
        """Loads dataset and evaluates all configured models."""
        dataset = DatasetLoader.load(self.config.dataset)
        results: Dict[str, DatasetEvaluationResult] = {}

        for model_cfg in self.config.models:
            logger.info(f"[BenchmarkRunner] Evaluating model: {model_cfg.name}...")
            evaluator = ModelEvaluator(
                model_config=model_cfg,
                matching_config=self.config.matching,
            )
            results[model_cfg.name] = evaluator.evaluate(dataset)

        return results
