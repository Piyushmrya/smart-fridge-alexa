from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from classifier.image_model.base import DetectedObject
from classifier.image_model.grounding_dino import clean_label


def compute_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
    """
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    b1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    b2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def normalize_label(label: str, case_sensitive: bool = False, clean: bool = True) -> str:
    """Normalize label strings for robust comparison."""
    norm = label.strip()
    if clean:
        norm = clean_label(norm)
    if not case_sensitive:
        norm = norm.lower()
    return norm


class BoxMatch(BaseModel):
    """Represents a successful match between a prediction and ground truth box."""
    pred_label: str
    gt_label: str
    confidence: float
    iou: float
    pred_bbox: List[float]
    gt_bbox: List[float]


class ImageEvaluationResult(BaseModel):
    """Evaluation result for a single image."""
    image_name: str
    extracted: int = Field(..., description="Total predictions made by model")
    total_in_gt: int = Field(..., description="Total ground truth items in image")
    matches_count: int = Field(..., description="Number of true positive matches (IoU >= threshold)")
    false_positives_count: int = Field(..., description="Number of false positive predictions")
    false_negatives_count: int = Field(..., description="Number of missed ground truth items")
    precision: float = Field(..., description="matches / extracted (0.0 if extracted == 0)")
    recall: float = Field(..., description="matches / total_in_gt (0.0 if total_in_gt == 0)")
    f1_score: float = Field(..., description="Harmonic mean of precision and recall")
    matches: List[BoxMatch] = Field(default_factory=list)
    false_positives: List[Dict[str, Any]] = Field(default_factory=list)
    false_negatives: List[Dict[str, Any]] = Field(default_factory=list)


class ClassMetrics(BaseModel):
    """Aggregated evaluation metrics for a single food item class."""
    class_name: str
    total_in_gt: int = 0
    extracted: int = 0
    matches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


class DatasetEvaluationResult(BaseModel):
    """Overall benchmark results across the entire dataset for one model."""
    model_name: str
    total_images: int
    total_extracted: int
    total_in_gt: int
    total_matches: int
    total_false_positives: int
    total_false_negatives: int
    precision: float
    recall: float
    f1_score: float
    macro_precision: float
    macro_recall: float
    macro_f1_score: float
    per_class: Dict[str, ClassMetrics]
    per_image: List[ImageEvaluationResult]
    latency_ms_per_image: float = 0.0
    fps: float = 0.0


def match_image_boxes(
    predictions: List[DetectedObject],
    ground_truths: List[Dict[str, Any]],
    image_name: str = "",
    iou_threshold: float = 0.50,
    case_sensitive: bool = False,
    normalize_labels: bool = True,
) -> ImageEvaluationResult:
    """
    Perform greedy bipartite matching between predictions and ground truths for one image.

    Args:
        predictions: List of DetectedObject instances returned by detector.
        ground_truths: List of ground truth dicts with 'label' and 'bbox' [x1, y1, x2, y2].
        image_name: Identifier of the image.
        iou_threshold: Minimum IoU overlap to consider a valid match (default 0.50).
        case_sensitive: Whether label comparison is case sensitive.
        normalize_labels: Whether to clean up repeated tokens.

    Returns:
        ImageEvaluationResult containing matched True Positives, False Positives, and False Negatives.
    """
    extracted_count = len(predictions)
    gt_count = len(ground_truths)

    # Sort predictions by confidence descending so highest confidence gets first pick
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    matched_gt_indices = set()
    matches: List[BoxMatch] = []
    false_positives: List[Dict[str, Any]] = []

    for pred in sorted_preds:
        norm_pred_label = normalize_label(pred.label, case_sensitive, normalize_labels)

        best_gt_idx = None
        best_iou = 0.0

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue

            norm_gt_label = normalize_label(gt.get("label", ""), case_sensitive, normalize_labels)
            if norm_pred_label != norm_gt_label:
                continue

            iou = compute_iou(pred.bbox, gt.get("bbox"))
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)
            matched_gt = ground_truths[best_gt_idx]
            matches.append(
                BoxMatch(
                    pred_label=pred.label,
                    gt_label=matched_gt.get("label", ""),
                    confidence=pred.confidence,
                    iou=round(best_iou, 4),
                    pred_bbox=pred.bbox or [],
                    gt_bbox=matched_gt.get("bbox", []),
                )
            )
        else:
            false_positives.append({
                "label": pred.label,
                "confidence": pred.confidence,
                "bbox": pred.bbox or [],
            })

    # Unmatched ground truths are False Negatives
    false_negatives: List[Dict[str, Any]] = []
    for gt_idx, gt in enumerate(ground_truths):
        if gt_idx not in matched_gt_indices:
            false_negatives.append({
                "label": gt.get("label", ""),
                "bbox": gt.get("bbox", []),
            })

    # Precision = match / extracted
    precision = round(len(matches) / extracted_count, 4) if extracted_count > 0 else 0.0
    # Recall = match / total_in_gt
    recall = round(len(matches) / gt_count, 4) if gt_count > 0 else 0.0
    # F1 score
    f1 = (
        round(2 * (precision * recall) / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

    return ImageEvaluationResult(
        image_name=image_name,
        extracted=extracted_count,
        total_in_gt=gt_count,
        matches_count=len(matches),
        false_positives_count=len(false_positives),
        false_negatives_count=len(false_negatives),
        precision=precision,
        recall=recall,
        f1_score=f1,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def aggregate_dataset_metrics(
    model_name: str,
    image_results: List[ImageEvaluationResult],
    total_time_ms: float = 0.0,
    case_sensitive: bool = False,
    normalize_labels: bool = True,
) -> DatasetEvaluationResult:
    """
    Aggregate image-level evaluation results into dataset-level and per-class metrics.
    """
    total_images = len(image_results)
    total_extracted = sum(r.extracted for r in image_results)
    total_in_gt = sum(r.total_in_gt for r in image_results)
    total_matches = sum(r.matches_count for r in image_results)
    total_fp = sum(r.false_positives_count for r in image_results)
    total_fn = sum(r.false_negatives_count for r in image_results)

    # Micro averages (pooled across all classes)
    precision = round(total_matches / total_extracted, 4) if total_extracted > 0 else 0.0
    recall = round(total_matches / total_in_gt, 4) if total_in_gt > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    # Per-class aggregations
    class_stats: Dict[str, Dict[str, int]] = {}

    def get_or_create(cls_name: str):
        norm = normalize_label(cls_name, case_sensitive, normalize_labels)
        if norm not in class_stats:
            class_stats[norm] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "ext": 0}
        return class_stats[norm]

    for img_res in image_results:
        for m in img_res.matches:
            stats = get_or_create(m.gt_label)
            stats["tp"] += 1
            stats["gt"] += 1
            stats["ext"] += 1

        for fp in img_res.false_positives:
            stats = get_or_create(fp["label"])
            stats["fp"] += 1
            stats["ext"] += 1

        for fn in img_res.false_negatives:
            stats = get_or_create(fn["label"])
            stats["fn"] += 1
            stats["gt"] += 1

    per_class: Dict[str, ClassMetrics] = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for cls_name, counts in sorted(class_stats.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        c_ext = counts["ext"]
        c_gt = counts["gt"]

        c_prec = round(tp / c_ext, 4) if c_ext > 0 else 0.0
        c_rec = round(tp / c_gt, 4) if c_gt > 0 else 0.0
        c_f1 = round(2 * (c_prec * c_rec) / (c_prec + c_rec), 4) if (c_prec + c_rec) > 0 else 0.0

        per_class[cls_name] = ClassMetrics(
            class_name=cls_name,
            total_in_gt=c_gt,
            extracted=c_ext,
            matches=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=c_prec,
            recall=c_rec,
            f1_score=c_f1,
        )

        if c_gt > 0 or c_ext > 0:
            macro_precisions.append(c_prec)
            macro_recalls.append(c_rec)
            macro_f1s.append(c_f1)

    macro_precision = round(sum(macro_precisions) / len(macro_precisions), 4) if macro_precisions else 0.0
    macro_recall = round(sum(macro_recalls) / len(macro_recalls), 4) if macro_recalls else 0.0
    macro_f1 = round(sum(macro_f1s) / len(macro_f1s), 4) if macro_f1s else 0.0

    latency_per_img = round(total_time_ms / total_images, 2) if total_images > 0 else 0.0
    fps = round(1000.0 / latency_per_img, 2) if latency_per_img > 0 else 0.0

    return DatasetEvaluationResult(
        model_name=model_name,
        total_images=total_images,
        total_extracted=total_extracted,
        total_in_gt=total_in_gt,
        total_matches=total_matches,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1_score=macro_f1,
        per_class=per_class,
        per_image=image_results,
        latency_ms_per_image=latency_per_img,
        fps=fps,
    )
