import json
import pytest
from PIL import Image

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.evaluation.config import DatasetConfig, EvalConfig, MatchingConfig, ModelEvalConfig
from classifier.evaluation.evaluator import BenchmarkRunner, DatasetLoader, ModelEvaluator
from classifier.evaluation.metrics import (
    aggregate_dataset_metrics,
    compute_iou,
    match_image_boxes,
)
from classifier.evaluation.reporter import EvaluationReporter


def test_compute_iou_known_values():
    # Identical boxes
    box = [0.0, 0.0, 100.0, 100.0]
    assert compute_iou(box, box) == 1.0

    # Non-overlapping
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [20.0, 20.0, 30.0, 30.0]
    assert compute_iou(box_a, box_b) == 0.0

    # Partial overlap (50% area overlap)
    # Box1: [0, 0, 10, 10] (area 100)
    # Box2: [5, 0, 15, 10] (area 100)
    # Intersection: [5, 0, 10, 10] (area 50)
    # Union: 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
    b1 = [0.0, 0.0, 10.0, 10.0]
    b2 = [5.0, 0.0, 15.0, 10.0]
    assert pytest.approx(compute_iou(b1, b2), 0.001) == (50.0 / 150.0)

    # Empty / Invalid
    assert compute_iou(None, b1) == 0.0
    assert compute_iou([1.0], b1) == 0.0


def test_match_image_boxes_perfect_match():
    preds = [
        DetectedObject(label="tomato", confidence=0.90, bbox=[10.0, 10.0, 50.0, 50.0])
    ]
    gt = [
        {"label": "tomato", "bbox": [10.0, 10.0, 50.0, 50.0]}
    ]
    res = match_image_boxes(preds, gt, image_name="img1.jpg", iou_threshold=0.50)
    assert res.extracted == 1
    assert res.total_in_gt == 1
    assert res.matches_count == 1
    assert res.false_positives_count == 0
    assert res.false_negatives_count == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert len(res.matches) == 1
    assert res.matches[0].iou == 1.0


def test_match_image_boxes_different_label_no_match():
    # Overlap is 100%, but label differs
    preds = [
        DetectedObject(label="apple", confidence=0.90, bbox=[10.0, 10.0, 50.0, 50.0])
    ]
    gt = [
        {"label": "tomato", "bbox": [10.0, 10.0, 50.0, 50.0]}
    ]
    res = match_image_boxes(preds, gt, image_name="img1.jpg", iou_threshold=0.50)
    assert res.matches_count == 0
    assert res.false_positives_count == 1
    assert res.false_negatives_count == 1
    assert res.precision == 0.0
    assert res.recall == 0.0


def test_match_image_boxes_low_iou_no_match():
    # Same label, but IoU < 0.50
    preds = [
        DetectedObject(label="tomato", confidence=0.85, bbox=[0.0, 0.0, 10.0, 10.0])
    ]
    gt = [
        {"label": "tomato", "bbox": [5.0, 0.0, 15.0, 10.0]}  # IoU is 0.33 < 0.50
    ]
    res = match_image_boxes(preds, gt, iou_threshold=0.50)
    assert res.matches_count == 0
    assert res.false_positives_count == 1
    assert res.false_negatives_count == 1
    assert res.precision == 0.0
    assert res.recall == 0.0


def test_match_image_boxes_greedy_duplicate_suppression():
    # Two predictions overlapping the same GT box
    # The higher confidence one should claim the match, the lower one should become FP
    preds = [
        DetectedObject(label="milk", confidence=0.95, bbox=[10.0, 10.0, 50.0, 50.0]),
        DetectedObject(label="milk", confidence=0.70, bbox=[12.0, 12.0, 48.0, 48.0]),
    ]
    gt = [
        {"label": "milk", "bbox": [10.0, 10.0, 50.0, 50.0]}
    ]
    res = match_image_boxes(preds, gt, iou_threshold=0.50)
    assert res.extracted == 2
    assert res.total_in_gt == 1
    assert res.matches_count == 1
    assert res.false_positives_count == 1
    assert res.false_negatives_count == 0
    assert res.precision == 0.5  # 1 match / 2 extracted
    assert res.recall == 1.0     # 1 match / 1 GT
    assert res.matches[0].confidence == 0.95


def test_match_image_boxes_zero_division_cases():
    # Case 1: Model made 0 predictions
    res1 = match_image_boxes([], [{"label": "milk", "bbox": [0, 0, 10, 10]}])
    assert res1.extracted == 0
    assert res1.precision == 0.0
    assert res1.recall == 0.0
    assert res1.false_negatives_count == 1

    # Case 2: Ground truth has 0 items
    pred = DetectedObject(label="milk", confidence=0.8, bbox=[0, 0, 10, 10])
    res2 = match_image_boxes([pred], [])
    assert res2.total_in_gt == 0
    assert res2.precision == 0.0
    assert res2.recall == 0.0
    assert res2.false_positives_count == 1


def test_aggregate_dataset_metrics():
    # Image 1: 1 match out of 1 pred, 1 GT
    res1 = match_image_boxes(
        [DetectedObject(label="apple", confidence=0.9, bbox=[0, 0, 10, 10])],
        [{"label": "apple", "bbox": [0, 0, 10, 10]}],
        image_name="img1.jpg",
    )
    # Image 2: 1 match out of 2 preds, 2 GT (1 missed)
    res2 = match_image_boxes(
        [
            DetectedObject(label="banana", confidence=0.9, bbox=[0, 0, 10, 10]),
            DetectedObject(label="orange", confidence=0.7, bbox=[50, 50, 60, 60]),  # FP
        ],
        [
            {"label": "banana", "bbox": [0, 0, 10, 10]},
            {"label": "grapes", "bbox": [20, 20, 30, 30]},  # FN
        ],
        image_name="img2.jpg",
    )

    dataset_res = aggregate_dataset_metrics("TestModel", [res1, res2], total_time_ms=100.0)
    assert dataset_res.total_images == 2
    assert dataset_res.total_extracted == 3
    assert dataset_res.total_in_gt == 3
    assert dataset_res.total_matches == 2
    assert dataset_res.total_false_positives == 1
    assert dataset_res.total_false_negatives == 1
    assert pytest.approx(dataset_res.precision, 0.001) == (2.0 / 3.0)
    assert pytest.approx(dataset_res.recall, 0.001) == (2.0 / 3.0)
    assert dataset_res.latency_ms_per_image == 50.0
    assert dataset_res.fps == 20.0
    assert "apple" in dataset_res.per_class
    assert "banana" in dataset_res.per_class


def test_dataset_loader_and_reporter(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Create two dummy image files
    img1_path = img_dir / "sample1.jpg"
    img2_path = img_dir / "sample2.jpg"
    Image.new("RGB", (100, 100), color="white").save(img1_path)
    Image.new("RGB", (100, 100), color="white").save(img2_path)

    # Write annotations file (ground_truth_bounding_boxes.json format)
    annot_file = tmp_path / "ground_truth_bounding_boxes.json"
    annot_file.write_text(
        json.dumps({
            "sample1.jpg": {
                "detections": [
                    {"label": "tomato", "bbox": [10.0, 10.0, 40.0, 40.0]}
                ]
            },
            "sample2.jpg": {
                "detections": [
                    {"label": "milk", "bbox": [20.0, 20.0, 60.0, 80.0]}
                ]
            }
        })
    )

    cfg = DatasetConfig(
        images_dir=str(img_dir),
        annotations_path=str(annot_file),
    )
    dataset = DatasetLoader.load(cfg)
    assert len(dataset) == 2
    assert "sample1.jpg" in dataset
    assert len(dataset["sample1.jpg"][1]) == 1

    # Mock detector
    class MockDetector(BaseDetector):
        def detect(self, image: Image.Image):
            return [DetectedObject(label="tomato", confidence=0.88, bbox=[10.0, 10.0, 40.0, 40.0])]

    model_cfg = ModelEvalConfig(
        name="Mock-Detector",
        detector_type="yolo_world",
        confidence_threshold=0.2,
    )
    matching_cfg = MatchingConfig(iou_threshold=0.5)
    evaluator = ModelEvaluator(model_cfg, matching_cfg, detector=MockDetector())
    res = evaluator.evaluate(dataset)

    assert res.total_matches == 1
    assert res.total_extracted == 2
    assert res.total_in_gt == 2
    assert res.precision == 0.5
    assert res.recall == 0.5

    # Test reporting
    out_dir = tmp_path / "results"
    paths = EvaluationReporter.export_all({"Mock-Detector": res}, str(out_dir))
    assert (out_dir / "benchmark_summary.md").exists()
    assert (out_dir / "benchmark_summary.csv").exists()
    assert (out_dir / "per_class_summary.csv").exists()
    assert (out_dir / "evaluation_results.json").exists()
