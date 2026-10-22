import json
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("ultralytics")

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.yolo_world import YoloWorldDetector
from classifier.image_model.owlv2 import OwlV2Detector
from classifier.image_model.florence2 import Florence2Detector
from classifier.image_model.grounding_dino import GroundingDinoDetector
from classifier.image_model.llm_detector import LLMVisionDetector
from classifier.image_model.factory import (
    DetectorConfig,
    load_detector_from_config,
    get_available_disk_space_gb,
)
from classifier.api.service import InventoryService
from classifier.image_model.camera import DummyCamera


def test_detector_config_pydantic_validation():
    cfg = DetectorConfig(
        detector_type="yolo_world",
        model_id="yolov8s-worldv2.pt",
        confidence_threshold=0.3,
        options={"iou_threshold": 0.45},
    )
    assert cfg.detector_type == "yolo_world"
    assert cfg.model_id == "yolov8s-worldv2.pt"
    assert cfg.confidence_threshold == 0.3
    assert cfg.options["iou_threshold"] == 0.45

    # Invalid detector type raises validation error
    with pytest.raises(Exception):
        DetectorConfig(detector_type="invalid_model_type")


def test_disk_space_checker():
    free_gb = get_available_disk_space_gb()
    assert isinstance(free_gb, float)
    assert free_gb > 0.0


def test_load_detector_from_config_dict():
    # Test YOLO-World
    yolo_det = load_detector_from_config(config_dict={"detector_type": "yolo_world"})
    assert isinstance(yolo_det, YoloWorldDetector)
    assert isinstance(yolo_det, BaseDetector)
    assert yolo_det.confidence_threshold == 0.20

    # Test Grounding DINO
    dino_det = load_detector_from_config(config_dict={"detector_type": "grounding_dino"})
    assert isinstance(dino_det, GroundingDinoDetector)
    assert isinstance(dino_det, BaseDetector)

    # Test OWLv2 (instantiation only, no eager download)
    owl_det = load_detector_from_config(config_dict={"detector_type": "owlv2"})
    assert isinstance(owl_det, OwlV2Detector)
    assert isinstance(owl_det, BaseDetector)
    assert owl_det.model is None  # Lazy loading preserved

    # Test Florence-2 (instantiation only, no eager download)
    flo_det = load_detector_from_config(config_dict={"detector_type": "florence2"})
    assert isinstance(flo_det, Florence2Detector)
    assert isinstance(flo_det, BaseDetector)
    assert flo_det.model is None  # Lazy loading preserved

    # Test LLM Vision (instantiation only, lazy connection)
    llm_det = load_detector_from_config(config_dict={"detector_type": "llm"})
    assert isinstance(llm_det, LLMVisionDetector)
    assert isinstance(llm_det, BaseDetector)

def test_load_detector_default():
    # Calling with default arguments should load llm from model_config.json
    detector = load_detector_from_config()
    assert isinstance(detector, LLMVisionDetector)
    assert isinstance(detector, BaseDetector)


def test_load_detector_from_json_file(tmp_path):
    custom_json = tmp_path / "custom_config.json"
    custom_json.write_text(
        json.dumps({
            "detector_type": "yolo_world",
            "model_id": "yolov8s-worldv2.pt",
            "confidence_threshold": 0.35,
            "options": {"iou_threshold": 0.4},
        })
    )

    detector = load_detector_from_config(config_path=str(custom_json))
    assert isinstance(detector, YoloWorldDetector)
    assert detector.confidence_threshold == 0.35
    assert detector.iou_threshold == 0.4


def test_yolo_world_detector_detect_on_blank_image():
    detector = YoloWorldDetector(confidence_threshold=0.20)
    img = Image.new("RGB", (100, 100), color="white")
    results = detector.detect(img)
    assert isinstance(results, list)
    for res in results:
        assert isinstance(res, DetectedObject)
        assert res.label
        assert 0.0 <= res.confidence <= 1.0


def test_inventory_service_with_yolo_world_detector():
    camera = DummyCamera()
    detector = YoloWorldDetector(confidence_threshold=0.20)
    service = InventoryService(camera=camera, detector=detector)

    response = service.sync_inventory()
    assert response.last_updated_epoch > 0
    assert isinstance(response.items, list)
    # The dummy camera reads sample_fridge.png where YOLO-World detects items!
    assert len(response.items) > 0
    first_item = response.items[0]
    assert first_item.label
    assert first_item.confidence > 0.0
    assert first_item.bbox is not None
    assert len(first_item.bbox) == 4
