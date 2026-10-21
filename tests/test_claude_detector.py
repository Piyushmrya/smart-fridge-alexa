import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from classifier.image_model.base import BaseDetector, DetectedObject
from classifier.image_model.claude_detector import ClaudeVisionDetector, _parse_json_from_response
from classifier.image_model.factory import DetectorConfig, load_detector_from_config


def test_parse_json_from_response():
    # Test raw json
    raw = '{"detections": [{"label": "apple", "confidence": 0.9, "box_2d": [100, 100, 200, 200]}]}'
    res = _parse_json_from_response(raw)
    assert len(res["detections"]) == 1

    # Test markdown fenced json
    fenced = '```json\n{"detections": [{"label": "milk", "confidence": 0.85, "box_2d": [50, 50, 300, 200]}]}\n```'
    res_fenced = _parse_json_from_response(fenced)
    assert res_fenced["detections"][0]["label"] == "milk"

    # Test conversational wrapper with json block
    conversational = 'Here are the items I detected:\n{"detections": [{"label": "butter", "confidence": 0.95, "box_2d": [10, 20, 30, 40]}]}\nHope this helps!'
    res_conv = _parse_json_from_response(conversational)
    assert res_conv["detections"][0]["label"] == "butter"

    # Test invalid json raises ValueError
    with pytest.raises(ValueError):
        _parse_json_from_response("No json here at all.")


def test_claude_vision_detector_init():
    detector = ClaudeVisionDetector(
        model_id="claude-sonnet-5-5",
        confidence_threshold=0.3,
        api_key="mock_key",
    )
    assert isinstance(detector, BaseDetector)
    assert detector.model_id == "claude-sonnet-5-5"
    assert detector.confidence_threshold == 0.3
    assert detector.client is None


def test_claude_vision_detector_factory():
    cfg = DetectorConfig(
        detector_type="claude",
        model_id="claude-sonnet-5-5",
        confidence_threshold=0.25,
        options={"api_key": "mock_key", "temperature": 0.2},
    )
    assert cfg.detector_type == "claude"

    detector = load_detector_from_config(
        config_dict={
            "detector_type": "claude",
            "model_id": "claude-sonnet-5-5",
            "confidence_threshold": 0.25,
            "options": {"api_key": "mock_key"},
        }
    )
    assert isinstance(detector, ClaudeVisionDetector)
    assert detector.confidence_threshold == 0.25


def test_claude_vision_detector_normalize_box():
    orig_w, orig_h = 1000, 500

    # 1. Normalized 0-1000: [ymin, xmin, ymax, xmax] -> [100, 200, 400, 600]
    # ymin=100 -> y1=50, xmin=200 -> x1=200, ymax=400 -> y2=200, xmax=600 -> x2=600
    det_norm = ClaudeVisionDetector(api_key="mock_key", coord_format="normalized_1000")
    box = det_norm._normalize_box([100, 200, 400, 600], orig_w, orig_h)
    assert box == [200.0, 50.0, 600.0, 200.0]

    # 2. Pixel format: [ymin, xmin, ymax, xmax] -> [50, 200, 200, 600]
    det_px = ClaudeVisionDetector(api_key="mock_key", coord_format="pixel")
    box_px = det_px._normalize_box([50, 200, 200, 600], orig_w, orig_h)
    assert box_px == [200.0, 50.0, 600.0, 200.0]

    # 3. Normalized 0-1.0: [ymin, xmin, ymax, xmax] -> [0.1, 0.2, 0.4, 0.6]
    box_float = det_norm._normalize_box([0.1, 0.2, 0.4, 0.6], orig_w, orig_h)
    assert box_float == [200.0, 50.0, 600.0, 200.0]

    # 4. Degenerate box (< 2px width/height) rejected
    assert det_norm._normalize_box([10, 10, 10, 10], orig_w, orig_h) is None

    # 5. Invalid length / non-numeric rejected
    assert det_norm._normalize_box([10, 20], orig_w, orig_h) is None
    assert det_norm._normalize_box("invalid", orig_w, orig_h) is None


def test_claude_vision_detector_detect_mock():
    detector = ClaudeVisionDetector(
        model_id="claude-sonnet-5-5",
        api_key="mock_key",
        confidence_threshold=0.20,
        coord_format="normalized_1000",
    )

    mock_client = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = json.dumps({
        "detections": [
            {
                "label": "tomato",
                "confidence": 0.95,
                "box_2d": [100, 200, 400, 500],
            },
            {
                "label": "butter",
                "confidence": 0.88,
                "box_2d": [500, 100, 800, 400],
            },
            {
                "label": "low_conf_item",
                "confidence": 0.10,  # Below threshold
                "box_2d": [10, 10, 50, 50],
            },
        ]
    })
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_client.messages.create.return_value = mock_response

    detector.client = mock_client

    img = Image.new("RGB", (800, 600), color="white")
    results = detector.detect(img)

    assert len(results) == 2
    assert results[0].label == "tomato"
    assert results[0].confidence == 0.95
    # For tomato: orig_w=800, orig_h=600, box_2d=[100, 200, 400, 500] (ymin, xmin, ymax, xmax)
    # xmin=200/1000*800=160, ymin=100/1000*600=60, xmax=500/1000*800=400, ymax=400/1000*600=240
    assert results[0].bbox == [160.0, 60.0, 400.0, 240.0]

    assert results[1].label == "butter"
    assert results[1].confidence == 0.88
