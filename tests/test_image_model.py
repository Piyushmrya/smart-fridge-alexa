import pytest

# These tests need the classifier extras: pip install -r classifier/requirements.txt
pytest.importorskip("torch")
pytest.importorskip("transformers")

import pytest
from PIL import Image

from classifier.image_model.base import BaseCamera, BaseDetector, DetectedObject
from classifier.image_model.camera import DummyCamera, DEFAULT_SAMPLE_IMAGE_PATH
from classifier.image_model.grounding_dino import GroundingDinoDetector, clean_label


def test_clean_label():
    assert clean_label("bottle bottle") == "bottle"
    assert clean_label("milk bottle bottle") == "milk bottle"
    assert clean_label("spinach") == "spinach"
    assert clean_label("  tomato  ") == "tomato"


def test_dummy_camera_reads_sample_image():
    camera = DummyCamera()
    assert camera.image_path == DEFAULT_SAMPLE_IMAGE_PATH
    img = camera.capture_frame()
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.width > 0 and img.height > 0


def test_dummy_camera_raises_when_file_not_found():
    camera = DummyCamera(image_path="/nonexistent/path/fridge.png")
    with pytest.raises(FileNotFoundError):
        camera.capture_frame()


def test_model_agnostic_detector_interface():
    class CustomMockDetector(BaseDetector):
        def detect(self, image: Image.Image):
            return [
                DetectedObject(label="apple", confidence=0.95, bbox=[10.0, 10.0, 50.0, 50.0])
            ]

    detector: BaseDetector = CustomMockDetector()
    dummy_img = Image.new("RGB", (100, 100))
    results = detector.detect(dummy_img)

    assert len(results) == 1
    assert results[0].label == "apple"
    assert results[0].confidence == 0.95
    assert results[0].bbox == [10.0, 10.0, 50.0, 50.0]


def test_grounding_dino_detector_structure():
    detector = GroundingDinoDetector()
    assert isinstance(detector, BaseDetector)
    assert detector.confidence_threshold == 0.25
    assert len(detector.prompt_chunks) > 0
    assert detector.total_vocab_size >= 200


def test_fridge_food_vocabulary_contents():
    from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS, get_food_prompt_chunks

    assert len(FRIDGE_FOOD_ITEMS) >= 200

    # Ensure non-food container/packaging items are excluded
    forbidden = ["bottle", "jar", "bowl", "plastic bag", "container", "box"]
    for item in FRIDGE_FOOD_ITEMS:
        assert item.lower() not in forbidden, f"Found non-food container item: {item}"

    # Verify chunking
    chunks = get_food_prompt_chunks(FRIDGE_FOOD_ITEMS, chunk_size=40)
    assert len(chunks) >= 5
    for chunk in chunks:
        # Verify chunk format
        assert " . " in chunk
