import pytest

# These tests need the classifier extras: pip install -r classifier/requirements.txt
pytest.importorskip("torch")
pytest.importorskip("transformers")

from typing import List
from PIL import Image
from starlette.testclient import TestClient

from classifier.api.main import app
from classifier.api.dependencies import get_inventory_service
from classifier.api.service import InventoryService
from classifier.api.models import PatchInventoryItem
from classifier.image_model.base import BaseCamera, BaseDetector, DetectedObject


class MockCamera(BaseCamera):
    def capture_frame(self) -> Image.Image:
        return Image.new("RGB", (200, 200), color="white")


class MockDetector(BaseDetector):
    def __init__(self, detections: List[DetectedObject]):
        self.detections = detections

    def detect(self, image: Image.Image) -> List[DetectedObject]:
        return self.detections


def test_inventory_service_sync_and_manual_override():
    initial_detections = [
        DetectedObject(label="spinach", confidence=0.85, bbox=[10.0, 10.0, 50.0, 50.0]),
        DetectedObject(label="tomato", confidence=0.75, bbox=[60.0, 60.0, 100.0, 100.0]),
    ]
    camera = MockCamera()
    detector = MockDetector(initial_detections)
    service = InventoryService(camera=camera, detector=detector)

    # First sync
    res = service.sync_inventory()
    assert len(res.items) == 2
    spinach_item = next(it for it in res.items if it.label == "spinach")
    assert spinach_item.manual_override is False

    # Patch spinach to kale with manual override
    service.patch_inventory([
        PatchInventoryItem(id=spinach_item.id, label="organic kale", confidence=0.99)
    ])

    assert service._inventory[spinach_item.id].label == "organic kale"
    assert service._inventory[spinach_item.id].manual_override is True

    # Run another sync with detector still seeing "spinach" at the same bbox
    second_sync = service.sync_inventory()
    updated_item = next(it for it in second_sync.items if it.id == spinach_item.id)
    # Manual override must be respected!
    assert updated_item.label == "organic kale"
    assert updated_item.manual_override is True


def test_api_endpoints_with_mock_service():
    detections = [
        DetectedObject(label="milk", confidence=0.92, bbox=[20.0, 20.0, 60.0, 120.0]),
        DetectedObject(label="egg", confidence=0.88, bbox=[80.0, 80.0, 120.0, 120.0]),
    ]
    mock_service = InventoryService(camera=MockCamera(), detector=MockDetector(detections))
    mock_service.sync_inventory()

    # Override dependency in FastAPI
    app.dependency_overrides[get_inventory_service] = lambda: mock_service

    with TestClient(app) as client:
        # GET /inventory
        get_res = client.get("/inventory")
        assert get_res.status_code == 200
        data = get_res.json()
        assert "items" in data
        assert len(data["items"]) == 2
        item_id = data["items"][0]["id"]

        # PATCH /inventory
        patch_res = client.patch(
            "/inventory",
            json=[{"id": item_id, "label": "oat milk", "confidence": 0.99}],
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()
        assert patch_data["success"] is True
        assert patch_data["updated_items"][0]["label"] == "oat milk"
        assert patch_data["updated_items"][0]["manual_override"] is True

        # GET /inventory with force_refresh
        refresh_res = client.get("/inventory?force_refresh=true")
        assert refresh_res.status_code == 200
        refresh_data = refresh_res.json()
        patched_in_refresh = next(it for it in refresh_data["items"] if it["id"] == item_id)
        assert patched_in_refresh["label"] == "oat milk"

        # PATCH with non-existent ID -> 404
        bad_patch = client.patch(
            "/inventory",
            json=[{"id": "nonexistent_item_999", "label": "butter"}],
        )
        assert bad_patch.status_code == 404
        assert bad_patch.json()["success"] is False

    app.dependency_overrides.clear()


def test_end_to_end_real_grounding_dino_integration():
    """
    Test real end-to-end flow using the default DummyCamera (reading sample_fridge.png)
    and GroundingDinoDetector via FastAPI test client.
    """
    from classifier.image_model.grounding_dino import GroundingDinoDetector
    from classifier.image_model.camera import DummyCamera
    service = InventoryService(camera=DummyCamera(), detector=GroundingDinoDetector())
    app.dependency_overrides[get_inventory_service] = lambda: service

    try:
        with TestClient(app) as client:
            # GET /inventory triggers detection or retrieves initial state
            res = client.get("/inventory")
            assert res.status_code == 200
            data = res.json()
            assert "items" in data
            assert "last_updated_epoch" in data
            assert len(data["items"]) > 0

            # Verify that detected items have labels, confidences, and valid bboxes
            first_item = data["items"][0]
            assert "id" in first_item
            assert "label" in first_item
            assert first_item["confidence"] > 0
            assert first_item["bbox"] is not None
            assert len(first_item["bbox"]) == 4

            # Test patch on a detected item
            patch_res = client.patch(
                "/inventory",
                json=[{"id": first_item["id"], "label": "custom_labeled_item"}],
            )
            assert patch_res.status_code == 200
            patch_data = patch_res.json()
            assert patch_data["success"] is True
            assert patch_data["updated_items"][0]["label"] == "custom_labeled_item"
            assert patch_data["updated_items"][0]["manual_override"] is True
    finally:
        app.dependency_overrides.clear()

