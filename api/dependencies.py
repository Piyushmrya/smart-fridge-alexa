from typing import Optional
from api.service import InventoryService
from image_model.camera import DummyCamera
from image_model.grounding_dino import GroundingDinoDetector

_inventory_service_instance: Optional[InventoryService] = None


def get_inventory_service() -> InventoryService:
    """
    FastAPI dependency that returns the configured InventoryService instance.
    Can be easily overridden in tests using app.dependency_overrides.
    """
    global _inventory_service_instance
    if _inventory_service_instance is None:
        camera = DummyCamera()
        detector = GroundingDinoDetector()
        _inventory_service_instance = InventoryService(camera=camera, detector=detector)
    return _inventory_service_instance


def set_inventory_service(service: Optional[InventoryService]) -> None:
    """Helper to set or reset the inventory service instance."""
    global _inventory_service_instance
    _inventory_service_instance = service
