from typing import Optional
from classifier.api.service import InventoryService
from classifier.image_model.camera import DummyCamera
from classifier.image_model.factory import load_detector_from_config
from classifier.profiler import Profiler

_inventory_service_instance: Optional[InventoryService] = None


@Profiler
def get_inventory_service() -> InventoryService:
    """
    FastAPI dependency that returns the configured InventoryService instance.
    Can be easily overridden in tests using app.dependency_overrides.
    """
    global _inventory_service_instance
    if _inventory_service_instance is None:
        camera = DummyCamera()
        detector = load_detector_from_config()
        _inventory_service_instance = InventoryService(camera=camera, detector=detector)
    return _inventory_service_instance


def set_inventory_service(service: Optional[InventoryService]) -> None:
    """Helper to set or reset the inventory service instance."""
    global _inventory_service_instance
    _inventory_service_instance = service
