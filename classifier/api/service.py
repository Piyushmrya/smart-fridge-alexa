import logging
import time
from typing import Callable, Dict, List, Optional, Union
from PIL import Image

from classifier.api.models import (
    InventoryItem,
    InventoryResponse,
    PatchInventoryItem,
    PatchInventoryResponse,
)
from classifier.image_model.base import BaseCamera, BaseDetector, DetectedObject
from classifier.image_model.factory import (
    DETECTOR_ALIASES,
    load_detector_by_name,
    normalize_detector_name,
)
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)


@Profiler
def compute_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    inter_area = inter_width * inter_height

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class ItemNotFoundError(Exception):
    """Raised when a patch targets an item ID not found in inventory."""
    @Profiler
    def __init__(self, item_id: str):
        super().__init__(f"Item id '{item_id}' not found in tracking inventory.")
        self.item_id = item_id


class InvalidDetectorError(Exception):
    """Raised when an unsupported detector is requested."""
    @Profiler
    def __init__(self, message: str):
        super().__init__(message)


class InventoryService:
    """
    Model-agnostic and hardware-agnostic inventory service.
    Coordinates image capture via BaseCamera, detection via BaseDetector,
    and maintains the persistent tracking state.
    """

    @Profiler
    def __init__(
        self,
        camera: BaseCamera,
        detector: Optional[BaseDetector] = None,
        detector_loader: Optional[Callable[[str], BaseDetector]] = None,
    ):
        self.camera = camera
        self._detector_cache: Dict[str, BaseDetector] = {}
        self._detector_loader = detector_loader or load_detector_by_name
        self._current_detector_name: Optional[str] = None

        if detector is not None:
            self.detector = detector
            det_type = getattr(detector, "detector_type", None)
            if not det_type:
                cls_name = detector.__class__.__name__.lower()
                if "yolo" in cls_name:
                    det_type = "yolo_world"
                elif "dino" in cls_name:
                    det_type = "grounding_dino"
                elif "owl" in cls_name:
                    det_type = "owlv2"
                elif "florence" in cls_name:
                    det_type = "florence2"
                else:
                    det_type = "llm"
            self._current_detector_name = det_type
            self._detector_cache[det_type] = detector
            if det_type == "llm":
                self._detector_cache["llm"] = detector
        else:
            self.detector = None

        self._inventory: Dict[str, InventoryItem] = {}
        self._first_seen_epoch: Dict[str, int] = {}
        self._last_updated_epoch: int = 0
        self._item_counter: int = 0
        logger.info(
            f"[InventoryService] Initialized with camera={camera.__class__.__name__}, "
            f"detector={self.detector.__class__.__name__ if self.detector else 'None (lazy)'}, "
            f"current_detector_name='{self._current_detector_name}'"
        )

    def register_detector(self, name: str, detector: BaseDetector) -> None:
        """Register a pre-instantiated detector under a given name in the cache."""
        canonical = normalize_detector_name(name) if name in DETECTOR_ALIASES else name.lower()
        self._detector_cache[canonical] = detector
        self._detector_cache[name.lower()] = detector

    @Profiler
    def normalize_detector_name(self, detector_name: str) -> str:
        """Validates and normalizes detector name to canonical form."""
        key = detector_name.strip().lower()
        if key in self._detector_cache:
            return key
        try:
            return normalize_detector_name(detector_name)
        except ValueError as err:
            raise InvalidDetectorError(str(err)) from err

    @Profiler
    def get_or_load_detector(self, detector_name: str) -> BaseDetector:
        """
        Retrieves a detector from in-memory cache or lazily instantiates it using the loader.
        """
        key = detector_name.strip().lower()
        if key in self._detector_cache:
            return self._detector_cache[key]

        canonical = self.normalize_detector_name(detector_name)
        if canonical in self._detector_cache:
            return self._detector_cache[canonical]

        logger.info(f"[InventoryService] Lazily loading detector '{canonical}' into cache...")
        det = self._detector_loader(canonical)
        self._detector_cache[canonical] = det
        self._detector_cache[key] = det
        return det

    @Profiler
    def _generate_item_id(self) -> str:
        self._item_counter += 1
        return f"item_{self._item_counter}"

    @Profiler
    def sync_inventory(self, detector: Optional[Union[str, BaseDetector]] = None) -> InventoryResponse:
        """
        Capture frame from camera, run detection, and update tracking inventory.
        Preserves manual overrides and continuous tracking state.
        """
        sync_start = time.perf_counter()
        now = int(time.time())

        # Determine active detector
        if isinstance(detector, BaseDetector):
            active_detector = detector
            det_name = getattr(detector, "detector_type", None) or detector.__class__.__name__.lower()
            self._detector_cache[det_name] = detector
            self._current_detector_name = det_name
            self.detector = active_detector
        elif isinstance(detector, str) and detector.strip():
            canonical = self.normalize_detector_name(detector)
            active_detector = self.get_or_load_detector(canonical)
            self._current_detector_name = canonical
            self.detector = active_detector
        elif self.detector is not None:
            active_detector = self.detector
            if not self._current_detector_name:
                self._current_detector_name = "llm"
        else:
            canonical = "llm"
            active_detector = self.get_or_load_detector(canonical)
            self._current_detector_name = canonical
            self.detector = active_detector

        logger.info(
            f"[InventoryService] >>> Starting inventory synchronization cycle at epoch {now} "
            f"using detector='{self._current_detector_name}' ({active_detector.__class__.__name__})..."
        )

        try:
            # 1. Image capture
            cam_start = time.perf_counter()
            frame: Image.Image = self.camera.capture_frame()
            cam_ms = (time.perf_counter() - cam_start) * 1000
            logger.info(
                f"[InventoryService] Camera capture complete in {cam_ms:.2f}ms. "
                f"Frame size: {frame.width}x{frame.height}"
            )

            # 2. Vision model inference
            det_start = time.perf_counter()
            detections: List[DetectedObject] = active_detector.detect(frame)
            det_ms = (time.perf_counter() - det_start) * 1000
            logger.info(
                f"[InventoryService] Model inference complete in {det_ms:.2f}ms. "
                f"Received {len(detections)} detection candidate(s)."
            )

            # 3. Match new detections with existing tracked items
            matched_existing_ids = set()
            new_inventory: Dict[str, InventoryItem] = {}
            matched_count = 0
            new_items_count = 0
            preserved_override_count = 0

            for det in detections:
                best_match_id: Optional[str] = None
                best_iou: float = 0.0

                # Find matching item by bbox IoU
                for item_id, item in self._inventory.items():
                    if item_id in matched_existing_ids:
                        continue
                    iou = compute_iou(item.bbox, det.bbox)
                    if iou > 0.4 and iou > best_iou:
                        best_iou = iou
                        best_match_id = item_id

                if best_match_id is not None:
                    matched_existing_ids.add(best_match_id)
                    matched_count += 1
                    existing_item = self._inventory[best_match_id]
                    first_seen = self._first_seen_epoch.get(best_match_id, now)
                    duration = max(existing_item.time_in_fridge_seconds, now - first_seen)

                    if existing_item.manual_override:
                        preserved_override_count += 1
                        logger.debug(
                            f"[InventoryService] Item '{best_match_id}' has manual_override=True! "
                            f"Preserving user label='{existing_item.label}', "
                            f"model detected='{det.label}' (IoU={best_iou:.2f})"
                        )
                        new_inventory[best_match_id] = existing_item.model_copy(
                            update={
                                "time_in_fridge_seconds": duration,
                                "bbox": det.bbox if det.bbox is not None else existing_item.bbox,
                            }
                        )
                    else:
                        new_inventory[best_match_id] = InventoryItem(
                            id=best_match_id,
                            label=det.label,
                            confidence=det.confidence,
                            time_in_fridge_seconds=duration,
                            manual_override=False,
                            bbox=det.bbox,
                        )
                else:
                    # New item detected
                    new_id = self._generate_item_id()
                    new_items_count += 1
                    self._first_seen_epoch[new_id] = now
                    logger.debug(
                        f"[InventoryService] Discovered new item '{new_id}': label='{det.label}', "
                        f"confidence={det.confidence:.2%}, bbox={det.bbox}"
                    )
                    new_inventory[new_id] = InventoryItem(
                        id=new_id,
                        label=det.label,
                        confidence=det.confidence,
                        time_in_fridge_seconds=0,
                        manual_override=False,
                        bbox=det.bbox,
                    )

            # Preserve any manually overridden items that were not re-detected in this frame
            for item_id, item in self._inventory.items():
                if item_id not in matched_existing_ids and item.manual_override:
                    first_seen = self._first_seen_epoch.get(item_id, now)
                    duration = max(item.time_in_fridge_seconds, now - first_seen)
                    new_inventory[item_id] = item.model_copy(
                        update={"time_in_fridge_seconds": duration}
                    )
                    logger.info(
                        f"[InventoryService] Retaining undetected item '{item_id}' due to active manual_override=True"
                    )

            self._inventory = new_inventory
            self._last_updated_epoch = now

            total_sync_ms = (time.perf_counter() - sync_start) * 1000
            logger.info(
                f"[InventoryService] <<< Sync finished in {total_sync_ms:.2f}ms. "
                f"Total tracked items: {len(self._inventory)} "
                f"(matched: {matched_count}, new: {new_items_count}, manual_overrides: {preserved_override_count})"
            )

            return InventoryResponse(
                last_updated_epoch=self._last_updated_epoch,
                items=list(self._inventory.values()),
            )

        except Exception as exc:
            logger.exception(f"[InventoryService] Synchronization failed with error: {exc}")
            raise

    @Profiler
    def get_inventory(
        self,
        force_refresh: bool = False,
        detector: Optional[str] = None,
    ) -> InventoryResponse:
        """
        Get current tracked inventory.
        
        Triggers camera capture and inference if:
        - force_refresh is True
        - a different detector model is requested than the last sync (Option A)
        - inventory has not been synced yet (epoch == 0)
        
        Otherwise, serves cached inventory with continuously updated presence durations.
        """
        now = int(time.time())

        if detector is not None and detector.strip():
            target_detector_name = self.normalize_detector_name(detector)
            detector_changed = (
                self._current_detector_name is not None
                and target_detector_name != self._current_detector_name
            )
        else:
            target_detector_name = self._current_detector_name or "llm"
            detector_changed = False

        if force_refresh or detector_changed or self._last_updated_epoch == 0:
            logger.info(
                f"[InventoryService] Triggering detection sync (force_refresh={force_refresh}, "
                f"detector_changed={detector_changed}, epoch={self._last_updated_epoch}, "
                f"detector='{target_detector_name}')..."
            )
            return self.sync_inventory(detector=target_detector_name)

        age_seconds = now - self._last_updated_epoch
        logger.info(
            f"[InventoryService] Serving cached inventory ({len(self._inventory)} items, detector='{self._current_detector_name}'). "
            f"Last updated: {self._last_updated_epoch} ({age_seconds}s ago)."
        )

        current_items = []
        for item_id, item in self._inventory.items():
            first_seen = self._first_seen_epoch.get(item_id, self._last_updated_epoch)
            duration = max(item.time_in_fridge_seconds, now - first_seen)
            current_items.append(
                item.model_copy(update={"time_in_fridge_seconds": duration})
            )

        return InventoryResponse(
            last_updated_epoch=self._last_updated_epoch,
            items=current_items,
        )

    @Profiler
    def patch_inventory(self, patches: List[PatchInventoryItem]) -> PatchInventoryResponse:
        """
        Batch partial update on tracked inventory items.
        """
        logger.info(f"[InventoryService] PATCH request received with {len(patches)} item patch(es)...")

        # Validate that all requested items exist
        for patch in patches:
            if patch.id not in self._inventory:
                err_msg = (
                    f"Item id '{patch.id}' not found in tracking inventory. "
                    f"Existing IDs: {list(self._inventory.keys())}"
                )
                logger.error(f"[InventoryService] Validation error: {err_msg}")
                raise ItemNotFoundError(patch.id)

        updated_items: List[InventoryItem] = []
        for patch in patches:
            item = self._inventory[patch.id]
            update_data = patch.model_dump(exclude_unset=True)

            # If user modifies label, confidence, etc., default manual_override to True unless specified
            if "manual_override" not in update_data:
                update_data["manual_override"] = True

            logger.info(
                f"[InventoryService] Applying patch to '{patch.id}': updates={update_data}, "
                f"previous_label='{item.label}'"
            )

            updated_item = item.model_copy(update=update_data)
            self._inventory[patch.id] = updated_item
            updated_items.append(updated_item)

        logger.info(f"[InventoryService] Successfully patched {len(updated_items)} item(s).")
        return PatchInventoryResponse(
            success=True,
            updated_items=updated_items,
        )
