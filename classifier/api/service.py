import logging
import time
from typing import Dict, List, Optional
from PIL import Image

from classifier.api.models import (
    InventoryItem,
    InventoryResponse,
    PatchInventoryItem,
    PatchInventoryResponse,
)
from classifier.image_model.base import BaseCamera, BaseDetector, DetectedObject
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
    def __init__(self, item_id: str):
        super().__init__(f"Item id '{item_id}' not found in tracking inventory.")
        self.item_id = item_id


class InventoryService:
    """
    Model-agnostic and hardware-agnostic inventory service.
    Coordinates image capture via BaseCamera, detection via BaseDetector,
    and maintains the persistent tracking state.
    """

    def __init__(self, camera: BaseCamera, detector: BaseDetector):
        self.camera = camera
        self.detector = detector

        self._inventory: Dict[str, InventoryItem] = {}
        self._first_seen_epoch: Dict[str, int] = {}
        self._last_updated_epoch: int = 0
        self._item_counter: int = 0
        logger.info(
            f"[InventoryService] Initialized with camera={camera.__class__.__name__}, "
            f"detector={detector.__class__.__name__}"
        )

    @Profiler
    def _generate_item_id(self) -> str:
        self._item_counter += 1
        return f"item_{self._item_counter}"

    @Profiler
    def sync_inventory(self) -> InventoryResponse:
        """
        Capture frame from camera, run detection, and update tracking inventory.
        Preserves manual overrides and continuous tracking state.
        """
        sync_start = time.perf_counter()
        now = int(time.time())
        logger.info(f"[InventoryService] >>> Starting inventory synchronization cycle at epoch {now}...")

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
            detections: List[DetectedObject] = self.detector.detect(frame)
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
    def get_inventory(self, force_refresh: bool = False) -> InventoryResponse:
        """
        Get current tracked inventory. Triggers camera capture & inference
        if force_refresh is True or if inventory has not been synced yet.
        """
        now = int(time.time())
        if force_refresh:
            logger.info("[InventoryService] GET request with force_refresh=True. Triggering live sync...")
            return self.sync_inventory()

        if self._last_updated_epoch == 0:
            logger.info("[InventoryService] GET request on unsynced inventory (epoch=0). Triggering initial sync...")
            return self.sync_inventory()

        age_seconds = now - self._last_updated_epoch
        logger.info(
            f"[InventoryService] Serving cached inventory ({len(self._inventory)} items). "
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
