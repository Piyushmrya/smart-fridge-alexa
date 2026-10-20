from typing import List, Optional
from pydantic import BaseModel, Field


class InventoryItem(BaseModel):
    id: str = Field(..., description="Unique persistent identifier for the tracked object")
    label: str = Field(..., description="Object classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    time_in_fridge_seconds: int = Field(..., ge=0, description="Duration in seconds the item has been continuously present")
    manual_override: bool = Field(..., description="True if a field was explicitly set/corrected by a user; false if purely inferred")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2] coordinates")


class InventoryResponse(BaseModel):
    last_updated_epoch: int = Field(..., description="Unix timestamp in seconds of the latest state sync")
    items: List[InventoryItem] = Field(..., description="List of currently detected items")


class PatchInventoryItem(BaseModel):
    id: str = Field(..., description="The identifier of the item to modify")
    label: Optional[str] = Field(None, description="New classification name")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Explicit confidence override")
    time_in_fridge_seconds: Optional[int] = Field(None, ge=0, description="Overwritten presence duration")
    manual_override: Optional[bool] = Field(None, description="Sets whether the vision model should ignore automatic relabeling")
    bbox: Optional[List[float]] = Field(None, description="Explicit bounding box override")


class PatchInventoryResponse(BaseModel):
    success: bool = True
    updated_items: List[InventoryItem]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
