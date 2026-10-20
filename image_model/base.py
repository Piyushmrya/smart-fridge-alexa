from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from PIL import Image


class DetectedObject(BaseModel):
    """Normalized object detection result, independent of the underlying vision model."""
    label: str = Field(..., description="Detected object label or classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box coordinates in [x1, y1, x2, y2] format"
    )


class BaseDetector(ABC):
    """
    Abstract interface for object detection models.
    Any model implementation (Grounding DINO, YOLO, DETR, Cloud API, etc.)
    must conform to this interface.
    """

    @abstractmethod
    def detect(self, image: Image.Image) -> List[DetectedObject]:
        """
        Run object detection on the provided PIL Image.

        Args:
            image: PIL Image to run detection on.

        Returns:
            List of DetectedObject instances with labels, confidence scores, and bounding boxes.
        """
        pass


class BaseCamera(ABC):
    """
    Abstract interface for image capture sources (sensors, webcams, dummy files, video streams).
    """

    @abstractmethod
    def capture_frame(self) -> Image.Image:
        """
        Capture and return a single frame as a PIL Image in RGB format.

        Returns:
            PIL.Image.Image in RGB mode.
        """
        pass
