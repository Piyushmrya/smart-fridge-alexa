import json
import os
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration for ground truth dataset and images."""
    images_dir: str = Field(..., description="Directory containing image files")
    annotations_path: str = Field(..., description="Path to ground truth JSON annotations file")
    image_extensions: List[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"],
        description="Allowed image file extensions",
    )


class MatchingConfig(BaseModel):
    """Configuration for bounding box matching criteria."""
    iou_threshold: float = Field(0.50, ge=0.0, le=1.0, description="Minimum IoU overlap to consider a match")
    case_sensitive: bool = Field(False, description="Whether label comparison is case sensitive")
    normalize_labels: bool = Field(True, description="Whether to apply clean_label deduplication to names")


class ModelEvalConfig(BaseModel):
    """Configuration for a specific model under evaluation."""
    name: str = Field(..., description="Human-readable identifier for the model in reports")
    detector_type: Literal[
        "yolo_world", "grounding_dino", "owlv2", "florence2", "claude", "claude_vision", "llm"
    ] = Field(
        ..., description="Detector class to instantiate"
    )
    model_id: Optional[str] = Field(None, description="Model checkpoint path or repo ID")
    confidence_threshold: float = Field(0.20, ge=0.0, le=1.0, description="Detection confidence threshold")
    device: str = Field("auto", description="'cuda', 'cpu', or 'auto'")
    options: Dict[str, Any] = Field(default_factory=dict, description="Model-specific extra options")


class EvalConfig(BaseModel):
    """Top-level evaluation framework configuration."""
    dataset: DatasetConfig
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    models: List[ModelEvalConfig] = Field(..., min_length=1, description="List of models to benchmark")
    output_dir: str = Field("classifier/evaluation/results", description="Directory to save evaluation reports")
    save_visualizations: bool = Field(True, description="Whether to generate visual debug images")

    @classmethod
    def from_json_file(cls, path: str) -> "EvalConfig":
        """Load and validate configuration from a JSON file."""
        expanded_path = os.path.expanduser(path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Evaluation config file not found: {expanded_path}")
        with open(expanded_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
