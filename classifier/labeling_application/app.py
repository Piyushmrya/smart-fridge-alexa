"""
Local Bounding Box Labeling Web App for Fridge Food Items.
Allows interactive labeling of bounding boxes on fridge images,
with instant overlay of Grounding DINO model predictions, autocomplete
from curated fridge vocabulary, and saving in ground-truth format
compatible with the classifier pipeline for precision/recall evaluation.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("labeling_app")

# Ensure repository root is in sys.path to load vocabulary if available
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS
except Exception as e:
    logger.warning(f"Could not load FRIDGE_FOOD_ITEMS from classifier: {e}")
    FRIDGE_FOOD_ITEMS = [
        "tomato", "cherry tomato", "onion", "red onion", "garlic", "ginger", "potato",
        "carrot", "cucumber", "bell pepper", "green capsicum", "red capsicum", "yellow capsicum",
        "jalapeno", "zucchini", "spinach", "lettuce", "iceberg lettuce", "broccoli",
        "cabbage", "apple", "banana", "orange", "lemon", "lime", "milk", "cheese", "butter",
        "yogurt", "eggs", "tofu", "bread", "chicken", "beef", "juice", "sauce", "ketchup"
    ]

CUSTOM_VOCAB_FILE = CURRENT_DIR / "custom_vocabulary.json"

def get_full_vocabulary() -> List[str]:
    """Combine base FRIDGE_FOOD_ITEMS with custom additions like 'musk melon' and 'water bottle'."""
    custom_items = ["musk melon", "water bottle"]
    if CUSTOM_VOCAB_FILE.exists():
        try:
            with open(CUSTOM_VOCAB_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    custom_items = loaded
        except Exception as e:
            logger.warning(f"Error reading custom_vocabulary.json: {e}")

    # Combine ensuring case-insensitive uniqueness and preserving order
    combined = []
    seen = set()
    for item in custom_items + FRIDGE_FOOD_ITEMS:
        item_clean = item.strip().lower()
        if item_clean and item_clean not in seen:
            seen.add(item_clean)
            combined.append(item.strip())
    return combined


# Default images folder from user's request
DEFAULT_IMAGES_DIR = "/home/niare/Downloads/WhatsApp Unknown 2026-10-20 at 3.15.51 PM"

app = FastAPI(title="Smart Fridge Bounding Box Labeler", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request bodies
class ScanRequest(BaseModel):
    directory_path: str

class SaveRequest(BaseModel):
    directory_path: str
    annotations: Dict[str, List[Dict[str, Any]]]  # filename -> list of boxes
    image_metadata: Optional[Dict[str, Dict[str, Any]]] = None

class AddVocabularyRequest(BaseModel):
    label: str


@app.get("/api/config")
def get_config():
    """Return default configuration and vocabulary items."""
    vocab = get_full_vocabulary()
    return {
        "default_directory": DEFAULT_IMAGES_DIR if os.path.isdir(DEFAULT_IMAGES_DIR) else str(CURRENT_DIR),
        "vocabulary": vocab,
        "vocabulary_count": len(vocab),
    }


@app.post("/api/vocabulary")
def add_vocabulary(payload: AddVocabularyRequest):
    """Add a new word to the custom vocabulary list."""
    new_label = payload.label.strip()
    if not new_label:
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    custom_items = ["musk melon", "water bottle"]
    if CUSTOM_VOCAB_FILE.exists():
        try:
            with open(CUSTOM_VOCAB_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    custom_items = loaded
        except Exception:
            pass

    if new_label.lower() not in [c.lower() for c in custom_items]:
        custom_items.append(new_label)
        with open(CUSTOM_VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(custom_items, f, indent=2)

    return {"success": True, "vocabulary": get_full_vocabulary()}


def find_prediction_files(directory: Path) -> List[Dict[str, str]]:
    """Look for any model prediction bounding_boxes.json files in directory or subdirectories."""
    found = []
    # Check directly inside directory and subdirectories up to 2 levels
    candidates = list(directory.glob("**/bounding_boxes.json")) + list(directory.glob("**/*predictions*.json"))
    for cand in candidates:
        if cand.is_file():
            rel = str(cand.relative_to(directory))
            found.append({
                "label": rel,
                "path": str(cand),
                "name": cand.name
            })
    return found


@app.post("/api/scan")
def scan_directory(payload: ScanRequest):
    """Scan a folder on the local machine for images, existing annotations, and model predictions."""
    dir_path = Path(payload.directory_path).expanduser().resolve()
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {dir_path}")

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    for item in sorted(dir_path.iterdir()):
        if item.is_file() and item.suffix.lower() in valid_exts:
            # Skip output detection images if they are in the root directory
            if "_detected" in item.stem:
                continue
            try:
                with Image.open(item) as img:
                    width, height = img.size
                image_files.append({
                    "filename": item.name,
                    "path": str(item),
                    "width": width,
                    "height": height,
                    "size_bytes": item.stat().st_size,
                })
            except Exception as e:
                logger.warning(f"Error opening image {item}: {e}")
                image_files.append({
                    "filename": item.name,
                    "path": str(item),
                    "width": 0,
                    "height": 0,
                    "size_bytes": item.stat().st_size,
                })

    # Check for existing annotations
    annotations_file = dir_path / "annotations.json"
    ground_truth_file = dir_path / "ground_truth_bounding_boxes.json"
    existing_annotations: Dict[str, List[Dict[str, Any]]] = {}

    if annotations_file.exists():
        try:
            with open(annotations_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "images" in data and isinstance(data["images"], dict):
                    for img_name, img_data in data["images"].items():
                        existing_annotations[img_name] = img_data.get("boxes", [])
                elif isinstance(data, dict):
                    # Direct mapping format
                    for img_name, boxes in data.items():
                        if isinstance(boxes, list):
                            existing_annotations[img_name] = boxes
                        elif isinstance(boxes, dict) and "detections" in boxes:
                            existing_annotations[img_name] = boxes["detections"]
        except Exception as e:
            logger.error(f"Failed to load existing annotations.json: {e}")
    elif ground_truth_file.exists():
        try:
            with open(ground_truth_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for img_name, img_info in data.items():
                        if isinstance(img_info, dict) and "detections" in img_info:
                            existing_annotations[img_name] = img_info["detections"]
        except Exception as e:
            logger.error(f"Failed to load ground_truth_bounding_boxes.json: {e}")

    # Check for model prediction files (e.g. output/bounding_boxes.json, output_grounding_dino/...)
    prediction_files = find_prediction_files(dir_path)

    return {
        "directory": str(dir_path),
        "images": image_files,
        "total_images": len(image_files),
        "existing_annotations": existing_annotations,
        "prediction_files": prediction_files,
    }


@app.get("/api/image")
def get_image(path: str = Query(..., description="Absolute path to the image file")):
    """Serve an image file directly from local disk."""
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Simple mime detection
    suffix = file_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)


@app.get("/api/predictions")
def get_predictions(path: str = Query(..., description="Path to prediction JSON file")):
    """Load model detection results from a prediction JSON file."""
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Prediction file not found")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Standardize format: return { filename: [ {label, bbox, confidence} ] }
        standardized = {}
        for img_name, img_data in data.items():
            if isinstance(img_data, dict) and "detections" in img_data:
                standardized[img_name] = img_data["detections"]
            elif isinstance(img_data, list):
                standardized[img_name] = img_data
            else:
                standardized[img_name] = []
        return {
            "file": str(file_path),
            "predictions": standardized,
        }
    except Exception as e:
        logger.error(f"Error reading prediction file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save")
def save_annotations(payload: SaveRequest):
    """
    Save annotations to local disk in two formats:
    1. annotations.json: Detailed format with image dimensions, timestamps, and boxes
    2. ground_truth_bounding_boxes.json: 100% compatible with model's bounding_boxes.json
       format ({ "filename": { "detections": [{"label": ..., "bbox": [x1, y1, x2, y2]}] } })
       ready for precision/recall calculation!
    """
    dir_path = Path(payload.directory_path).expanduser().resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    now_iso = datetime.now().isoformat()
    image_metadata = payload.image_metadata or {}

    # Format 1: Detailed annotations.json
    detailed_data = {
        "version": "1.0",
        "updated_at": now_iso,
        "directory": str(dir_path),
        "total_images": len(payload.annotations),
        "images": {}
    }

    # Format 2: ground_truth_bounding_boxes.json (exact match with output/bounding_boxes.json)
    ground_truth_data = {}

    total_boxes_count = 0
    for filename, boxes in payload.annotations.items():
        meta = image_metadata.get(filename, {})
        cleaned_boxes = []
        gt_detections = []
        for box in boxes:
            bbox = box.get("bbox", [])
            label = box.get("label", "").strip()
            if not label or len(bbox) != 4:
                continue
            
            # Ensure coordinates are rounded floats [x1, y1, x2, y2]
            clean_bbox = [round(float(c), 1) for c in bbox]
            cleaned_boxes.append({
                "id": box.get("id"),
                "label": label,
                "bbox": clean_bbox,
                "created_at": box.get("created_at", now_iso),
            })
            gt_detections.append({
                "label": label,
                "confidence": 1.0,  # ground truth has 100% confidence
                "bbox": clean_bbox
            })

        detailed_data["images"][filename] = {
            "width": meta.get("width"),
            "height": meta.get("height"),
            "boxes_count": len(cleaned_boxes),
            "boxes": cleaned_boxes,
        }
        ground_truth_data[filename] = {
            "detections": gt_detections
        }
        total_boxes_count += len(cleaned_boxes)

    # Atomic write for annotations.json
    annot_file = dir_path / "annotations.json"
    annot_tmp = dir_path / "annotations.json.tmp"
    with open(annot_tmp, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2, ensure_ascii=False)
    annot_tmp.replace(annot_file)

    # Atomic write for ground_truth_bounding_boxes.json
    gt_file = dir_path / "ground_truth_bounding_boxes.json"
    gt_tmp = dir_path / "ground_truth_bounding_boxes.json.tmp"
    with open(gt_tmp, "w", encoding="utf-8") as f:
        json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
    gt_tmp.replace(gt_file)

    logger.info(f"Saved {total_boxes_count} bounding boxes across {len(payload.annotations)} images to {annot_file} and {gt_file}")

    return {
        "success": True,
        "saved_path": str(annot_file),
        "ground_truth_path": str(gt_file),
        "total_images": len(payload.annotations),
        "total_boxes": total_boxes_count,
        "updated_at": now_iso,
    }


# Mount static assets
STATIC_DIR = CURRENT_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """Serve the single page labeling application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Labeling application frontend is initializing...</h1>")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Starting Smart Fridge Bounding Box Labeler on http://localhost:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
