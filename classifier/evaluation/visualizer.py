import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from classifier.evaluation.metrics import ImageEvaluationResult

logger = logging.getLogger(__name__)

# Colors for bounding box debug overlays
COLOR_TP = "#2ECC71"  # Vibrant Green for True Positives (Matches)
COLOR_FP = "#E74C3C"  # Vibrant Red for False Positives (Extra / wrong)
COLOR_FN = "#3498DB"  # Vibrant Blue for False Negatives (Missed GT)


def get_default_font(size: int = 14):
    """Attempt to load a crisp TrueType font, falling back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_evaluation_overlay(
    image_path: str,
    eval_result: ImageEvaluationResult,
    model_name: str,
    output_path: str,
) -> str:
    """
    Renders an annotated debug image highlighting:
    - Green boxes: True Positives (IoU >= 0.50 with correct label)
    - Red boxes: False Positives (Extracted by model but not in GT)
    - Blue boxes: False Negatives (In GT but missed by model)

    Args:
        image_path: Path to raw input image.
        eval_result: ImageEvaluationResult from matching.
        model_name: Name of model for label header.
        output_path: Destination path for saving the annotated image.

    Returns:
        Absolute path to the saved annotated image.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as base_img:
        annotated = base_img.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font = get_default_font(14)
        title_font = get_default_font(16)

        # 1. Draw False Negatives first (Blue dashed / solid)
        for fn in eval_result.false_negatives:
            bbox = fn.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                draw.rectangle([x1, y1, x2, y2], outline=COLOR_FN, width=2)
                text = f"MISS: {fn.get('label', '')}"
                draw_badge(draw, x1, y1, text, COLOR_FN, font)

        # 2. Draw False Positives (Red)
        for fp in eval_result.false_positives:
            bbox = fp.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                draw.rectangle([x1, y1, x2, y2], outline=COLOR_FP, width=3)
                conf = fp.get("confidence", 0.0)
                text = f"FP: {fp.get('label', '')} ({conf:.0%})"
                draw_badge(draw, x1, y1, text, COLOR_FP, font)

        # 3. Draw True Positives (Green)
        for match in eval_result.matches:
            bbox = match.pred_bbox
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                draw.rectangle([x1, y1, x2, y2], outline=COLOR_TP, width=3)
                text = f"MATCH: {match.pred_label} (IoU:{match.iou:.0%}, {match.confidence:.0%})"
                draw_badge(draw, x1, y1, text, COLOR_TP, font)

        # 4. Draw Header Banner
        header_text = (
            f"[{model_name}]  Precision: {eval_result.precision:.1%} | Recall: {eval_result.recall:.1%} | "
            f"Matches: {eval_result.matches_count} | Extracted: {eval_result.extracted} | GT: {eval_result.total_in_gt}"
        )
        draw.rectangle([0, 0, annotated.width, 30], fill="#1E272C")
        draw.text((10, 6), header_text, fill="white", font=title_font)

        annotated.save(output_path, quality=90)

    logger.debug(f"[Visualizer] Saved debug visualization to: {output_path}")
    return output_path


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, bg_color: str, font: ImageFont.ImageFont):
    """Draw a compact text badge with background pill."""
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        # Shift down if close to top edge
        badge_y1 = max(32, y - 18)
        badge_y2 = badge_y1 + (bbox[3] - bbox[1]) + 4
        badge_x2 = x + (bbox[2] - bbox[0]) + 6
        draw.rectangle([x, badge_y1, badge_x2, badge_y2], fill=bg_color)
        draw.text((x + 3, badge_y1 + 1), text, fill="white", font=font)
    except Exception:
        draw.text((x, y), text, fill=bg_color)


def generate_benchmark_visualizations(
    model_name: str,
    dataset: Dict[str, Tuple[str, List[Dict[str, Any]]]],
    eval_results: List[ImageEvaluationResult],
    output_dir: str,
) -> List[str]:
    """Generates visualization images for all images in the evaluated dataset."""
    saved_paths = []
    model_viz_dir = Path(output_dir) / "visualizations" / model_name.replace(" ", "_")
    model_viz_dir.mkdir(parents=True, exist_ok=True)

    result_map = {r.image_name: r for r in eval_results}

    for img_name, (img_path, _) in dataset.items():
        if img_name in result_map:
            out_file = str(model_viz_dir / f"eval_{img_name}")
            render_evaluation_overlay(
                image_path=img_path,
                eval_result=result_map[img_name],
                model_name=model_name,
                output_path=out_file,
            )
            saved_paths.append(out_file)

    logger.info(f"[Visualizer] Generated {len(saved_paths)} evaluation overlays in {model_viz_dir}")
    return saved_paths
