"""
Fridge Item Detection using Grounding DINO (Zero-Shot)

Takes a fridge photo and detects/labels items using text prompts.
No training required — just specify what items to look for.
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Configuration ──────────────────────────────────────────────
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS
except ModuleNotFoundError:
    try:
        from vocabulary import FRIDGE_FOOD_ITEMS
    except ModuleNotFoundError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from classifier.image_model.vocabulary import FRIDGE_FOOD_ITEMS

# Items to detect in the fridge — pure food ingredients!
FRIDGE_ITEMS = " . ".join(FRIDGE_FOOD_ITEMS[:40])

# Detection threshold — lower = more detections (but more false positives)
CONFIDENCE_THRESHOLD = 0.25

# ── Load Model ─────────────────────────────────────────────────
def load_model():
    """Load Grounding DINO model and processor."""
    print(f"Loading model: {MODEL_ID}")
    print(f"Cache directory: {MODEL_CACHE_DIR}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_DIR
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, cache_dir=MODEL_CACHE_DIR
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on: {device}")

    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM used: {vram_used:.2f} GB")

    return processor, model, device


# ── Run Detection ──────────────────────────────────────────────
def detect_items(image_path, processor, model, device, text_prompt=FRIDGE_ITEMS,
                 threshold=CONFIDENCE_THRESHOLD):
    """Detect items in an image using Grounding DINO."""

    image = Image.open(image_path).convert("RGB")
    print(f"\nImage: {image_path} ({image.size[0]}x{image.size[1]})")
    print(f"Looking for: {text_prompt}")
    print(f"Confidence threshold: {threshold}")

    # Prepare inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]

    return image, results


# ── Display Results ────────────────────────────────────────────
def print_results(results):
    """Print detected items as a list."""
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    if len(labels) == 0:
        print("\n⚠️  No items detected. Try lowering the confidence threshold.")
        return []

    print(f"\n{'='*50}")
    print(f"  🧊 FRIDGE INVENTORY — {len(labels)} item(s) detected")
    print(f"{'='*50}")

    detected_items = []
    for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
        confidence = score.item()
        bbox = [round(c.item(), 1) for c in box]
        print(f"  {i+1}. {label:<20s}  confidence: {confidence:.0%}  bbox: {bbox}")
        detected_items.append({"label": label, "confidence": confidence, "bbox": bbox})

    print(f"{'='*50}\n")

    # Summary: unique items
    unique_items = {}
    for item in detected_items:
        label = item["label"]
        if label not in unique_items or item["confidence"] > unique_items[label]:
            unique_items[label] = item["confidence"]

    print("📋 Unique items found:")
    for label, conf in sorted(unique_items.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {label} ({conf:.0%})")

    return detected_items


# ── Draw Bounding Boxes ────────────────────────────────────────
def draw_detections(image, results, output_path="output_detected.jpg"):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)

    # Try to use a nicer font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F0B27A", "#82E0AA", "#F1948A", "#85929E", "#D5DBDB",
    ]

    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(c.item()) for c in box]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background
        text = f"{label} {score.item():.0%}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color
        )
        draw.text((x1, y1), text, fill="black", font=font)

    image.save(output_path)
    print(f"📸 Annotated image saved to: {output_path}")
    return output_path


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect fridge items using Grounding DINO")
    parser.add_argument("image", nargs="?", default="sample_fridge.jpg",
                        help="Path to the fridge image")
    parser.add_argument("--prompt", default=FRIDGE_ITEMS,
                        help="Items to detect (separated by ' . ')")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--output", default=None,
                        help="Output path for annotated image")
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_detected{ext}"

    # Load model
    processor, model, device = load_model()

    # Detect items
    image, results = detect_items(args.image, processor, model, device,
                                  text_prompt=args.prompt, threshold=args.threshold)

    # Print results
    print_results(results)

    # Save annotated image
    draw_detections(image, results, output_path=args.output)
