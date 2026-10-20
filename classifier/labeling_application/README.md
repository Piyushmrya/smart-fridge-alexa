# 🧊 Smart Fridge Bounding Box Labeler

A lightweight, high-performance web application designed for interactive bounding box annotation of fridge images.

Built specifically to produce ground-truth bounding box annotations that directly align with the Grounding DINO detection output from `classifier/image_model/detect_fridge_items.py` for evaluating **Precision**, **Recall**, and **mAP / IoU overlap**.

---

## ⚡ Quick Start

Run the application using the launcher script:

```bash
./classifier/labeling_application/run.sh
```

Or run directly with Python / Uvicorn:

```bash
cd classifier/labeling_application
/home/niare/Projects/amazon_project/venv/bin/python -m uvicorn app:app --port 8000 --reload
```

Then open your browser at: **[http://localhost:8000](http://localhost:8000)**.

---

## 🚀 Key Features

1. **Local Image Loading**:
   - Pre-configured to load from `/home/niare/Downloads/WhatsApp Unknown 2026-10-20 at 3.15.51 PM`.
   - Supports any custom directory path entered via the top bar.
   - Shows image list with thumbnail preview, dimensions, and annotation status badges.

2. **Interactive Bounding Box Annotation**:
   - **Draw**: Click & drag with crosshair to create bounding boxes.
   - **Move**: Click & drag any box to reposition it.
   - **Resize**: 8 responsive control handles (corners and edges).
   - **Fine-tune**: Direct numeric pixel input for `[x1, y1, x2, y2, width, height]` and arrow-key nudging.

3. **Curated Fridge Food Vocabulary Autocomplete**:
   - Auto-suggests from all 200+ curated items in [`classifier/image_model/vocabulary.py`](../image_model/vocabulary.py) (`tomato`, `iceberg lettuce`, `cucumber`, `bell pepper`, `green capsicum`, etc.).
   - Instant 1-click tag chips for frequent fridge ingredients.
   - Distinct color palettes automatically generated per label category.

4. **Model Detections Overlay & Assisted Labeling**:
   - Automatically detects model outputs like `output/bounding_boxes.json`, `output_grounding_dino/bounding_boxes.json`, or `output_threshold_0.1/bounding_boxes.json`.
   - **Toggle Model Overlay**: Displays model detections as dashed cyan outlines with confidence scores for instant visual verification.
   - **Import as Labels**: 1-click button to copy model detections into editable ground truth boxes, allowing you to quickly delete false positives or adjust boundaries instead of labeling from scratch.

5. **Smooth Pan & Zoom**:
   - **Zoom**: Mouse wheel (centered on cursor) or `+` / `-` buttons.
   - **Pan**: Hold `Space` + drag, or click & drag with Middle Mouse Button.
   - **Fit**: Reset zoom to fit screen with `0`.

6. **Ground-Truth Output Format**:
   Saving (`Ctrl+S` or "Save Ground Truth") writes two files to the image folder:
   - `annotations.json`: Complete format with image dimensions, counts, timestamps, and boxes.
   - `ground_truth_bounding_boxes.json`: Exactly matches the schema of the classifier's `bounding_boxes.json` (`{ "filename": { "detections": [{"label": ..., "bbox": [x1, y1, x2, y2]}] } }`), ready for plug-and-play IoU calculation!

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| <kbd>B</kbd> | Activate Draw Bounding Box tool |
| <kbd>V</kbd> or <kbd>Esc</kbd> | Activate Select / Edit tool |
| <kbd>Del</kbd> or <kbd>Backspace</kbd> | Delete selected bounding box |
| <kbd>Ctrl</kbd> + <kbd>S</kbd> | Save ground-truth annotations |
| <kbd>Ctrl</kbd> + <kbd>Z</kbd> | Undo last action |
| <kbd>Ctrl</kbd> + <kbd>Y</kbd> | Redo action |
| <kbd>[</kbd> / <kbd>]</kbd> or <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> | Previous / Next Image |
| <kbd>Space</kbd> + Drag / Middle Click | Pan canvas |
| <kbd>Mouse Wheel</kbd> | Zoom in / out |
| <kbd>0</kbd> | Fit image to screen |
| <kbd>Arrow Keys</kbd> | Nudge selected box by 1px (Hold <kbd>Shift</kbd> for 10px) |
