# classifier/ — Image Classifier + Edge API (Block 4)

(`classifier/api/` = the HTTP service, `classifier/image_model/` = camera + vision model.)

Real implementation of the Smart Fridge Edge API and vision model detection pipeline. Replaces `classifier_stub/mock_edge_server.py`.

## Architecture

```
                      +-------------------+
                      |   FastAPI Layer   |
                      |   (classifier/api/main.py)   |
                      +---------+---------+
                                |
                    Depends on InventoryService
                                |
                      +---------v---------+
                      | InventoryService  |
                      |  (classifier/api/service.py) |
                      +----+---------+----+
                           |         |
               Calls capture_frame() Calls detect()
                           |         |
          +----------------v-+     +-v--------------------+
          |    BaseCamera    |     |     BaseDetector     |
          | (Hardware Source)|     |    (Vision Model)    |
          +--------+---------+     +----------+-----------+
                   |                          |
          +--------v---------+     +----------v-----------+
          |   DummyCamera    |     | GroundingDinoDetector|
          | (Local Image)    |     | (Grounding DINO Tiny)|
          +------------------+     +----------------------+
```

## Running the Real Edge API

```bash
# From the project root (first time only: install the extra dependencies, ~2 GB incl. PyTorch):
pip install -r classifier/requirements.txt

# Run the Edge API server:
python -m classifier.api.main
```
Listens on `http://0.0.0.0:9000` (or configured via `EDGE_PORT` and `EDGE_HOST`).

Set in your `.env`:
```bash
EDGE_URL=http://127.0.0.1:9000
```

## Endpoints

- **`GET /inventory?force_refresh=true|false&detector=llm|yolo_world|grounding_dino|owlv2|florence2`**
  - `force_refresh=false` (default): Returns tracked inventory from memory with updated `time_in_fridge_seconds`. If a different `detector` is specified than the current active one, automatically triggers a new detection sync.
  - `force_refresh=true`: Captures a fresh frame, runs vision detection with the requested or current detector, and updates tracking state.
  - `detector` (optional, default: `llm` / `LLMdetector`): Selects the vision detection model. Supported values include `llm` (alias `LLMdetector`, `claude`), `yolo_world`, `grounding_dino`, `owlv2`, and `florence2`. Detectors are cached in memory so models are loaded lazily once and reused. Invalid detectors return HTTP 400 Bad Request.
- **`PATCH /inventory`**
  - Batch updates tracked items (`label`, `confidence`, `time_in_fridge_seconds`, `manual_override`).
  - Sets `manual_override=true`, preventing vision inference from overwriting user-corrected labels.

## Vocabulary & Models

- Grounding DINO Zero-Shot Object Detector (`IDEA-Research/grounding-dino-tiny`).
- Curated vocabulary of 200+ food ingredients in `classifier/image_model/vocabulary.py`.
- Non-Maximum Suppression (NMS) and automatic prompt chunking to operate within BERT's 512-token limit.
- Hardware-agnostic camera interface in `classifier/image_model/base.py` (`DummyCamera` reading `classifier/image_model/sample_fridge.png`).

## Notes
- The first start downloads the Grounding DINO weights (~700 MB) into `classifier/image_model/models/` (git-ignored) and runs one detection.
- Labels the detector returns (e.g. `red onion`, `milk bottle`) are folded onto plain item names by `mcp_server/store.py: norm()`; see `tests/test_labels.py`.
- Tests: `python -m pytest tests/test_api.py tests/test_image_model.py` (skipped automatically if the extras are not installed).
