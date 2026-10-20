# api/ & image_model/ — Image Classifier + Edge API (Block 4)

Real implementation of the Smart Fridge Edge API and vision model detection pipeline. Replaces `classifier_stub/mock_edge_server.py`.

## Architecture

```
                      +-------------------+
                      |   FastAPI Layer   |
                      |   (api/main.py)   |
                      +---------+---------+
                                |
                    Depends on InventoryService
                                |
                      +---------v---------+
                      | InventoryService  |
                      |  (api/service.py) |
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
# From the project root:
python -m api.main
```
Listens on `http://0.0.0.0:9000` (or configured via `EDGE_PORT` and `EDGE_HOST`).

Set in your `.env`:
```bash
EDGE_URL=http://127.0.0.1:9000
```

## Endpoints

- **`GET /inventory?force_refresh=true|false`**
  - `force_refresh=false` (default): Returns tracked inventory from memory with updated `time_in_fridge_seconds`.
  - `force_refresh=true`: Captures a fresh frame, runs vision detection, and updates tracking state.
- **`PATCH /inventory`**
  - Batch updates tracked items (`label`, `confidence`, `time_in_fridge_seconds`, `manual_override`).
  - Sets `manual_override=true`, preventing vision inference from overwriting user-corrected labels.

## Vocabulary & Models

- Grounding DINO Zero-Shot Object Detector (`IDEA-Research/grounding-dino-tiny`).
- Curated vocabulary of 200+ food ingredients in `image_model/vocabulary.py`.
- Non-Maximum Suppression (NMS) and automatic prompt chunking to operate within BERT's 512-token limit.
- Hardware-agnostic camera interface in `image_model/base.py` (`DummyCamera` reading `image_model/sample_fridge.png`).
