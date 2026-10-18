# classifier_stub/ — stand-ins for the classifier teammate's block (block 4)

The real image classifier + Edge API is built by the teammate. These files let us develop without it.

| File | What it does |
|---|---|
| `mock_edge_server.py` | **Runnable copy of the agreed API** on port 9000: `GET /inventory`, `PATCH /inventory`, plus `PUT /_mock/items` to change the fridge for demos. Use it by setting `EDGE_URL=http://127.0.0.1:9000`. Doubles as the spec the real API must match. |
| `push_example.py` | Old "push mode" client: POSTs snapshots to the MCP server's `/api/inventory`. |
| `../mock_fridge.py` | Loop that pushes random snapshots via `push_example` (push mode only). |

Run: `python -m classifier_stub.mock_edge_server`
