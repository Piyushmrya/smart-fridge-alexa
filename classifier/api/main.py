from contextlib import asynccontextmanager
import logging
import sys
from typing import List, Optional
from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from classifier.api.dependencies import get_inventory_service
from classifier.api.models import (
    DetectorType,
    ErrorResponse,
    InventoryResponse,
    PatchInventoryItem,
    PatchInventoryResponse,
)
from classifier.api.service import (
    InventoryService,
    InvalidDetectorError,
    ItemNotFoundError,
)

# Configure standardized logging across the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API Lifecycle] >>> Booting Smart Fridge Edge API server...")
    try:
        service_factory = app.dependency_overrides.get(get_inventory_service, get_inventory_service)
        service = service_factory()
        logger.info("[API Lifecycle] Triggering startup image detection and initial inventory sync...")
        sync_result = service.sync_inventory()
        logger.info(
            f"[API Lifecycle] Initial boot sync succeeded. Tracked items count: {len(sync_result.items)}, "
            f"sync epoch: {sync_result.last_updated_epoch}"
        )
    except Exception as exc:
        logger.exception(f"[API Lifecycle] CRITICAL: Failed during startup inventory sync: {exc}")
        raise

    logger.info("[API Lifecycle] <<< Server initialization complete. Ready to receive HTTP traffic.")
    yield
    logger.info("[API Lifecycle] Smart Fridge Edge API server is shutting down.")


app = FastAPI(
    title="Smart Fridge Edge API",
    description="Edge API for tracking fridge inventory and partial manual overrides.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = "; ".join(f"{'.'.join(str(l) for l in e.get('loc', []))}: {e.get('msg')}" for e in errors)
    logger.warning(f"[HTTP Validation] Error on {request.method} {request.url.path}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": f"Validation error: {msg}"},
    )


@app.get(
    "/inventory",
    response_model=InventoryResponse,
    summary="Get current tracked inventory",
    responses={
        200: {"model": InventoryResponse, "description": "Current tracked inventory state"},
        400: {"model": ErrorResponse, "description": "Invalid detector or parameter specified"},
        500: {"model": ErrorResponse, "description": "Internal server error during sync"},
    },
)
def get_inventory(
    request: Request,
    force_refresh: bool = Query(
        default=False,
        description="Instructs pipeline to capture an immediate live frame, run inference, and bypass temporal smoothing.",
    ),
    detector: Optional[DetectorType] = Query(
        default=DetectorType.LLM,
        description="Vision model to use for detection (defaults to 'llm'). Selectable via dropdown in OpenAPI docs.",
    ),
    service: InventoryService = Depends(get_inventory_service),
):
    explicit_detector = (
        detector.value if ("detector" in request.query_params and detector) else None
    )
    logger.info(
        f"[HTTP GET /inventory] Received request (force_refresh={force_refresh}, "
        f"detector={detector}, explicit={explicit_detector is not None})"
    )
    try:
        response = service.get_inventory(force_refresh=force_refresh, detector=explicit_detector)
        logger.info(
            f"[HTTP GET /inventory] Responding 200 OK: {len(response.items)} items, "
            f"last_updated_epoch={response.last_updated_epoch}"
        )
        return response
    except InvalidDetectorError as err:
        logger.warning(f"[HTTP GET /inventory] Responding 400 BAD REQUEST: {err}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(err),
            },
        )
    except Exception as exc:
        logger.exception(f"[HTTP GET /inventory] Unhandled exception occurred: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"Internal error during inventory retrieval: {str(exc)}",
            },
        )


@app.patch(
    "/inventory",
    response_model=PatchInventoryResponse,
    summary="Batch partial update on tracked inventory items",
    responses={
        200: {"model": PatchInventoryResponse, "description": "Successfully updated items"},
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Item not found in tracking inventory"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def patch_inventory(
    patches: List[PatchInventoryItem],
    service: InventoryService = Depends(get_inventory_service),
):
    patch_ids = [p.id for p in patches]
    logger.info(f"[HTTP PATCH /inventory] Received request targeting item IDs: {patch_ids}")
    try:
        response = service.patch_inventory(patches)
        logger.info(
            f"[HTTP PATCH /inventory] Responding 200 OK: {len(response.updated_items)} items successfully patched"
        )
        return response
    except ItemNotFoundError as err:
        logger.warning(f"[HTTP PATCH /inventory] Responding 404 NOT FOUND: {err}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "error": str(err),
            },
        )
    except Exception as exc:
        logger.exception(f"[HTTP PATCH /inventory] Unhandled exception occurred during patch: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"Internal error during inventory patch: {str(exc)}",
            },
        )


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.environ.get("EDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("EDGE_PORT", "9000"))
    uvicorn.run("classifier.api.main:app", host=host, port=port)
