import logging
import os
import time
from PIL import Image

from classifier.image_model.base import BaseCamera
from classifier.profiler import Profiler

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_IMAGE_PATH = os.environ.get(
    "SAMPLE_IMAGE_PATH",
    os.path.join(os.path.dirname(__file__), "sample_fridge.png"),
)


class DummyCamera(BaseCamera):
    """
    Dummy camera implementation that reads a local image from disk.
    Simulates a physical sensor/camera capture pipeline.
    """

    def __init__(self, image_path: str = DEFAULT_SAMPLE_IMAGE_PATH):
        self.image_path = image_path
        logger.info(f"[DummyCamera] Initialized with target image path: {self.image_path}")

    @Profiler
    def capture_frame(self) -> Image.Image:
        """
        Reads the local image and returns it as a PIL Image in RGB format.
        """
        start_time = time.perf_counter()
        logger.info(f"[DummyCamera] capture_frame() triggered. Reading image from: {self.image_path}")

        if not os.path.exists(self.image_path):
            err_msg = f"[DummyCamera] Image file does not exist at path: {self.image_path}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        try:
            image = Image.open(self.image_path)
            rgb_image = image.convert("RGB")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[DummyCamera] Frame captured successfully. Dimensions: {rgb_image.width}x{rgb_image.height}, "
                f"mode: {rgb_image.mode} (elapsed: {elapsed_ms:.2f}ms)"
            )
            return rgb_image
        except Exception as exc:
            logger.exception(f"[DummyCamera] Failed to open or decode image from {self.image_path}: {exc}")
            raise
