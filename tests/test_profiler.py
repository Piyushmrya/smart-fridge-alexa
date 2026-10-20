import os
import shutil
import time
import pytest

from classifier.profiler import (
    Profiler,
    ENABLE_PROFILING,
    set_profiling_enabled,
    set_log_file,
    get_profiling_log_path,
    reset_session_log_path,
    ProfileNode,
)
from classifier.api.service import InventoryService
from classifier.image_model.base import BaseCamera, BaseDetector, DetectedObject
from PIL import Image


@pytest.fixture(autouse=True)
def cleanup_profiler():
    """Ensure profiler is re-enabled and reset after each test."""
    set_profiling_enabled(True)
    set_log_file(None)
    yield
    set_profiling_enabled(True)
    set_log_file(None)


def test_profiler_enabled_disabled_toggle(tmp_path):
    log_file = tmp_path / "test_toggle.txt"
    set_log_file(str(log_file))

    @Profiler
    def sample_fn(x):
        return x * 2

    # Profiling ON
    set_profiling_enabled(True)
    res = sample_fn(5)
    assert res == 10
    assert log_file.exists()
    content = log_file.read_text()
    assert "sample_fn" in content
    assert "Hits" in content

    # Clear log
    log_file.unlink()

    # Profiling OFF
    set_profiling_enabled(False)
    res = sample_fn(10)
    assert res == 20
    assert not log_file.exists()


def test_profiler_hierarchy_and_hit_counts(tmp_path):
    log_file = tmp_path / "test_hierarchy.txt"
    set_log_file(str(log_file))

    @Profiler
    def leaf_task():
        time.sleep(0.005)
        return "leaf"

    @Profiler
    def mid_task():
        for _ in range(3):
            leaf_task()
        return "mid"

    @Profiler
    def root_task():
        mid_task()
        return "root"

    root_task()

    assert log_file.exists()
    content = log_file.read_text()

    # Verify hierarchy display
    assert "root_task" in content
    assert "mid_task" in content
    assert "leaf_task" in content

    # Verify branches
    assert "├──" in content or "└──" in content

    # Verify hit counts: leaf_task should have hit count 3
    lines = content.splitlines()
    leaf_line = next(line for line in lines if "leaf_task" in line)
    # The columns: Function | Hits | Total (ms) | Avg (ms) | Self (ms) | % Total | % Parent
    parts = [p.strip() for p in leaf_line.split("|")]
    assert len(parts) >= 7
    hits = int(parts[1])
    assert hits == 3

    # leaf_task percentage of parent (mid_task) should be significant
    pct_parent = float(parts[6].replace("%", ""))
    assert pct_parent > 0.0


def test_profiler_with_class_methods(tmp_path):
    log_file = tmp_path / "test_class_methods.txt"
    set_log_file(str(log_file))

    class Worker:
        @Profiler
        def compute(self, a, b):
            return self._inner_add(a, b)

        @Profiler
        def _inner_add(self, a, b):
            return a + b

    worker = Worker()
    result = worker.compute(3, 4)
    assert result == 7

    assert log_file.exists()
    content = log_file.read_text()
    assert "Worker.compute" in content
    assert "Worker._inner_add" in content


def test_profiler_auto_profile_class(tmp_path):
    log_file = tmp_path / "test_profile_class.txt"
    set_log_file(str(log_file))

    @Profiler.profile_class
    class Calculator:
        def step_a(self):
            return self.step_b()

        def step_b(self):
            return 42

    calc = Calculator()
    assert calc.step_a() == 42

    content = log_file.read_text()
    assert "Calculator.step_a" in content
    assert "Calculator.step_b" in content


def test_profiler_exception_handling(tmp_path):
    log_file = tmp_path / "test_exception.txt"
    set_log_file(str(log_file))

    @Profiler
    def risky_child():
        raise ValueError("Intentional child failure")

    @Profiler
    def safe_parent():
        risky_child()

    with pytest.raises(ValueError, match="Intentional child failure"):
        safe_parent()

    # Log should still be recorded even though an exception occurred
    assert log_file.exists()
    content = log_file.read_text()
    assert "safe_parent" in content
    assert "risky_child" in content


def test_profiler_custom_timestamp_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILER_BASE_DIR", str(tmp_path / "custom_profiling_logs"))
    set_log_file(None)
    reset_session_log_path()

    @Profiler
    def standalone_work():
        return 123

    standalone_work()

    resolved_path = get_profiling_log_path()
    assert os.path.exists(resolved_path)
    assert "classifier_profiling.txt" in resolved_path
    assert "custom_profiling_logs" in resolved_path


def test_inventory_service_pipeline_profiling(tmp_path):
    log_file = tmp_path / "test_service_pipeline.txt"
    set_log_file(str(log_file))

    class MockTestCamera(BaseCamera):
        @Profiler
        def capture_frame(self):
            return Image.new("RGB", (100, 100))

    class MockTestDetector(BaseDetector):
        @Profiler
        def detect(self, image):
            return [
                DetectedObject(label="apple", confidence=0.9, bbox=[10.0, 10.0, 50.0, 50.0]),
                DetectedObject(label="milk", confidence=0.8, bbox=[60.0, 60.0, 90.0, 90.0]),
            ]

    service = InventoryService(camera=MockTestCamera(), detector=MockTestDetector())
    response = service.sync_inventory()
    assert len(response.items) == 2

    assert log_file.exists()
    content = log_file.read_text()

    # Verify pipeline components appear in hierarchy
    assert "InventoryService.sync_inventory" in content
    assert "MockTestCamera.capture_frame" in content
    assert "MockTestDetector.detect" in content
    assert "InventoryService._generate_item_id" in content
    assert "CLASSIFIER PIPELINE PROFILING REPORT" in content


def test_profiler_init_methods(tmp_path):
    log_file = tmp_path / "test_init_methods.txt"
    set_log_file(str(log_file))

    # Test class with decorated __init__
    class Component:
        @Profiler
        def __init__(self, name):
            self.name = name

    c = Component("sensor")
    assert c.name == "sensor"

    content = log_file.read_text()
    assert "Component.__init__" in content

    # Test DummyCamera __init__ profiling
    from classifier.image_model.camera import DummyCamera
    cam = DummyCamera()
    assert cam.image_path is not None

    content = log_file.read_text()
    assert "DummyCamera.__init__" in content

