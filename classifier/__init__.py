"""Classifier package containing api and image_model subpackages."""
from classifier.profiler import (
    Profiler,
    ENABLE_PROFILING,
    PROFILING_BASE_DIR,
    get_profiling_log_path,
    set_profiling_enabled,
    set_log_file,
    reset_session_log_path,
)

__all__ = [
    "app",
    "Profiler",
    "ENABLE_PROFILING",
    "PROFILING_BASE_DIR",
    "get_profiling_log_path",
    "set_profiling_enabled",
    "set_log_file",
    "reset_session_log_path",
]


def __getattr__(name: str):
    """Lazy import for app to avoid RuntimeWarning when running python -m classifier.api.main."""
    if name == "app":
        from classifier.api.main import app as _app
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
