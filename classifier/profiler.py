"""
Hierarchical profiling decorator class for the Classifier pipeline.

Provides function, method, and class profiling with:
- Global on/off toggle (ENABLE_PROFILING).
- Automatic logging to timestamped directories: profiling_logs/<date_time>/classifier_profiling.txt
- Hierarchical call tree formatting with visual indentation.
- Timing metrics: Total (ms), Avg (ms), Self (ms), Hits, % Total, and % Parent.
- Thread-local context tracking and thread-safe file writing.
"""

from collections import OrderedDict
from datetime import datetime
import functools
import inspect
import os
import threading
import time
import types
from typing import Any, Callable, Dict, List, Optional, Union

# Global profiling toggle: can be toggled at runtime or via environment variable
_ENV_ENABLED = os.environ.get("ENABLE_CLASSIFIER_PROFILING", "true").strip().lower()
ENABLE_PROFILING: bool = _ENV_ENABLED in ("1", "true", "yes", "on")

# Base directory for profiling outputs
PROFILING_BASE_DIR: str = os.environ.get("PROFILER_BASE_DIR", "profiling_logs")

# Session timestamp and custom log file override
_session_datetime: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_custom_log_file: Optional[str] = os.environ.get("PROFILER_LOG_FILE")
_file_lock = threading.Lock()


def set_profiling_enabled(enabled: bool) -> None:
    """Globally enable or disable profiling."""
    global ENABLE_PROFILING
    ENABLE_PROFILING = bool(enabled)


def is_profiling_enabled() -> bool:
    """Check if profiling is currently enabled."""
    return ENABLE_PROFILING


def set_log_file(file_path: Optional[str]) -> None:
    """Set an explicit custom log file path, or None to revert to default."""
    global _custom_log_file
    _custom_log_file = file_path


def reset_session_log_path() -> None:
    """Reset the session timestamp folder (e.g. for a new test or execution cycle)."""
    global _session_datetime
    _session_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_profiling_log_path() -> str:
    """
    Resolve the target log file path.
    Defaults to: profiling_logs/<date_time>/classifier_profiling.txt
    """
    if _custom_log_file:
        return _custom_log_file

    base_dir = os.environ.get("PROFILER_BASE_DIR", PROFILING_BASE_DIR)
    session_dir = os.path.join(base_dir, _session_datetime)
    return os.path.join(session_dir, "classifier_profiling.txt")


class ProfileNode:
    """Represents a node in the hierarchical execution call tree."""

    def __init__(self, name: str, parent: Optional["ProfileNode"] = None):
        self.name: str = name
        self.parent: Optional["ProfileNode"] = parent
        self.children: Dict[str, "ProfileNode"] = OrderedDict()
        self.hit_count: int = 0
        self.total_time_seconds: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return self.total_time_seconds * 1000.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.hit_count if self.hit_count > 0 else 0.0

    @property
    def self_time_ms(self) -> float:
        children_ms = sum(child.total_time_ms for child in self.children.values())
        return max(0.0, self.total_time_ms - children_ms)

    def get_child(self, name: str, active_ancestors: List["ProfileNode"]) -> "ProfileNode":
        """
        Lookup or create child node.
        If a child with this name is already in active_ancestors (recursive call),
        disambiguate the key to prevent cyclic graph loops.
        """
        key = name
        if any(ancestor.name == name for ancestor in active_ancestors):
            key = f"{name} (depth {len(active_ancestors)})"

        if key not in self.children:
            self.children[key] = ProfileNode(name=name, parent=self)
        return self.children[key]


class _ThreadLocalState(threading.local):
    """Thread-local call stack tracker."""

    def __init__(self):
        super().__init__()
        self.stack: List[ProfileNode] = []
        self.root: Optional[ProfileNode] = None


_local_state = _ThreadLocalState()


def _clean_name(qualname: str) -> str:
    """Simplify qualification name by stripping '<locals>.' if present."""
    if "<locals>." in qualname:
        return qualname.split("<locals>.")[-1]
    return qualname


def _format_hierarchy_table(root: ProfileNode) -> str:
    """
    Formats the hierarchical call tree into an ASCII table with tree branch indentation,
    hit counts, timings (ms), self time, and percentage breakdowns.
    """
    lines: List[str] = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    total_duration_ms = root.total_time_ms

    def _calc_max_name_len(node: ProfileNode, depth: int = 0) -> int:
        cur = depth * 4 + len(node.name)
        child_lens = [_calc_max_name_len(child, depth + 1) for child in node.children.values()]
        return max([cur] + child_lens)

    col_width = max(55, min(120, _calc_max_name_len(root) + 4))
    header_border = "=" * (col_width + 65)
    section_border = "-" * (col_width + 65)

    lines.append(header_border)
    lines.append(f"CLASSIFIER PIPELINE PROFILING REPORT")
    lines.append(f"Timestamp: {timestamp_str} | Root: {root.name} | Total Duration: {total_duration_ms:.2f} ms")
    lines.append(section_border)
    lines.append(
        f"{'Function / Call Hierarchy':<{col_width}} | {'Hits':>5} | {'Total (ms)':>10} | {'Avg (ms)':>9} | "
        f"{'Self (ms)':>9} | {'% Total':>7} | {'% Parent':>8}"
    )
    lines.append(section_border)

    def _render_node(node: ProfileNode, prefix: str = "", is_last: bool = True, is_root: bool = False):
        if is_root:
            display_name = node.name
            child_prefix = ""
            pct_parent = 100.0
        else:
            branch = "└── " if is_last else "├── "
            display_name = f"{prefix}{branch}{node.name}"
            child_prefix = prefix + ("    " if is_last else "│   ")
            parent_ms = node.parent.total_time_ms if node.parent and node.parent.total_time_ms > 0 else total_duration_ms
            pct_parent = (node.total_time_ms / parent_ms * 100.0) if parent_ms > 0 else 0.0

        pct_total = (node.total_time_ms / total_duration_ms * 100.0) if total_duration_ms > 0 else 0.0

        if len(display_name) > col_width:
            truncated_name = display_name[: col_width - 3] + "..."
        else:
            truncated_name = display_name

        lines.append(
            f"{truncated_name:<{col_width}} | {node.hit_count:>5d} | {node.total_time_ms:>10.2f} | "
            f"{node.avg_time_ms:>9.2f} | {node.self_time_ms:>9.2f} | {pct_total:>6.1f}% | {pct_parent:>7.1f}%"
        )

        children = list(node.children.values())
        for idx, child in enumerate(children):
            _render_node(child, prefix=child_prefix, is_last=(idx == len(children) - 1), is_root=False)

    _render_node(root, is_root=True)
    lines.append(header_border)
    lines.append("\n")
    return "\n".join(lines)


def _append_report_to_log(report_text: str) -> None:
    """Thread-safe append of profiling report to the destination text file."""
    log_path = get_profiling_log_path()
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with _file_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(report_text)


class Profiler:
    """
    Class-based profiling decorator.
    
    Can be used as:
        @Profiler
        def my_function(): ...

        @Profiler(name="custom_name")
        def my_function(): ...

        @Profiler
        class MyPipelineClass: ...

        @Profiler.profile_class
        class MyPipelineClass: ...
    """

    def __init__(
        self,
        target: Optional[Union[Callable, type]] = None,
        *,
        name: Optional[str] = None,
    ):
        self._name: Optional[str] = name
        self._target: Optional[Union[Callable, type]] = None
        self._is_class: bool = False

        if target is not None:
            if inspect.isclass(target):
                self._is_class = True
                self._target = self._decorate_class(target)
            elif callable(target):
                self._target = target
                if not self._name:
                    raw_name = getattr(target, "__qualname__", getattr(target, "__name__", "callable"))
                    self._name = _clean_name(raw_name)
                functools.update_wrapper(self, target)

    @classmethod
    def profile_class(cls, target_cls: type) -> type:
        """Helper to decorate all callable methods of a class."""
        return cls._decorate_class(target_cls)

    @staticmethod
    def _decorate_class(target_cls: type) -> type:
        """Iterates over class attributes and decorates all methods with Profiler."""
        cls_name = _clean_name(target_cls.__name__)
        for attr_name, attr_value in list(target_cls.__dict__.items()):
            # Avoid magic internal methods except __init__, skip other standard dunders
            if attr_name.startswith("__") and attr_name.endswith("__") and attr_name != "__init__":
                continue

            if inspect.isfunction(attr_value) or isinstance(attr_value, (classmethod, staticmethod)):
                if isinstance(attr_value, staticmethod):
                    orig_fn = attr_value.__func__
                    setattr(target_cls, attr_name, staticmethod(Profiler(orig_fn, name=f"{cls_name}.{attr_name}")))
                elif isinstance(attr_value, classmethod):
                    orig_fn = attr_value.__func__
                    setattr(target_cls, attr_name, classmethod(Profiler(orig_fn, name=f"{cls_name}.{attr_name}")))
                else:
                    setattr(target_cls, attr_name, Profiler(attr_value, name=f"{cls_name}.{attr_name}"))
        return target_cls

    def __get__(self, instance: Any, owner: Any) -> Any:
        """Descriptor support for binding to instance methods."""
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # If Profiler was instantiated without target e.g. @Profiler(name="foo")
        if self._target is None:
            if len(args) == 1 and (callable(args[0]) or inspect.isclass(args[0])) and not kwargs:
                target = args[0]
                if inspect.isclass(target):
                    return self._decorate_class(target)
                return Profiler(target, name=self._name)
            raise TypeError("Profiler decorator called without a valid callable or class target.")

        if self._is_class:
            return self._target(*args, **kwargs)

        # If profiling is disabled globally, execute target with zero overhead
        if not ENABLE_PROFILING:
            return self._target(*args, **kwargs)

        # Async coroutine support
        if inspect.iscoroutinefunction(self._target):
            return self._call_async(*args, **kwargs)

        return self._call_sync(*args, **kwargs)

    def _call_sync(self, *args: Any, **kwargs: Any) -> Any:
        func_name = self._name or _clean_name(getattr(self._target, "__qualname__", self._target.__name__))
        stack = _local_state.stack

        if not stack:
            # Root call in this thread
            node = ProfileNode(name=func_name)
            _local_state.root = node
            stack.append(node)
            is_root = True
        else:
            parent = stack[-1]
            node = parent.get_child(name=func_name, active_ancestors=stack)
            stack.append(node)
            is_root = False

        start_time = time.perf_counter()
        try:
            return self._target(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            node.total_time_seconds += elapsed
            node.hit_count += 1
            stack.pop()

            if is_root:
                root_node = _local_state.root
                _local_state.root = None
                if root_node:
                    try:
                        report = _format_hierarchy_table(root_node)
                        _append_report_to_log(report)
                    except Exception:
                        # Ensure profiling formatting errors never crash business logic
                        pass

    async def _call_async(self, *args: Any, **kwargs: Any) -> Any:
        func_name = self._name or _clean_name(getattr(self._target, "__qualname__", self._target.__name__))
        stack = _local_state.stack

        if not stack:
            node = ProfileNode(name=func_name)
            _local_state.root = node
            stack.append(node)
            is_root = True
        else:
            parent = stack[-1]
            node = parent.get_child(name=func_name, active_ancestors=stack)
            stack.append(node)
            is_root = False

        start_time = time.perf_counter()
        try:
            return await self._target(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            node.total_time_seconds += elapsed
            node.hit_count += 1
            stack.pop()

            if is_root:
                root_node = _local_state.root
                _local_state.root = None
                if root_node:
                    try:
                        report = _format_hierarchy_table(root_node)
                        _append_report_to_log(report)
                    except Exception:
                        pass
