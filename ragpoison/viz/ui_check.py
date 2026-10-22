"""Headless verification of the dashboard.

Two backends, because one of them cannot run everywhere:

``apptest``
    ``streamlit.testing.v1.AppTest`` runs the real app through the real Streamlit
    runtime and reports its tabs, widgets and exceptions. This is the strongest check
    and it is what a normal machine will use.

``stub``
    The app file is imported with a recording double in place of ``streamlit``. Every
    layout call, every data access and every figure then executes for real; only the
    browser plumbing is faked. This exists because Streamlit creates and deletes a
    temporary directory at startup, and some sandboxed environments refuse the
    permission change that cleanup needs, which aborts the whole check for a reason
    that has nothing to do with the dashboard.

Both backends render the figures to ``figures/`` as PNGs, which is how the chart layer
gets looked at when there is no screenshot to look at.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..paths import path as repo_path, rel

APP = Path(__file__).with_name("streamlit_app.py")

#: The dashboard must produce exactly this many tabs.
EXPECTED_TABS = 5


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _export_figures() -> list[str]:
    """Delegate to the figures command so there is one code path, not two."""
    from ..viz import figures

    return figures.export_all(42)


# --------------------------------------------------------------------------- #
# Backend: the recording double
# --------------------------------------------------------------------------- #


class _Section:
    """A context manager that forwards layout calls to the recorder."""

    def __init__(self, recorder: "_Stub") -> None:
        self._recorder = recorder

    def __enter__(self) -> "_Stub":
        return self._recorder

    def __exit__(self, *exc_info) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self._recorder, name)


class _Stub:
    """The subset of the Streamlit API the dashboard uses, with recording."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.tab_labels: list[str] = []
        self.log: list[tuple[str, tuple]] = []

    # -- internal ---------------------------------------------------------
    def _hit(self, name: str, *args) -> None:
        self.counts[name] += 1
        if len(self.log) < 500:
            self.log.append((name, args))

    def _section(self) -> _Section:
        return _Section(self)

    # -- layout -----------------------------------------------------------
    def cache_data(self, *args, **kwargs):
        """Decorator-compatible no-op: the double caches nothing, it just wraps."""

        def decorator(function):
            return function

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    def set_page_config(self, *args, **kwargs):
        self._hit("set_page_config", *args)

    def title(self, *args, **kwargs):
        self._hit("title", *args)

    def header(self, *args, **kwargs):
        self._hit("header", *args)

    def subheader(self, *args, **kwargs):
        self._hit("subheader", *args)

    def caption(self, *args, **kwargs):
        self._hit("caption", *args)

    def markdown(self, *args, **kwargs):
        self._hit("markdown", *args)

    def write(self, *args, **kwargs):
        self._hit("write", *args)

    def code(self, *args, **kwargs):
        self._hit("code", *args)

    def info(self, *args, **kwargs):
        self._hit("info", *args)

    def warning(self, *args, **kwargs):
        self._hit("warning", *args)

    def error(self, *args, **kwargs):
        self._hit("error", *args)

    def progress(self, *args, **kwargs):
        self._hit("progress", *args)

    @property
    def sidebar(self) -> _Section:
        return self._section()

    def expander(self, *args, **kwargs) -> _Section:
        self._hit("expander", *args)
        return self._section()

    def container(self, *args, **kwargs) -> _Section:
        return self._section()

    def tabs(self, labels: list[str]) -> list[_Section]:
        self._hit("tabs", *labels)
        self.tab_labels = list(labels)
        return [self._section() for _ in labels]

    def columns(self, spec) -> list[_Section]:
        count = spec if isinstance(spec, int) else len(spec)
        self._hit("columns", count)
        return [self._section() for _ in range(count)]

    # -- widgets ----------------------------------------------------------
    def selectbox(self, label, options, index=0, *args, **kwargs):
        self._hit("selectbox", label)
        values = list(options)
        if not values:
            return None
        return values[min(int(index or 0), len(values) - 1)]

    def multiselect(self, label, options, *args, **kwargs):
        self._hit("multiselect", label)
        return list(kwargs.get("default") or [])

    def radio(self, label, options, *args, **kwargs):
        self._hit("radio", label)
        values = list(options)
        return values[0] if values else None

    def slider(self, label, *args, **kwargs):
        """Mirror Streamlit's positional order: label, min, max, value, step."""
        self._hit("slider", label)
        if "value" in kwargs:
            return kwargs["value"]
        if len(args) >= 3:
            return args[2]
        if args:
            return args[0]
        return 0.0

    def number_input(self, label, *args, **kwargs):
        self._hit("number_input", label)
        return kwargs.get("value", 0)

    def checkbox(self, label, *args, **kwargs):
        self._hit("checkbox", label)
        return bool(kwargs.get("value", False))

    def button(self, label, *args, **kwargs):
        self._hit("button", label)
        return False

    def dataframe(self, data, *args, **kwargs):
        self._hit("dataframe", len(data) if hasattr(data, "__len__") else 0)

    def metric(self, label, value, *args, **kwargs):
        self._hit("metric", label)

    def pyplot(self, figure=None, *args, **kwargs):
        self._hit("pyplot")
        if figure is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)


def _run_stub() -> dict:
    """Import the app file with the double in place of streamlit."""
    import importlib.util
    import sys

    stub = _Stub()
    previous = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub  # type: ignore[assignment]
    try:
        module_name = "ragpoison.viz._app_under_test"
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, APP)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
    finally:
        if previous is not None:
            sys.modules["streamlit"] = previous
        else:
            sys.modules.pop("streamlit", None)

    return {
        "backend": "stub",
        "tabs": len(stub.tab_labels),
        "tab_labels": stub.tab_labels,
        "metrics": stub.counts["metric"],
        "dataframes": stub.counts["dataframe"],
        "sliders": stub.counts["slider"],
        "buttons": stub.counts["button"],
        "figures_drawn": stub.counts["pyplot"],
        "selectboxes": stub.counts["selectbox"],
        "exceptions": [],
        "errors": [],
    }


# --------------------------------------------------------------------------- #
# Backend: the real Streamlit runtime
# --------------------------------------------------------------------------- #


def _run_apptest() -> dict:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    tabs = [tab.label for tab in app.tabs] if hasattr(app, "tabs") else []
    return {
        "backend": "apptest",
        "tabs": len(tabs),
        "tab_labels": tabs,
        "metrics": len(app.metric),
        "dataframes": len(app.dataframe),
        "sliders": len(app.slider),
        "buttons": len(app.button),
        "figures_drawn": len(app.get("arrow_vega_lite_chart")) if hasattr(app, "get") else 0,
        "selectboxes": len(app.selectbox),
        "exceptions": [str(item.value) for item in app.exception],
        "errors": [str(item.value) for item in app.error],
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_ui_check(export: bool = True, backend: str = "auto") -> dict:
    """Render the dashboard headlessly and report what happened."""
    checks: dict = {}
    if backend in ("auto", "apptest"):
        try:
            checks.update(_run_apptest())
        except Exception as exc:  # noqa: BLE001 - any environment failure falls back
            checks["apptest_error"] = f"{type(exc).__name__}: {exc}"
            if backend == "apptest":
                checks["passed"] = False
                return checks
    if not checks.get("backend"):
        checks.update(_run_stub())

    if export and not checks.get("exceptions"):
        checks["figures"] = _export_figures()

    checks["tabs_expected"] = EXPECTED_TABS
    checks["passed"] = (
        not checks.get("exceptions")
        and not checks.get("errors")
        and checks.get("tabs") == EXPECTED_TABS
        and checks.get("metrics", 0) >= 6
        and checks.get("dataframes", 0) >= 2
        and checks.get("sliders", 0) >= 8
    )
    return checks
