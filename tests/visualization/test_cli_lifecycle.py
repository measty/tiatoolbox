"""Lifecycle tests for the OpenLayers tile-server entry points."""

from __future__ import annotations

import importlib
from http.client import HTTPException
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import pytest

from tiatoolbox.cli.visualize import run_openlayers, run_tileserver

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

visualize_module = importlib.import_module("tiatoolbox.cli.visualize")


class _FakeResponse:
    """Minimal HTTP response exposing a status code."""

    def __init__(self, status: int) -> None:
        self.status = status


class _FakeConnection:
    """Record a readiness probe and return a configured response."""

    instances: ClassVar[list[_FakeConnection]] = []
    status = 200
    error: Exception | None = None

    def __init__(self, host: str, port: int, *, timeout: float) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.request_args: tuple[str, str] | None = None
        self.closed = False
        type(self).instances.append(self)

    def request(self, method: str, path: str) -> None:
        """Record the requested endpoint."""
        self.request_args = (method, path)

    def getresponse(self) -> _FakeResponse:
        """Return the configured response or raise the configured error."""
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.status)

    def close(self) -> None:
        """Record that the connection was released."""
        self.closed = True


class _FakeServices:
    """Record deterministic service shutdown calls."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        """Record one close call."""
        self.close_calls += 1


class _FakeTileServer:
    """Minimal TileServer substitute with a configurable run failure."""

    instance: _FakeTileServer | None = None
    run_error: Exception | None = None

    def __init__(self, **_kwargs: object) -> None:
        type(self).instance = self
        self.json = SimpleNamespace(sort_keys=True)
        self.viewer_services = _FakeServices()
        self.run_kwargs: dict[str, object] | None = None

    def run(self, **kwargs: object) -> None:
        """Record invocation and optionally emulate a Flask failure."""
        self.run_kwargs = kwargs
        if self.run_error is not None:
            raise self.run_error


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (200, None, True),
        (302, None, True),
        (503, None, False),
        (200, HTTPException("not ready"), False),
    ],
)
def test_viewer_endpoint_readiness_probe(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error: Exception | None,
    expected: object,
) -> None:
    """Only a successful response marks the viewer as ready."""
    _FakeConnection.instances.clear()
    _FakeConnection.status = status
    _FakeConnection.error = error
    monkeypatch.setattr(visualize_module, "HTTPConnection", _FakeConnection)

    assert (
        visualize_module._viewer_endpoint_is_ready(8765, request_timeout=0.25)
        is expected
    )
    connection = _FakeConnection.instances[0]
    assert (connection.host, connection.port, connection.timeout) == (
        "127.0.0.1",
        8765,
        0.25,
    )
    assert connection.request_args == ("GET", "/viewer/")
    assert connection.closed


def test_browser_opens_only_after_viewer_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient connection failures do not expose a dead browser tab."""
    readiness = iter((False, False, True))
    readiness_calls: list[tuple[int, float]] = []
    delays: list[float] = []
    opened_urls: list[str] = []

    def fake_ready(port: int, *, request_timeout: float) -> bool:
        readiness_calls.append((port, request_timeout))
        return next(readiness)

    monkeypatch.setattr(visualize_module, "_viewer_endpoint_is_ready", fake_ready)
    monkeypatch.setattr(visualize_module.time, "sleep", delays.append)
    monkeypatch.setattr(
        visualize_module.webbrowser,
        "open",
        lambda url: not opened_urls.append(url),
    )

    assert visualize_module._open_browser_when_viewer_ready(
        8765,
        startup_timeout=1,
        poll_interval=0.01,
        request_timeout=0.25,
    )
    assert readiness_calls == [(8765, 0.25)] * 3
    assert delays == [0.01, 0.01]
    assert opened_urls == ["http://127.0.0.1:8765/viewer/"]


def test_browser_is_not_opened_after_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded readiness timeout leaves the browser closed."""
    opened_urls: list[str] = []
    monkeypatch.setattr(
        visualize_module,
        "_viewer_endpoint_is_ready",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        visualize_module.webbrowser,
        "open",
        lambda url: not opened_urls.append(url),
    )

    assert not visualize_module._open_browser_when_viewer_ready(
        8765,
        startup_timeout=0,
    )
    assert opened_urls == []


@pytest.mark.parametrize("outcome", ["return", "raise"])
def test_run_tileserver_always_closes_services(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    """Returning and exceptional Flask runs both release viewer services."""
    from tiatoolbox.visualization import tileserver  # noqa: PLC0415

    error = RuntimeError("run failed") if outcome == "raise" else None
    monkeypatch.setattr(_FakeTileServer, "run_error", error)
    monkeypatch.setattr(tileserver, "TileServer", _FakeTileServer)

    if outcome == "raise":
        with pytest.raises(RuntimeError, match="run failed"):
            run_tileserver(port=8765, block=True, enable_cors=False)
    else:
        run_tileserver(port=8765, block=True, enable_cors=False)

    app = _FakeTileServer.instance
    assert app is not None
    assert app.run_kwargs == {
        "host": "127.0.0.1",
        "port": 8765,
        "threaded": True,
    }
    assert app.viewer_services.close_calls == 1


def test_run_openlayers_uses_blocking_lifecycle_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """OpenLayers delegates to the synchronous no-CORS server lifecycle."""
    calls: list[tuple[object, ...]] = []

    def fake_run_tileserver(*args: object, **kwargs: object) -> None:
        calls.append((*args, kwargs))

    monkeypatch.setattr(visualize_module, "run_tileserver", fake_run_tileserver)
    slide_roots = [tmp_path / "slides"]
    overlay_roots = [tmp_path / "overlays"]

    run_openlayers(slide_roots, overlay_roots, 8765, noshow=True)

    assert calls == [
        (
            slide_roots,
            overlay_roots,
            8765,
            {"block": True, "enable_cors": False},
        ),
    ]


def test_run_openlayers_starts_browser_readiness_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Browser launch waits independently while the server owns the main thread."""
    ready_ports: list[int] = []

    class _ImmediateThread:
        """Run a captured thread target synchronously for this lifecycle test."""

        def __init__(
            self,
            *,
            target: Callable[..., object],
            args: tuple[object, ...],
            daemon: bool,
            name: str,
        ) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon
            self.name = name
            threads.append(self)

        def start(self) -> None:
            """Invoke the captured target."""
            self.target(*self.args)

    threads: list[_ImmediateThread] = []

    monkeypatch.setattr(visualize_module, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        visualize_module,
        "_open_browser_when_viewer_ready",
        ready_ports.append,
    )
    monkeypatch.setattr(
        visualize_module,
        "run_tileserver",
        lambda *_args, **_kwargs: None,
    )

    run_openlayers(
        [tmp_path / "slides"],
        [tmp_path / "overlays"],
        8765,
        noshow=False,
    )

    assert ready_ports == [8765]
    assert len(threads) == 1
    assert threads[0].daemon
    assert threads[0].name == "tiatoolbox-viewer-browser"
