"""Command line interface for visualization tool."""

from __future__ import annotations

import importlib.resources as importlib_resources
import os
import subprocess
import time
import webbrowser
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from threading import Thread

import click
from flask_cors import CORS

from tiatoolbox import logger
from tiatoolbox.cli.common import tiatoolbox_cli

_VIEWER_STARTUP_TIMEOUT_SECONDS = 60.0
_VIEWER_POLL_INTERVAL_SECONDS = 0.1
_VIEWER_REQUEST_TIMEOUT_SECONDS = 1.0


def run_tileserver(
    slide_roots: tuple[str | Path, ...] | list[str | Path] = (),
    overlay_roots: tuple[str | Path, ...] | list[str | Path] = (),
    port: int | None = None,
    *,
    block: bool = False,
    enable_cors: bool = True,
) -> None:
    """Launch the shared legacy and versioned tile server."""
    port = port or int(os.environ.get("TIATOOLBOX_TILESERVER_PORT", "5000"))

    def run_app() -> None:
        """Run the tileserver app."""
        from tiatoolbox.visualization.tileserver import TileServer  # noqa: PLC0415

        app = TileServer(
            title="Tiatoolbox TileServer",
            layers={},
            slide_roots=slide_roots,
            overlay_roots=overlay_roots,
        )
        try:
            app.json.sort_keys = False
            if enable_cors:
                CORS(app, send_wildcard=True)
            app.run(host="127.0.0.1", port=port, threaded=True)
        finally:
            app.viewer_services.close()

    if block:
        run_app()
        return
    proc = Thread(target=run_app, daemon=True)
    proc.start()


def _viewer_endpoint_is_ready(port: int, *, request_timeout: float) -> bool:
    """Return whether the local OpenLayers viewer endpoint is responding."""
    connection = HTTPConnection("127.0.0.1", port, timeout=request_timeout)
    try:
        connection.request("GET", "/viewer/")
        response = connection.getresponse()
    except (HTTPException, OSError):
        return False
    else:
        return HTTPStatus.OK <= response.status < HTTPStatus.BAD_REQUEST
    finally:
        connection.close()


def _open_browser_when_viewer_ready(
    port: int,
    *,
    startup_timeout: float = _VIEWER_STARTUP_TIMEOUT_SECONDS,
    poll_interval: float = _VIEWER_POLL_INTERVAL_SECONDS,
    request_timeout: float = _VIEWER_REQUEST_TIMEOUT_SECONDS,
) -> bool:
    """Wait for the local viewer endpoint and then open it in a browser."""
    url = f"http://127.0.0.1:{port}/viewer/"
    deadline = time.monotonic() + startup_timeout
    while not _viewer_endpoint_is_ready(port, request_timeout=request_timeout):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Viewer did not become ready at %s within %.0f seconds; "
                "the browser was not opened.",
                url,
                startup_timeout,
            )
            return False
        time.sleep(min(poll_interval, remaining))

    try:
        opened = webbrowser.open(url)
    except (OSError, webbrowser.Error) as error:
        logger.warning("Unable to open the viewer at %s: %s", url, error)
        return False
    if not opened:
        logger.warning("Unable to open the viewer at %s.", url)
    return opened


def run_openlayers(
    slide_roots: list[Path],
    overlay_roots: list[Path],
    port: int,
    *,
    noshow: bool,
) -> None:
    """Run the single-origin OpenLayers application."""
    if not noshow:
        browser_thread = Thread(
            target=_open_browser_when_viewer_ready,
            args=(port,),
            daemon=True,
            name="tiatoolbox-viewer-browser",
        )
        browser_thread.start()
    run_tileserver(
        slide_roots,
        overlay_roots,
        port,
        block=True,
        enable_cors=False,
    )


def run_bokeh(img_input: list[str], port: int, *, noshow: bool) -> None:
    """Start the bokeh server."""
    bokeh_path = importlib_resources.files("tiatoolbox.visualization.bokeh_app")
    cmd = [
        "bokeh",
        "serve",
    ]
    if not noshow:
        cmd = [*cmd, "--show"]  # pragma: no cover
    cmd = [
        *cmd,
        bokeh_path,
        "--port",
        str(port),
        "--unused-session-lifetime",
        "1000",
        "--check-unused-sessions",
        "1000",
        "--args",
        *img_input,
    ]
    subprocess.run(cmd, check=True, cwd=str(Path.cwd()), env=os.environ)  # noqa: S603


@tiatoolbox_cli.command()
@click.option(
    "--base-path",
    help="""Path to base directory containing images to be displayed.
    Slides and overlays to be visualized are expected in subdirectories of the
    base directory named slides and overlays, respectively. It is also possible
    to provide a slide and overlay path separately
    (use --slides and --overlays).""",
)
@click.option(
    "--slides",
    help="""Path to directory containing slides to be displayed.
    This option must be used in conjunction with --overlay-path.
    The --base-path option should not be used in this case.""",
)
@click.option(
    "--overlays",
    help="""Path to directory containing overlays to be displayed.
    This option must be used in conjunction with --slides.
    The --base-path option should not be used in this case.""",
)
@click.option(
    "--port",
    type=int,
    help="Port to launch the visualization tool on.",
    default=5006,
)
@click.option("--noshow", is_flag=True, help="Do not launch browser.")
@click.option(
    "--ui",
    type=click.Choice(("bokeh", "openlayers"), case_sensitive=False),
    default="bokeh",
    show_default=True,
    help="Viewer frontend to launch.",
)
def visualize(
    base_path: str,
    slides: str,
    overlays: str,
    port: int,
    *,
    noshow: bool,
    ui: str,
) -> None:
    """Launches the visualization tool for the given directory(s).

    If only base-path is given, Slides and overlays to be visualized are expected in
    subdirectories of the base directory named slides and overlays, respectively.

    Args:
        base_path (str): Path to base directory containing images to be displayed.
        slides (str): Path to directory containing slides to be displayed.
        overlays (str): Path to directory containing overlays to be displayed.
        port (int): Port to launch the visualization tool on.
        noshow (bool): Do not launch in browser (mainly intended for testing).
        ui (str): Viewer frontend to launch.

    """
    # sanity check the input args
    if base_path is None and (slides is None or overlays is None):
        msg = "Must specify either base-path or both slides and overlays."
        raise ValueError(msg)
    img_input = [base_path, slides, overlays]
    img_input = [p for p in img_input if p is not None]
    # check that the input paths exist
    for input_path in img_input:
        if not Path(input_path).exists():
            msg = f"{input_path} does not exist"
            raise FileNotFoundError(msg)

    if base_path is not None:
        slide_roots = [Path(base_path) / "slides"]
        overlay_roots = [Path(base_path) / "overlays"]
    else:
        slide_roots = [Path(slides)]
        overlay_roots = [Path(overlays)]
    for resource_root in (*slide_roots, *overlay_roots):
        if not resource_root.is_dir():
            msg = f"{resource_root} does not exist or is not a directory"
            raise FileNotFoundError(msg)

    # Keep Bokeh as the default compatibility path until the phase-4 gates pass.
    if ui.lower() == "bokeh":
        run_tileserver(slide_roots, overlay_roots)  # pragma: no cover
        run_bokeh(img_input, port, noshow=noshow)  # pragma: no cover
        return
    run_openlayers(  # pragma: no cover
        slide_roots,
        overlay_roots,
        port,
        noshow=noshow,
    )
