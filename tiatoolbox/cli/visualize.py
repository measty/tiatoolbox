"""Command line interface for visualization tool."""

from __future__ import annotations

import importlib.resources as importlib_resources
import os
import subprocess
import webbrowser
from pathlib import Path
from threading import Thread, Timer

import click
from flask_cors import CORS

from tiatoolbox.cli.common import tiatoolbox_cli


def run_tileserver() -> None:
    """Helper function to launch a tileserver."""

    def run_app() -> None:
        """Run the tileserver app."""
        from tiatoolbox.visualization.tileserver import TileServer  # noqa: PLC0415

        app = TileServer(
            title="Tiatoolbox TileServer",
            layers={},
        )
        app.json.sort_keys = False
        CORS(app, send_wildcard=True)
        port = int(os.environ.get("TIATOOLBOX_TILESERVER_PORT", "5000"))
        app.run(host="127.0.0.1", port=port, threaded=True)

    proc = Thread(target=run_app, daemon=True)
    proc.start()


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


def resolve_visualization_paths(
    base_path: str | None,
    slides: str | None,
    overlays: str | None,
) -> tuple[Path, Path]:
    """Resolve and validate visualize CLI paths."""
    if base_path is None and (slides is None or overlays is None):
        msg = "Must specify either base-path or both slides and overlays."
        raise ValueError(msg)

    if base_path is not None:
        slide_path = Path(base_path) / "slides"
        overlay_path = Path(base_path) / "overlays"
        paths_to_check = [Path(base_path), slide_path, overlay_path]
    else:
        slide_path = Path(slides)
        overlay_path = Path(overlays)
        paths_to_check = [slide_path, overlay_path]

    for input_path in paths_to_check:
        if not input_path.exists():
            msg = f"{input_path} does not exist"
            raise FileNotFoundError(msg)

    return slide_path, overlay_path


def run_visualizer(slides: Path, overlays: Path, port: int, *, noshow: bool) -> None:
    """Run the Flask/OpenLayers visualization frontend."""
    from tiatoolbox.visualization.tileserver import TileServer  # noqa: PLC0415

    app = TileServer(
        title="TIAToolbox Visualizer",
        layers={},
        project_config={
            "slide_folder": slides,
            "overlay_folder": overlays,
        },
    )
    app.json.sort_keys = False
    CORS(app, send_wildcard=True)

    if not noshow:  # pragma: no cover
        Timer(
            1,
            lambda: webbrowser.open_new_tab(f"http://127.0.0.1:{port}/"),
        ).start()

    app.run(host="127.0.0.1", port=port, threaded=True)


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
def visualize(
    base_path: str,
    slides: str,
    overlays: str,
    port: int,
    *,
    noshow: bool,
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

    """
    slide_path, overlay_path = resolve_visualization_paths(base_path, slides, overlays)
    run_visualizer(slide_path, overlay_path, port, noshow=noshow)  # pragma: no cover
