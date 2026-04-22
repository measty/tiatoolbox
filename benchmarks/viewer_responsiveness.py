"""Backend benchmark harness for viewer overlay responsiveness.

This harness measures the local Flask TileServer backend paths used by the
OpenLayers viewer when loading a slide and a SQLite-backed annotation overlay.
It does not launch a browser, so frontend fetch, decode, style, and paint work
are intentionally out of scope. The reported first-visible-overlay metric is a
backend proxy based on the first metadata request plus the first uncached MVT
response.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import platform
import statistics
import sys
import time
import typing
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UTC = getattr(dt, "UTC", dt.timezone.utc)

DEFAULT_SLIDE_PATH = Path(
    "/media/mark-eastwood/Work/TTB_vis_test/slides/TCGA-SC-A6LN-01Z-00-DX1.svs",
)
DEFAULT_OVERLAY_PATH = Path(
    "/media/mark-eastwood/Work/TTB_vis_test/overlays/"
    "TCGA-SC-A6LN-01Z-00-DX1_saved_anns.db",
)
DEFAULT_LAYER_NAME = "overlay"
FULL_REPRESENTATION = "full"
DEFAULT_COLOR_PROPERTY = "type"


@dataclass(frozen=True)
class TileSpec:
    """A deterministic annotation vector tile to request."""

    representation: str
    z: int
    x: int
    y: int

    @property
    def label(self) -> str:
        """Return a compact stable label for output."""
        return f"{self.representation}:{self.z}:{self.x}:{self.y}"


@dataclass(frozen=True)
class TileProbe:
    """A selected tile together with how it was chosen."""

    label: str
    tile: TileSpec
    grid_width: int
    grid_height: int
    sampled_tiles: int
    candidate_count: int
    prefiltered: bool


@contextlib.contextmanager
def capture_annotation_perf(tile_server_cls: Any) -> Iterator[list[dict[str, Any]]]:
    """Capture TileServer annotation perf hooks as structured records."""

    records: list[dict[str, Any]] = []
    original = tile_server_cls._log_annotation_perf

    def capture(event: str, elapsed_seconds: float, **fields: object) -> None:
        records.append(
            {
                "event": event,
                "elapsed_ms": round(elapsed_seconds * 1000, 3),
                "fields": dict(fields),
            },
        )
        original(event, elapsed_seconds, **fields)

    tile_server_cls._log_annotation_perf = staticmethod(capture)
    try:
        yield records
    finally:
        tile_server_cls._log_annotation_perf = staticmethod(original)


def add_repo_root_to_path() -> None:
    """Allow running this script from the repository root or directly by path."""

    if not hasattr(typing, "Self"):
        from typing_extensions import Self

        typing.Self = Self
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


def load_tile_server_class() -> Any:
    """Import TileServer lazily after the repository root is on sys.path."""

    add_repo_root_to_path()
    from tiatoolbox.visualization import TileServer

    return TileServer


def parse_tile_spec(value: str) -> TileSpec:
    """Parse REPR:Z:X:Y or Z:X:Y tile specifications."""

    parts = value.split(":")
    if len(parts) == 3:
        representation = FULL_REPRESENTATION
        coord_parts = parts
    elif len(parts) == 4:
        representation = parts[0].strip() or FULL_REPRESENTATION
        coord_parts = parts[1:]
    else:
        msg = "tile must be REPR:Z:X:Y or Z:X:Y"
        raise ValueError(msg)

    try:
        z, x, y = (int(part) for part in coord_parts)
    except ValueError as exc:
        msg = "tile coordinates must be integers"
        raise ValueError(msg) from exc

    if z < 0 or x < 0 or y < 0:
        msg = "tile coordinates must be non-negative"
        raise ValueError(msg)
    return TileSpec(representation=representation, z=z, x=x, y=y)


def build_mvt_path(
    layer_name: str,
    session_id: str,
    tile: TileSpec,
    *,
    color_property: str | None = None,
    fields: tuple[str, ...] = (),
    where: str | None = None,
) -> str:
    """Build the MVT endpoint path used by the OpenLayers frontend."""

    quoted_layer = urllib.parse.quote(layer_name, safe="")
    prefix = f"/tileserver/layer/{quoted_layer}/{session_id}/mvt"
    if tile.representation == FULL_REPRESENTATION:
        path = f"{prefix}/{tile.z}/{tile.x}/{tile.y}.pbf"
    else:
        path = f"{prefix}/{tile.representation}/{tile.z}/{tile.x}/{tile.y}.pbf"

    query: dict[str, str] = {}
    if color_property:
        query["cprop"] = color_property
    if fields:
        query["fields"] = json.dumps(list(fields), separators=(",", ":"))
    if where:
        query["where"] = where
    if not query:
        return path
    return f"{path}?{urllib.parse.urlencode(query)}"


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise endpoint timing samples with stable JSON scalar values."""

    if not samples:
        return {
            "count": 0,
            "elapsed_ms_min": None,
            "elapsed_ms_median": None,
            "elapsed_ms_mean": None,
            "elapsed_ms_p95": None,
            "elapsed_ms_max": None,
            "status_codes": [],
            "bytes_min": None,
            "bytes_max": None,
        }

    elapsed = sorted(float(sample["elapsed_ms"]) for sample in samples)
    p95_index = min(len(elapsed) - 1, max(0, math.ceil(len(elapsed) * 0.95) - 1))
    status_codes = sorted({int(sample["status_code"]) for sample in samples})
    byte_counts = [int(sample["bytes"]) for sample in samples]
    return {
        "count": len(samples),
        "elapsed_ms_min": round(min(elapsed), 3),
        "elapsed_ms_median": round(statistics.median(elapsed), 3),
        "elapsed_ms_mean": round(statistics.fmean(elapsed), 3),
        "elapsed_ms_p95": round(elapsed[p95_index], 3),
        "elapsed_ms_max": round(max(elapsed), 3),
        "status_codes": status_codes,
        "bytes_min": min(byte_counts),
        "bytes_max": max(byte_counts),
    }


def first_visible_overlay_proxy(
    metadata_endpoint: dict[str, Any],
    tile_endpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a backend-only proxy for time to first visible overlay."""

    if tile_endpoint is None:
        return None
    metadata_sample = metadata_endpoint["samples"][0]
    tile_sample = tile_endpoint["samples"][0]
    return {
        "description": (
            "Backend proxy only: first metadata refresh plus first selected "
            "vector tile response. Browser fetch, decode, style, and paint "
            "timing are not captured."
        ),
        "elapsed_ms": round(
            float(metadata_sample["elapsed_ms"]) + float(tile_sample["elapsed_ms"]),
            3,
        ),
        "sequence": [
            {
                "label": metadata_endpoint["label"],
                "method": metadata_endpoint["method"],
                "path": metadata_endpoint["path"],
                "elapsed_ms": metadata_sample["elapsed_ms"],
                "status_code": metadata_sample["status_code"],
            },
            {
                "label": tile_endpoint["label"],
                "method": tile_endpoint["method"],
                "path": tile_endpoint["path"],
                "elapsed_ms": tile_sample["elapsed_ms"],
                "status_code": tile_sample["status_code"],
            },
        ],
    }


def _json_body(response: Any) -> Any:
    return json.loads(response.get_data(as_text=True))


def _request_once(
    client: Any,
    *,
    method: str,
    path: str,
    perf_records: list[dict[str, Any]],
    data: dict[str, str] | None = None,
) -> dict[str, Any]:
    client_method = getattr(client, method.lower())
    before_perf = len(perf_records)
    start = time.perf_counter()
    response = client_method(path, data=data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    payload = response.get_data()
    sample = {
        "elapsed_ms": round(elapsed_ms, 3),
        "status_code": int(response.status_code),
        "content_type": response.content_type,
        "bytes": len(payload),
        "annotation_perf": perf_records[before_perf:],
    }
    etag = response.headers.get("ETag")
    if etag:
        sample["etag"] = etag
    return {"response": response, "sample": sample}


def measure_endpoint(
    client: Any,
    *,
    label: str,
    method: str,
    path: str,
    perf_records: list[dict[str, Any]],
    warm_runs: int,
    data: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure one backend endpoint with one cold request and warm repeats."""

    cold_result = _request_once(
        client,
        method=method,
        path=path,
        perf_records=perf_records,
        data=data,
    )
    warm_samples = [
        _request_once(
            client,
            method=method,
            path=path,
            perf_records=perf_records,
            data=data,
        )["sample"]
        for _ in range(warm_runs)
    ]
    return {
        "label": label,
        "method": method.upper(),
        "path": path,
        "cold": cold_result["sample"],
        "warm_summary": summarize_samples(warm_samples),
        "samples": [cold_result["sample"], *warm_samples],
        "response": cold_result["response"],
    }


def validate_input_path(path: Path, label: str) -> None:
    """Fail early with a useful message for missing benchmark data."""

    if not path.is_file():
        msg = f"{label} does not exist or is not a file: {path}"
        raise FileNotFoundError(msg)


def create_app() -> Any:
    """Create an empty TileServer instance for backend request benchmarking."""

    tile_server = load_tile_server_class()
    app = tile_server("Viewer Responsiveness Benchmark", {})
    app.config.from_mapping({"TESTING": True})
    return app


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure backend TileServer responsiveness for viewer slide, "
            "annotation metadata, and MVT vector tile requests."
        ),
    )
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "SLIDE", "OVERLAY"),
        help="Named dataset to benchmark. Repeat for multiple datasets.",
    )
    parser.add_argument("--slide", type=Path, default=DEFAULT_SLIDE_PATH)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY_PATH)
    parser.add_argument("--layer-name", default=DEFAULT_LAYER_NAME)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument(
        "--sample-points",
        type=int,
        default=3,
        help="Number of evenly spaced tile positions to sample on each axis.",
    )
    parser.add_argument(
        "--tile",
        action="append",
        default=[],
        help="Tile to measure as REPR:Z:X:Y or Z:X:Y. May be passed more than once.",
    )
    parser.add_argument(
        "--color-property",
        default=DEFAULT_COLOR_PROPERTY,
        help="Annotation property requested as the vector tile colour field.",
    )
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="Extra annotation property to include in vector tile requests.",
    )
    parser.add_argument(
        "--where",
        help="Optional TileServer where filter passed through to vector tile requests.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON output to this path.")
    parser.add_argument(
        "--json-out",
        dest="output",
        type=Path,
        help="Alias for --output.",
    )
    return parser


def _session_id_from_client(client: Any) -> str:
    session_cookie = client.get_cookie("session_id")
    if session_cookie is None:
        msg = "TileServer did not return a session cookie."
        raise RuntimeError(msg)
    return str(session_cookie.value)


def evenly_spaced_indices(length: int, sample_points: int) -> list[int]:
    """Return a stable sample of indices across a grid axis."""

    if length <= 0:
        return []
    if length <= sample_points:
        return list(range(length))
    return sorted(
        {
            round(position * (length - 1) / (sample_points - 1))
            for position in range(sample_points)
        },
    )


def select_dense_tile(
    app: Any,
    *,
    session_id: str,
    layer_name: str,
    representation: str,
    z: int,
    color_property: str | None,
    sample_points: int,
) -> TileProbe:
    """Pick a dense tile by sampling the target zoom level across the grid."""

    ann_layer = app.get_ann_layer(session_id, layer_name=layer_name)
    grid_width, grid_height = ann_layer.tile_grid_size(z)
    x_positions = evenly_spaced_indices(grid_width, sample_points)
    y_positions = evenly_spaced_indices(grid_height, sample_points)
    render_hints = app._get_annotation_mvt_hints(ann_layer, z)  # noqa: SLF001
    center_x = (grid_width - 1) / 2
    center_y = (grid_height - 1) / 2

    best_probe: TileProbe | None = None
    best_score: tuple[int, int] | None = None
    for x_index in x_positions:
        for y_index in y_positions:
            tile_bounds = app._get_annotation_tile_bounds(ann_layer, z, x_index, y_index)  # noqa: SLF001
            annotations, prefiltered = app._query_annotations_for_mvt(  # noqa: SLF001
                ann_layer,
                tile_bounds,
                None,
                render_hints,
                representation,
                color_property,
                (),
            )
            candidate_count = len(annotations)
            distance_score = -int(abs(x_index - center_x) + abs(y_index - center_y))
            score = (candidate_count, distance_score)
            if best_score is not None and score <= best_score:
                continue
            best_probe = TileProbe(
                label=representation,
                tile=TileSpec(representation=representation, z=z, x=x_index, y=y_index),
                grid_width=grid_width,
                grid_height=grid_height,
                sampled_tiles=len(x_positions) * len(y_positions),
                candidate_count=candidate_count,
                prefiltered=bool(prefiltered),
            )
            best_score = score

    if best_probe is None:
        msg = f"No tile could be selected for representation {representation!r} at z={z}."
        raise RuntimeError(msg)
    return best_probe


def select_default_tiles(
    app: Any,
    *,
    session_id: str,
    layer_name: str,
    layer_metadata: dict[str, Any],
    color_property: str | None,
    sample_points: int,
) -> list[TileProbe]:
    """Select representative dense tiles from the advertised vector reps."""

    ann_layer = app.get_ann_layer(session_id, layer_name=layer_name)
    representations = layer_metadata.get("vector_representations") or [
        {"id": FULL_REPRESENTATION, "min_zoom": 0},
    ]
    probes: list[TileProbe] = []
    seen: set[tuple[str, int]] = set()
    level_count = ann_layer.level_count

    for representation in representations:
        rep_id = str(representation.get("id", FULL_REPRESENTATION))
        min_zoom = int(representation.get("min_zoom", 0))
        max_zoom = int(representation.get("max_zoom", level_count - 1))
        zoom_targets: list[tuple[str, int]]
        midpoint = min_zoom + ((max_zoom - min_zoom) // 2)
        if rep_id == FULL_REPRESENTATION:
            zoom_targets = [("full_entry", min_zoom)]
            detail_zoom = max(min_zoom, level_count - 2)
            if detail_zoom != min_zoom:
                zoom_targets.append(("full_detail", detail_zoom))
        else:
            zoom_targets = [(rep_id, midpoint)]

        for label, z in zoom_targets:
            key = (rep_id, z)
            if key in seen:
                continue
            seen.add(key)
            probe = select_dense_tile(
                app,
                session_id=session_id,
                layer_name=layer_name,
                representation=rep_id,
                z=z,
                color_property=color_property,
                sample_points=sample_points,
            )
            probes.append(
                TileProbe(
                    label=label,
                    tile=probe.tile,
                    grid_width=probe.grid_width,
                    grid_height=probe.grid_height,
                    sampled_tiles=probe.sampled_tiles,
                    candidate_count=probe.candidate_count,
                    prefiltered=probe.prefiltered,
                ),
            )
    return probes


def _extract_layer_metadata(layers: list[dict[str, Any]], layer_name: str) -> dict[str, Any]:
    for layer in layers:
        if layer.get("name") == layer_name:
            return layer
    msg = f"layer {layer_name!r} was not returned by /tileserver/layers"
    raise RuntimeError(msg)


def run_single_case(args: argparse.Namespace) -> dict[str, Any]:
    """Run the benchmark for one slide and overlay pair."""

    if args.warm_runs < 1:
        msg = "--warm-runs must be at least 1"
        raise ValueError(msg)
    if args.sample_points < 2:
        msg = "--sample-points must be at least 2"
        raise ValueError(msg)

    validate_input_path(args.slide, "slide")
    validate_input_path(args.overlay, "overlay")

    tile_server_cls = load_tile_server_class()
    with capture_annotation_perf(tile_server_cls) as perf_records:
        app_start = time.perf_counter()
        app = create_app()
        app_init_ms = (time.perf_counter() - app_start) * 1000

        with app.test_client() as client:
            session_endpoint = measure_endpoint(
                client,
                label="session_id",
                method="GET",
                path="/tileserver/session_id",
                perf_records=perf_records,
                warm_runs=args.warm_runs,
            )
            session_id = _session_id_from_client(client)

            slide_endpoint = measure_endpoint(
                client,
                label="slide_load",
                method="PUT",
                path="/tileserver/slide",
                data={"slide_path": str(args.slide)},
                perf_records=perf_records,
                warm_runs=args.warm_runs,
            )
            overlay_endpoint = measure_endpoint(
                client,
                label="overlay_load",
                method="PUT",
                path="/tileserver/overlay",
                data={"overlay_path": str(args.overlay)},
                perf_records=perf_records,
                warm_runs=args.warm_runs,
            )

            property_query = urllib.parse.urlencode({"layer_name": args.layer_name})
            property_names_endpoint = measure_endpoint(
                client,
                label="property_names_all",
                method="GET",
                path=f"/tileserver/prop_names/all?{property_query}",
                perf_records=perf_records,
                warm_runs=args.warm_runs,
            )
            properties = _json_body(property_names_endpoint["response"])
            summary_property = (
                args.color_property if args.color_property in properties else properties[0]
            ) if properties else None

            property_summary_endpoint = None
            if summary_property is not None:
                property_summary_endpoint = measure_endpoint(
                    client,
                    label=f"property_summary_{summary_property}_all",
                    method="GET",
                    path=(
                        f"/tileserver/prop_summary/"
                        f"{urllib.parse.quote(summary_property, safe='')}/all?"
                        f"{property_query}"
                    ),
                    perf_records=perf_records,
                    warm_runs=args.warm_runs,
                )

            layers_response = client.get("/tileserver/layers")
            layers = _json_body(layers_response)
            layer_metadata = _extract_layer_metadata(layers, args.layer_name)

            if args.tile:
                tile_probes = [
                    TileProbe(
                        label=parse_tile_spec(tile).label,
                        tile=parse_tile_spec(tile),
                        grid_width=0,
                        grid_height=0,
                        sampled_tiles=0,
                        candidate_count=0,
                        prefiltered=False,
                    )
                    for tile in args.tile
                ]
            else:
                tile_probes = select_default_tiles(
                    app,
                    session_id=session_id,
                    layer_name=args.layer_name,
                    layer_metadata=layer_metadata,
                    color_property=summary_property,
                    sample_points=args.sample_points,
                )

            tile_endpoints: list[dict[str, Any]] = []
            for tile_probe in tile_probes:
                endpoint = measure_endpoint(
                    client,
                    label=f"mvt_{tile_probe.label}",
                    method="GET",
                    path=build_mvt_path(
                        args.layer_name,
                        session_id,
                        tile_probe.tile,
                        color_property=summary_property,
                        fields=tuple(args.field),
                        where=args.where,
                    ),
                    perf_records=perf_records,
                    warm_runs=args.warm_runs,
                )
                endpoint["tile_probe"] = {
                    "label": tile_probe.label,
                    "representation": tile_probe.tile.representation,
                    "z": tile_probe.tile.z,
                    "x": tile_probe.tile.x,
                    "y": tile_probe.tile.y,
                    "grid_width": tile_probe.grid_width,
                    "grid_height": tile_probe.grid_height,
                    "sampled_tiles": tile_probe.sampled_tiles,
                    "candidate_count": tile_probe.candidate_count,
                    "prefiltered": tile_probe.prefiltered,
                }
                tile_endpoints.append(endpoint)

        first_tile_endpoint = tile_endpoints[0] if tile_endpoints else None
        return {
            "benchmark": "viewer_responsiveness_backend",
            "schema_version": 2,
            "created_at": dt.datetime.now(UTC).isoformat(),
            "scope": {
                "frontend_paint_timing_captured": False,
                "backend_proxy": "property metadata plus first selected MVT response",
                "cache_policy": "one cold request followed by warm repeats",
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "dataset": {
                "slide_path": str(args.slide),
                "slide_bytes": args.slide.stat().st_size,
                "overlay_path": str(args.overlay),
                "overlay_bytes": args.overlay.stat().st_size,
                "layer_name": args.layer_name,
            },
            "app_init": {"elapsed_ms": round(app_init_ms, 3)},
            "session_endpoint": {k: v for k, v in session_endpoint.items() if k != "response"},
            "slide_load_endpoint": {k: v for k, v in slide_endpoint.items() if k != "response"},
            "overlay_load_endpoint": {k: v for k, v in overlay_endpoint.items() if k != "response"},
            "property_names_endpoint": {k: v for k, v in property_names_endpoint.items() if k != "response"},
            "property_summary_endpoint": (
                {k: v for k, v in property_summary_endpoint.items() if k != "response"}
                if property_summary_endpoint is not None
                else None
            ),
            "selected_summary_property": summary_property,
            "properties_count": len(properties),
            "annotation_count": int(app.get_ann_layer(session_id, layer_name=args.layer_name).store.__len__()),
            "layer_metadata": layer_metadata,
            "tile_endpoints": [
                {k: v for k, v in endpoint.items() if k != "response"}
                for endpoint in tile_endpoints
            ],
            "first_visible_overlay_proxy": first_visible_overlay_proxy(
                property_names_endpoint,
                first_tile_endpoint,
            ),
        }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one or more benchmark cases and return structured JSON results."""

    if args.case:
        case_reports = []
        for name, slide_path, overlay_path in args.case:
            case_args = argparse.Namespace(**vars(args))
            case_args.case = []
            case_args.slide = Path(slide_path).expanduser().resolve()
            case_args.overlay = Path(overlay_path).expanduser().resolve()
            report = run_single_case(case_args)
            report["dataset"]["name"] = name
            case_reports.append(report)
        return {
            "benchmark": "viewer_responsiveness_backend",
            "schema_version": 2,
            "created_at": dt.datetime.now(UTC).isoformat(),
            "cases": case_reports,
        }

    args.slide = args.slide.expanduser().resolve()
    args.overlay = args.overlay.expanduser().resolve()
    return run_single_case(args)


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    msg = f"Object of type {value.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    output_text = json.dumps(result, indent=2, sort_keys=True, default=_json_default)
    if args.output:
        args.output.write_text(f"{output_text}\n", encoding="utf-8")
        print(args.output)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
