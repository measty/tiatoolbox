"""Simple Flask WSGI apps to display tiles as slippery maps."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
import secrets
import sys
import tempfile
import time
import urllib
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from flask import Flask, Response, jsonify, make_response, request, send_file
from flask.templating import render_template
from matplotlib import colormaps
from PIL import Image
from shapely.affinity import scale as scale_geometry
from shapely.geometry import Point, box

from tiatoolbox import data, logger
from tiatoolbox.annotation import AnnotationStore, SQLiteStore
from tiatoolbox.annotation.storage import Annotation
from tiatoolbox.tools.pyramid import AnnotationTileGenerator, ZoomifyGenerator
from tiatoolbox.utils.misc import store_from_dat
from tiatoolbox.utils.postproc_defs import MultichannelToRGB
from tiatoolbox.utils.visualization import AnnotationRenderer, colourise_image
from tiatoolbox.visualization.mvt import (
    DEFAULT_MVT_BUFFER,
    DEFAULT_MVT_EXTENT,
    encode_annotation_layer,
    encode_empty_annotation_layer,
)
from tiatoolbox.wsicore.wsireader import (
    OpenSlideWSIReader,
    TransformedWSIReader,
    VirtualWSIReader,
    WSIReader,
)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.colors import Colormap

    from tiatoolbox.wsicore import WSIMeta


SLIDE_EXTENSIONS = (
    ".svs",
    ".ndpi",
    ".tiff",
    ".mrxs",
    ".jpg",
    ".png",
    ".tif",
    ".qptiff",
    ".dcm",
)
RASTER_OVERLAY_EXTENSIONS = (
    ".jpg",
    ".png",
    ".tiff",
    ".svs",
    ".ndpi",
    ".mrxs",
    ".tif",
    ".npy",
    ".mha",
)
ANNOTATION_OVERLAY_EXTENSIONS = (
    ".db",
    ".dat",
    ".geojson",
)
PROJECT_OVERLAY_EXTENSIONS = (
    *RASTER_OVERLAY_EXTENSIONS,
    *ANNOTATION_OVERLAY_EXTENSIONS,
)
MVT_CACHE_MAX_AGE_SECONDS = 300
MVT_SIMPLIFICATION_PIXEL_TOLERANCE = 0.5
MVT_MIN_VISIBLE_GEOMETRY_PIXELS = 0.75
ANNOTATION_PERF_LOG_THRESHOLD_SECONDS = 0.1
ANNOTATION_MVT_OVERVIEW_REPRESENTATION = "overview"
ANNOTATION_MVT_FULL_REPRESENTATION = "full"
ANNOTATION_MVT_CENTROID_REPRESENTATION = "centroids"
ANNOTATION_MVT_LOW_ZOOM_POINT_MIN_FEATURES = 5000
ANNOTATION_MVT_FULL_GEOMETRY_MAX_DOWNSAMPLE = 8.0
ANNOTATION_MVT_OVERVIEW_GRID_PIXEL_SIZE = 24
ANNOTATION_MVT_OVERVIEW_GEOMETRY_MIN_SPAN_CELLS = 1.25
ANNOTATION_MVT_OVERVIEW_GEOMETRY_MIN_AREA_CELLS = 1.5
ANNOTATION_MVT_OVERVIEW_SIMPLIFICATION_PIXEL_TOLERANCE = 1.0
ANNOTATION_MVT_OVERVIEW_CENTROID_SAMPLE_FRACTION = 0.25
ANNOTATION_MVT_OVERVIEW_KIND_PROPERTY = "overview_kind"
ANNOTATION_MVT_OVERVIEW_KIND_DENSITY = "density"
ANNOTATION_MVT_OVERVIEW_KIND_GEOMETRY = "geometry"
ANNOTATION_TILE_DEFAULT_PROPERTY_FIELDS = ("type",)
ANNOTATION_MVT_CACHE_MAX_ENTRIES = 256
ANNOTATION_GEOJSON_AUTO_MAX_FEATURES = 2000
ANNOTATION_GEOJSON_DEBUG_TRUE_VALUES = {"1", "true", "yes", "on"}


class TileServer(Flask):
    """A Flask app to display Zoomify tiles as a slippery map.

    Args:
        title (str):
            The title of the tile server, displayed in the browser as
            the page title.
        layers (Dict[str, WSIReader | str] | List[WSIReader | str]):
            A dictionary mapping layer names to image paths, annotation paths,
            or :obj:`WSIReader` objects to display. The dictionary should have
            a 'slide' key which is the base slide for the visualization.
            May also be a list, in which case generic names 'slide', 'layer-1',
            'layer-2' etc. will be used. First entry in list will be assumed to
            be the base slide. If a layer is a single-channel low-res overlay,
            it will be colourized using the 'viridis' colourmap.

    Examples:
        >>> from tiatoolbox.wsicore.wsireader import WSIReader
        >>> from tiatoolbox.visualization.tileserver import TileServer
        >>> wsi = WSIReader.open("CMU-1.svs")
        >>> app = TileServer(
        ...     title="Testing TileServer",
        ...     layers={
        ...         "My SVS": wsi,
        ...     },
        ... )
        >>> app.run()

    """

    def __init__(  # noqa: PLR0915
        self: TileServer,
        title: str,
        layers: dict[str, WSIReader | str] | list[WSIReader | str],
        renderer: AnnotationRenderer | None = None,
        project_config: dict | None = None,
    ) -> None:
        """Initialize :class:`TileServer`."""
        super().__init__(
            __name__,
            template_folder=data._local_sample_path(  # noqa: SLF001
                Path("visualization") / "templates",
            ),
            static_url_path="",
            static_folder=data._local_sample_path(  # noqa: SLF001
                Path("visualization") / "static",
            ),
        )
        self.title = title
        self.layers = {}
        self.pyramids = {}
        self.renderer = renderer
        self.overlap = 0
        if renderer is None:  # pragma: no branch
            self.renderer = AnnotationRenderer(
                score_prop="type",
                thickness=-1,
                edge_thickness=1,
                zoomed_out_strat="scale",
                max_scale=8,
                blur_radius=0,
            )
        self.slide_mpps = {}
        self.renderers = {}
        self.overlaps = {}
        self.annotation_revisions = {}
        self.annotation_metadata_cache = {}
        self.annotation_mvt_cache = OrderedDict()
        self.annotation_prefilter_warnings: set[tuple[str, str, str]] = set()
        self.project_config = self._load_project_config(project_config)

        # Generic layer names if none provided.
        if isinstance(layers, list):
            layers_dict = {"slide": layers[0]}
            for i, p in enumerate(layers[1:]):
                layers_dict[f"layer-{i + 1}"] = p
            layers = layers_dict
        # Set up the layer dict.
        meta = None
        # if layers provided directly, not using with app,
        # so just set default session_id
        self.default_session_id = len(layers) > 0
        if self.default_session_id:
            self.layers["default"] = {}
            self.pyramids["default"] = {}
            self.renderers["default"] = copy.deepcopy(self.renderer)
            self.annotation_revisions["default"] = 0
        for i, (key, layer) in enumerate(layers.items()):
            layer = self._get_layer_as_wsireader(layer, meta)  # noqa: PLW2901

            self.layers["default"][key] = layer

            if isinstance(layer, WSIReader):
                self.pyramids["default"][key] = ZoomifyGenerator(layer)
            else:
                self.pyramids["default"][key] = layer  # it's an AnnotationTileGenerator
                if isinstance(layer.store, SQLiteStore):
                    self._warn_annotation_coarse_prefilter_unavailable(
                        "default",
                        key,
                        layer.store,
                    )

            if i == 0:
                meta = layer.info  # base slide info
                self.slide_mpps["default"] = meta.mpp

        self.route(
            "/tileserver/layer/<layer>/<session_id>/zoomify/TileGroup<int:tile_group>/"
            "<int:z>-<int:x>-<int:y>@<int:res>x.jpg",
        )(
            self.zoomify,
        )
        self.route("/")(self.index)
        self.route("/tileserver/session_id")(self.session_id)
        self.route("/tileserver/project", methods=["GET"])(self.get_project)
        self.route("/tileserver/project/overlays", methods=["GET"])(
            self.get_project_overlays,
        )
        self.route("/tileserver/layers", methods=["GET"])(self.get_layers)
        self.route("/tileserver/color_prop", methods=["PUT"])(self.change_prop)
        self.route("/tileserver/slide", methods=["PUT"])(self.change_slide)
        self.route("/tileserver/clear_overlays", methods=["PUT"])(self.clear_overlays)
        self.route("/tileserver/cmap", methods=["PUT"])(self.change_mapper)
        self.route(
            "/tileserver/annotations",
            methods=["PUT"],
        )(self.load_annotations)
        self.route("/tileserver/overlay", methods=["PUT"])(self.change_overlay)
        self.route("/tileserver/commit", methods=["POST"])(self.commit_db)
        self.route("/tileserver/renderer/<prop>", methods=["PUT"])(self.update_renderer)
        self.route("/tileserver/reset/<session_id>", methods=["PUT"])(self.reset)
        self.route("/tileserver/secondary_cmap", methods=["PUT"])(
            self.change_secondary_cmap,
        )
        self.route("/tileserver/prop_names/<ann_type>")(self.get_properties)
        self.route("/tileserver/prop_values/<prop>/<ann_type>")(
            self.get_property_values,
        )
        self.route("/tileserver/prop_summary/<prop>/<ann_type>")(
            self.get_property_summary,
        )
        self.route("/tileserver/color_prop", methods=["GET"])(self.get_color_prop)
        self.route("/tileserver/slide", methods=["GET"])(self.get_slide)
        self.route("/tileserver/cmap", methods=["GET"])(self.get_mapper)
        self.route("/tileserver/annotations", methods=["GET"])(self.get_annotations)
        self.route("/tileserver/annotations/geojson", methods=["GET"])(
            self.get_annotations_geojson,
        )
        self.route("/tileserver/annotations/detail", methods=["GET"])(
            self.get_annotation_details,
        )
        self.route(
            "/tileserver/layer/<layer>/<session_id>/mvt/<representation>/"
            "<int:z>/<int:x>/<int:y>.pbf",
            methods=["GET"],
        )(self.get_annotations_mvt)
        self.route(
            "/tileserver/layer/<layer>/<session_id>/mvt/<int:z>/<int:x>/<int:y>.pbf",
            defaults={"representation": ANNOTATION_MVT_FULL_REPRESENTATION},
            methods=["GET"],
        )(self.get_annotations_mvt)
        self.route("/tileserver/overlay", methods=["GET"])(self.get_overlay)
        self.route("/tileserver/renderer/<prop>", methods=["GET"])(self.get_renderer)
        self.route("/tileserver/secondary_cmap", methods=["GET"])(
            self.get_secondary_cmap,
        )
        self.route("/tileserver/tap_query/<x>/<y>")(self.tap_query)
        self.route("/tileserver/prop_range", methods=["PUT"])(self.prop_range)
        self.route("/tileserver/channels", methods=["GET"])(self.get_channels)
        self.route("/tileserver/channels", methods=["PUT"])(self.set_channels)
        self.route("/tileserver/enhance", methods=["PUT"])(self.set_enhance)
        self.route("/tileserver/shutdown", methods=["POST"])(self.shutdown)
        self.route("/tileserver/sessions", methods=["GET"])(self.sessions)
        self.route("/tileserver/healthcheck", methods=["GET"])(self.healthcheck)

    def _get_session_id(self: TileServer) -> str:
        """Get the session_id from the request.

        Returns:
            str: The session_id name.

        """
        if self.default_session_id:
            return "default"
        return request.cookies.get("session_id")

    @staticmethod
    def _log_annotation_perf(
        event: str,
        elapsed_seconds: float,
        **fields: object,
    ) -> None:
        """Log annotation performance measurements with low default noise."""
        field_text = " ".join(f"{key}={value}" for key, value in fields.items())
        message = "Annotation perf: event=%s elapsed_ms=%.1f %s"
        args = (event, elapsed_seconds * 1000, field_text)
        if elapsed_seconds >= ANNOTATION_PERF_LOG_THRESHOLD_SECONDS:
            logger.info(message, *args)
            return
        logger.debug(message, *args)

    @staticmethod
    def _discover_files(
        root: Path | None,
        extensions: tuple[str, ...],
    ) -> list[Path]:
        """Discover files recursively for a set of suffixes."""
        if root is None or not root.exists():
            return []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

    @staticmethod
    def _annotation_type_where_clause(ann_type: str) -> str | None:
        """Build a where clause for a specific annotation type."""
        if ann_type == "all":
            return None
        try:
            ann_value = json.loads(ann_type)
        except json.JSONDecodeError:
            try:
                ann_value = ast.literal_eval(ann_type)
            except (ValueError, SyntaxError):
                ann_value = ann_type
        return f'props["type"] == {json.dumps(ann_value)}'

    @staticmethod
    def _decode_optional_json(value: str | None) -> str | list | dict | None:
        """Decode JSON string values used by the frontend."""
        if value in (None, "", "null", "None"):
            return None
        return json.loads(value)

    def _load_project_config(self: TileServer, project_config: dict | None) -> dict:
        """Load visualization project configuration for the OpenLayers UI."""
        if project_config is None:
            return {}

        slide_folder = Path(project_config["slide_folder"]).expanduser().resolve()
        overlay_folder = Path(project_config["overlay_folder"]).expanduser().resolve()
        config = {
            "base_folder": str(slide_folder.parent),
            "slide_folder": str(slide_folder),
            "overlay_folder": str(overlay_folder),
            "auto_load": False,
            "default_cprop": "type",
            "color_dict": {},
            "initial_views": {},
        }

        config_files = sorted(overlay_folder.glob("*config.json"))
        if config_files:
            config.update(json.loads(config_files[0].read_text()))

        config.update(
            {
                key: value
                for key, value in project_config.items()
                if key not in {"slide_folder", "overlay_folder"}
            },
        )
        config["slide_folder"] = str(slide_folder)
        config["overlay_folder"] = str(overlay_folder)
        config["base_folder"] = str(slide_folder.parent)
        config["auto_load"] = bool(int(config["auto_load"])) if isinstance(
            config["auto_load"],
            str,
        ) else bool(config["auto_load"])
        return config

    def _get_project_slides(self: TileServer) -> list[dict[str, str]]:
        """Discover available project slides."""
        slide_root = self.project_config.get("slide_folder")
        if slide_root is None:
            return []
        slide_root = Path(slide_root)
        return [
            {
                "label": str(path.relative_to(slide_root)),
                "path": str(path),
                "stem": path.stem,
            }
            for path in self._discover_files(slide_root, SLIDE_EXTENSIONS)
        ]

    def _get_project_default_slide(self: TileServer, slides: list[dict]) -> str | None:
        """Resolve the default slide to load."""
        requested_slide = request.args.get("slide") or self.project_config.get(
            "first_slide",
        )
        if requested_slide is not None:
            for slide in slides:
                if requested_slide in {slide["label"], slide["path"]}:
                    return slide["path"]
        if slides:
            return slides[0]["path"]
        return None

    @staticmethod
    def _overlay_kind(overlay_path: Path) -> str:
        """Classify a project overlay for the OpenLayers UI."""
        if overlay_path.suffix.lower() in ANNOTATION_OVERLAY_EXTENSIONS:
            return "annotation"
        if overlay_path.suffix.lower() in {".npy", ".mha"}:
            return "transform"
        return "raster"

    def _get_project_overlays_for_slide(
        self: TileServer,
        slide_path: str,
    ) -> list[dict]:
        """Discover overlays associated with a slide path."""
        overlay_root = self.project_config.get("overlay_folder")
        if overlay_root is None:
            return []

        slide_path = Path(slide_path)
        overlay_root = Path(overlay_root)
        overlays = self._discover_files(overlay_root, PROJECT_OVERLAY_EXTENSIONS)
        matching = []
        for overlay in overlays:
            overlay_label = str(overlay.relative_to(overlay_root))
            if slide_path.stem not in overlay_label:
                continue
            matching.append(
                {
                    "label": overlay_label,
                    "path": str(overlay),
                    "kind": self._overlay_kind(overlay),
                },
            )
        return matching

    def _serialise_layer(
        self: TileServer,
        name: str,
        session_id: str,
    ) -> dict[str, object]:
        """Serialise layer metadata for the frontend."""
        layer = self.layers[session_id][name]
        pyramid = self.pyramids[session_id][name]
        metadata: dict[str, object] = {}
        if isinstance(pyramid, AnnotationTileGenerator):
            kind = "annotation"
            slide_dimensions = pyramid.info.slide_dimensions
            source_path = str(pyramid.store.path)
            vector_representations = self._get_annotation_vector_representations(
                name,
                session_id,
                pyramid,
            )
            geojson_policy = self._get_annotation_geojson_policy(pyramid)
            metadata = {
                "vector_format": "mvt",
                "vector_url": self._annotation_mvt_url(name, session_id),
                "vector_tile_extent": DEFAULT_MVT_EXTENT,
                "vector_tile_buffer": DEFAULT_MVT_BUFFER,
                "vector_tile_default_fields": list(
                    ANNOTATION_TILE_DEFAULT_PROPERTY_FIELDS,
                ),
                "detail_url": "/tileserver/annotations/detail",
                "vector_revision": int(self.annotation_revisions.get(session_id, 0)),
                "vector_representation_mode": "zoom",
                "default_vector_representation": ANNOTATION_MVT_FULL_REPRESENTATION,
                "vector_representations": vector_representations,
                "geojson_url": "/tileserver/annotations/geojson",
                "geojson_policy": geojson_policy,
                "coarse_prefilter": self._get_annotation_coarse_prefilter_status(
                    pyramid.store,
                ),
            }
        else:
            kind = "slide" if name == "slide" else "raster"
            slide_dimensions = layer.info.slide_dimensions
            source_path = str(layer.info.file_path)

        mpp = [1, 1] if getattr(layer.info, "mpp", None) is None else layer.info.mpp
        tile_revision = int(self.annotation_revisions.get(session_id, 0))
        return {
            "name": name,
            "kind": kind,
            "path": source_path,
            "url": f"/tileserver/layer/{urllib.parse.quote(name, safe='')}/"
            f"{session_id}/zoomify/"
            f"{{TileGroup}}/{{z}}-{{x}}-{{y}}@1x.jpg?rev={tile_revision}",
            "tile_revision": tile_revision,
            "size": [int(x) for x in slide_dimensions],
            "mpp": float(np.mean(mpp)),
            **metadata,
        }

    def _serialise_layers(self: TileServer, session_id: str | None) -> list[dict]:
        """Serialise all current layers for a session."""
        if session_id is None or session_id not in self.layers:
            return []
        return [
            self._serialise_layer(name, session_id)
            for name in self.layers[session_id]
        ]

    def _serialise_project(self: TileServer) -> dict:
        """Serialise project configuration for the OpenLayers frontend."""
        slides = self._get_project_slides()
        return {
            "slides": slides,
            "default_slide": self._get_project_default_slide(slides),
            "slide_folder": self.project_config.get("slide_folder"),
            "overlay_folder": self.project_config.get("overlay_folder"),
            "base_folder": self.project_config.get("base_folder"),
            "auto_load": self.project_config.get("auto_load", False),
            "default_cprop": self.project_config.get("default_cprop", "type"),
            "color_dict": self.project_config.get("color_dict", {}),
            "initial_views": self.project_config.get("initial_views", {}),
        }

    @staticmethod
    def _get_cmap(cmap: str | dict) -> Colormap:
        """Get the colourmap from the string sent."""
        if cmap == "None":
            return colormaps["jet"]
        if isinstance(cmap, str):
            return colormaps[cmap]

        def cmapp(x: Colormap) -> Colormap:
            """Dictionary colormap callable wrapper."""
            return cmap[x]

        return cmapp

    def _get_layer_as_wsireader(
        self: TileServer,
        layer: str | np.ndarray | WSIReader,
        meta: WSIMeta,
    ) -> WSIReader:
        """Gets appropriate image provider for layer.

        Args:
            layer (str | ndarray | WSIReader):
                A reference to an image or annotations to be displayed.
            meta (WSIMeta):
                The metadata of the base slide.

        Returns:
            WSIReader or AnnotationTileGenerator:
                The appropriate image source for the layer.

        """
        if isinstance(layer, (str, Path)):
            layer_path = Path(layer)
            if layer_path.suffix in [".jpg", ".png"]:
                # Assume it's a low-res heatmap.
                layer = np.array(Image.open(layer_path))
            elif layer_path.suffix == ".db":
                # Assume it's an annotation store.
                layer = AnnotationTileGenerator(
                    meta,
                    SQLiteStore(layer_path),
                    self.renderers["default"],
                    overlap=self.overlap,
                )
            elif layer_path.suffix == ".geojson":
                # Assume annotations in geojson format
                layer = AnnotationTileGenerator(
                    meta,
                    SQLiteStore.from_geojson(layer_path),
                    self.renderers["default"],
                    overlap=self.overlap,
                )
            else:
                # Assume it's a WSI.
                return WSIReader.open(layer_path)

        if isinstance(layer, np.ndarray):
            # Make into rgb if single channel.
            layer = colourise_image(layer)
            return VirtualWSIReader(layer, info=meta)

        if isinstance(layer, AnnotationStore):
            layer = AnnotationTileGenerator(
                meta,
                layer,
                self.renderers["default"],
                overlap=self.overlap,
            )

        return layer

    def zoomify(
        self: TileServer,
        layer: str,
        session_id: str,
        tile_group: int,  # skipcq: PYL-W0613  #noqa: ARG002
        z: int,
        x: int,
        y: int,
        res: int,
    ) -> Response:
        """Serve a Zoomify tile for a particular layer.

        Note that this should not be called directly, but will be called
        automatically by the Flask framework when a client requests a
        tile at the registered URL.

        Args:
            layer (str):
                The layer name.
            session_id (str):
                Session ID. Unique ID to disambiguate requests from different sessions.
            tile_group (int):
                The tile group. Currently unused.
            z (int):
                The zoom level.
            x (int):
                The x coordinate.
            y (int):
                The y coordinate.
            res (int):
                Resolution to save the tiles at.
                Helps to specify high resolution tiles. Valid options are 1 and 2.

        Returns:
            flask.Response:
                The tile image response.

        """
        try:
            pyramid = self.pyramids[session_id][layer]
            if isinstance(self.layers[session_id][layer], VirtualWSIReader):
                interpolation = "nearest"
                transparent_value = 0
            else:
                interpolation = "optimise"
                transparent_value = None
            if isinstance(pyramid, AnnotationTileGenerator):
                interpolation = None
        except KeyError:
            return Response("Layer not found", status=404)
        try:
            tile_image = pyramid.get_tile(
                level=z,
                x=x,
                y=y,
                res=res,
                interpolation=interpolation,
                transparent_value=transparent_value,
            )
        except IndexError:
            return Response("Tile not found", status=404)
        image_io = io.BytesIO()
        tile_image.save(image_io, format="webp")
        image_io.seek(0)
        return send_file(image_io, mimetype="image/webp")

    @staticmethod
    def update_types(sq: SQLiteStore) -> tuple:
        """Get the available types from the store."""
        types = sq.pquery("props['type']")
        types = [t for t in types if t is not None]
        return tuple(types)

    @staticmethod
    def decode_safe_name(name: str) -> Path:
        """Decode a URL-safe name."""
        return Path(urllib.parse.unquote(name).replace("\\", os.sep))

    @staticmethod
    def decode_layer_name(name: str) -> str:
        """Decode a URL-safe layer name."""
        return urllib.parse.unquote(name)

    @staticmethod
    def _get_annotation_tile_bounds(
        pyramid: AnnotationTileGenerator,
        z: int,
        x: int,
        y: int,
    ) -> tuple[float, float, float, float]:
        """Return tile bounds in baseline slide coordinates."""
        width, height = pyramid.info.slide_dimensions
        scale = pyramid.level_downsample(z)
        tile_width = pyramid.tile_size * scale
        min_x = x * tile_width
        min_y = y * tile_width
        max_x = min(width, min_x + tile_width)
        max_y = min(height, min_y + tile_width)
        return (min_x, min_y, max_x, max_y)

    @staticmethod
    def _get_annotation_mvt_hints(
        pyramid: AnnotationTileGenerator,
        z: int,
    ) -> dict[str, float]:
        """Return scale-aware simplification thresholds for annotation MVT tiles."""
        slide_units_per_pixel = float(max(1, pyramid.level_downsample(z)))
        min_geometry_size = slide_units_per_pixel * MVT_MIN_VISIBLE_GEOMETRY_PIXELS
        return {
            "slide_units_per_pixel": slide_units_per_pixel,
            "simplify_tolerance": 0.0
            if slide_units_per_pixel <= 1
            else slide_units_per_pixel * MVT_SIMPLIFICATION_PIXEL_TOLERANCE,
            "min_line_length": min_geometry_size,
            "min_polygon_area": min_geometry_size**2,
        }

    @staticmethod
    def _get_annotation_overview_hints(
        slide_units_per_pixel: float,
    ) -> dict[str, float]:
        """Return thresholds for the hybrid overview representation."""
        cell_size = max(
            slide_units_per_pixel * ANNOTATION_MVT_OVERVIEW_GRID_PIXEL_SIZE,
            1.0,
        )
        return {
            "cell_size": cell_size,
            "min_bbox_size": (
                cell_size * ANNOTATION_MVT_OVERVIEW_GEOMETRY_MIN_SPAN_CELLS
            ),
            "min_polygon_area": (
                (cell_size**2) * ANNOTATION_MVT_OVERVIEW_GEOMETRY_MIN_AREA_CELLS
            ),
            "simplify_tolerance": (
                slide_units_per_pixel
                * ANNOTATION_MVT_OVERVIEW_SIMPLIFICATION_PIXEL_TOLERANCE
            ),
        }

    @staticmethod
    def _annotation_mvt_url(
        name: str,
        session_id: str,
        representation: str | None = None,
    ) -> str:
        """Build an annotation MVT URL template for a layer."""
        prefix = (
            f"/tileserver/layer/{urllib.parse.quote(name, safe='')}/{session_id}/mvt"
        )
        if representation is None:
            return f"{prefix}/{{z}}/{{x}}/{{y}}.pbf"
        return f"{prefix}/{representation}/{{z}}/{{x}}/{{y}}.pbf"

    @staticmethod
    def _get_annotation_min_zoom_for_downsample(
        pyramid: AnnotationTileGenerator,
        max_downsample: float,
    ) -> int:
        """Return the first zoom level whose downsample meets a threshold."""
        for zoom in range(pyramid.level_count):
            if pyramid.level_downsample(zoom) <= max_downsample:
                return zoom
        return pyramid.level_count - 1

    @staticmethod
    def _get_annotation_full_geometry_min_zoom(
        pyramid: AnnotationTileGenerator,
    ) -> int:
        """Return the first zoom level where full geometry should be used."""
        return TileServer._get_annotation_min_zoom_for_downsample(
            pyramid,
            ANNOTATION_MVT_FULL_GEOMETRY_MAX_DOWNSAMPLE,
        )

    @staticmethod
    def _aggregate_annotation_overview(
        annotations: dict[str, Annotation],
        tile_bounds: tuple[int, int, int, int],
        slide_units_per_pixel: float,
        color_property: str | None,
        excluded_keys: set[str] | None = None,
        *,
        sample_fraction: float = 1.0,
    ) -> dict[str, Annotation]:
        """Aggregate centroid points into a coarse grid for overview rendering."""
        if not 0 < sample_fraction <= 1:
            msg = "sample_fraction must be in the interval (0, 1]."
            raise ValueError(msg)

        overview_hints = TileServer._get_annotation_overview_hints(
            slide_units_per_pixel,
        )
        cell_size = overview_hints["cell_size"]
        min_x, min_y, max_x, max_y = tile_bounds
        max_cell_x = max(int(np.ceil((max_x - min_x) / cell_size)) - 1, 0)
        max_cell_y = max(int(np.ceil((max_y - min_y) / cell_size)) - 1, 0)
        excluded_keys = excluded_keys or set()
        sample_weight = 1.0 / sample_fraction

        overview_cells: dict[
            tuple[int, int],
            dict[str, object],
        ] = {}
        for key, annotation in annotations.items():
            if key in excluded_keys:
                continue
            centroid = annotation.geometry
            cell_x = int(np.floor((centroid.x - min_x) / cell_size))
            cell_y = int(np.floor((centroid.y - min_y) / cell_size))
            cell_x = min(max(cell_x, 0), max_cell_x)
            cell_y = min(max(cell_y, 0), max_cell_y)
            cell_key = (cell_x, cell_y)
            cell = overview_cells.setdefault(
                cell_key,
                {
                    "count": 0.0,
                    "properties": defaultdict(float),
                    "property_values": {},
                },
            )
            cell["count"] = float(cell["count"]) + sample_weight
            if color_property:
                value = annotation.properties.get(color_property)
                if value is not None:
                    value_key = json.dumps(value, sort_keys=True, default=str)
                    cell["properties"][value_key] += sample_weight
                    cell["property_values"][value_key] = value

        overview_annotations: dict[str, Annotation] = {}
        for cell_x, cell_y in sorted(overview_cells):
            cell = overview_cells[(cell_x, cell_y)]
            cell_min_x = min_x + (cell_x * cell_size)
            cell_min_y = min_y + (cell_y * cell_size)
            cell_max_x = min(max_x, cell_min_x + cell_size)
            cell_max_y = min(max_y, cell_min_y + cell_size)
            properties = {
                "count": round(float(cell["count"])),
                ANNOTATION_MVT_OVERVIEW_KIND_PROPERTY: (
                    ANNOTATION_MVT_OVERVIEW_KIND_DENSITY
                ),
            }
            if color_property and cell["properties"]:
                dominant_value_key = max(
                    cell["properties"].items(),
                    key=lambda item: item[1],
                )[0]
                properties[color_property] = cell["property_values"][
                    dominant_value_key
                ]
            overview_annotations[f"{cell_x}:{cell_y}"] = Annotation(
                box(cell_min_x, cell_min_y, cell_max_x, cell_max_y),
                properties,
            )
        return overview_annotations

    @staticmethod
    def _annotation_is_prominent_overview_geometry(
        annotation: Annotation,
        overview_hints: dict[str, float],
    ) -> bool:
        """Return whether an annotation should keep real geometry in overview."""
        geometry = annotation.geometry
        if geometry.is_empty or geometry.geom_type in {"Point", "MultiPoint"}:
            return False

        min_bbox_size = float(overview_hints["min_bbox_size"])
        min_polygon_area = float(overview_hints["min_polygon_area"])
        bounds = geometry.bounds
        max_span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        return geometry.area >= min_polygon_area or max_span >= min_bbox_size

    @staticmethod
    def _project_annotation_overview_geometry(
        annotations: dict[str, Annotation],
        overview_hints: dict[str, float],
        color_property: str | None = None,
        requested_fields: tuple[str, ...] = (),
    ) -> dict[str, Annotation]:
        """Keep visibly large annotations as simplified overview geometry."""
        overview_geometry: dict[str, Annotation] = {}
        for key in sorted(annotations):
            annotation = annotations[key]
            if not TileServer._annotation_is_prominent_overview_geometry(
                annotation,
                overview_hints,
            ):
                continue

            properties = TileServer._project_annotation_tile_properties(
                key,
                annotation.properties,
                color_property=color_property,
                requested_fields=requested_fields,
            )
            properties[ANNOTATION_MVT_OVERVIEW_KIND_PROPERTY] = (
                ANNOTATION_MVT_OVERVIEW_KIND_GEOMETRY
            )
            overview_geometry[key] = Annotation(annotation.geometry, properties)

        return overview_geometry

    def _get_annotation_vector_representations(
        self: TileServer,
        name: str,
        session_id: str,
        pyramid: AnnotationTileGenerator,
    ) -> list[dict[str, str | int]]:
        """Describe zoom-aware vector representations for an annotation overlay."""
        full_representation: dict[str, str | int] = {
            "id": ANNOTATION_MVT_FULL_REPRESENTATION,
            "geometry_type": "mixed",
            "min_zoom": 0,
            "vector_format": "mvt",
            "vector_url": self._annotation_mvt_url(name, session_id),
        }

        if not isinstance(pyramid.store, SQLiteStore):
            return [full_representation]

        if len(pyramid.store) < ANNOTATION_MVT_LOW_ZOOM_POINT_MIN_FEATURES:
            return [full_representation]

        full_geometry_min_zoom = self._get_annotation_full_geometry_min_zoom(pyramid)
        if full_geometry_min_zoom <= 0:
            return [full_representation]

        representations: list[dict[str, str | int]] = []
        if full_geometry_min_zoom > 0:
            representations.append(
                {
                    "id": ANNOTATION_MVT_OVERVIEW_REPRESENTATION,
                    "geometry_type": "mixed",
                    "min_zoom": 0,
                    "max_zoom": full_geometry_min_zoom - 1,
                    "vector_format": "mvt",
                    "vector_url": self._annotation_mvt_url(
                        name,
                        session_id,
                        ANNOTATION_MVT_OVERVIEW_REPRESENTATION,
                    ),
                },
            )

        full_representation["min_zoom"] = full_geometry_min_zoom
        return [*representations, full_representation]

    @staticmethod
    def _get_annotation_geojson_policy(
        pyramid: AnnotationTileGenerator,
    ) -> dict[str, object]:
        """Describe when the GeoJSON endpoint is safe for an annotation overlay."""
        if not isinstance(pyramid.store, SQLiteStore):
            return {
                "allowed": True,
                "debug_only": False,
                "reason": None,
                "auto_feature_limit": ANNOTATION_GEOJSON_AUTO_MAX_FEATURES,
            }

        feature_count = len(pyramid.store)
        if feature_count <= ANNOTATION_GEOJSON_AUTO_MAX_FEATURES:
            return {
                "allowed": True,
                "debug_only": False,
                "reason": None,
                "feature_count": feature_count,
                "auto_feature_limit": ANNOTATION_GEOJSON_AUTO_MAX_FEATURES,
            }

        return {
            "allowed": False,
            "debug_only": True,
            "reason": "large_sqlite_overlay_requires_mvt",
            "message": (
                "Large SQLite-backed overlays stay on the MVT path during normal "
                "viewing. GeoJSON is limited to small overlays or explicit debug "
                "requests with bounds."
            ),
            "feature_count": feature_count,
            "auto_feature_limit": ANNOTATION_GEOJSON_AUTO_MAX_FEATURES,
        }

    @staticmethod
    def _get_annotation_coarse_prefilter_status(
        store: AnnotationStore,
    ) -> dict[str, object]:
        """Describe whether coarse-scale polygon prefiltering is available."""
        if not isinstance(store, SQLiteStore):
            return {
                "available": False,
                "reason": "store_not_sqlite",
                "message": None,
                "prepare_hint": None,
                "store_path": None,
            }

        if store.has_area_column():
            return {
                "available": True,
                "reason": None,
                "message": None,
                "prepare_hint": None,
                "store_path": str(store.path),
            }

        if store.path.is_file():
            prepare_hint = (
                "from tiatoolbox.annotation.storage import SQLiteStore; "
                f"store = SQLiteStore({str(store.path)!r}); "
                "store.ensure_area_column(); store.close()"
            )
        else:
            prepare_hint = None

        return {
            "available": False,
            "reason": "missing_area_column",
            "message": (
                "This SQLite overlay is missing the optional 'area' column used "
                "for coarse-scale polygon prefiltering. Low-zoom polygon tiles "
                "will use a slower fallback path until the store is prepared."
            ),
            "prepare_hint": prepare_hint,
            "store_path": str(store.path),
        }

    def _warn_annotation_coarse_prefilter_unavailable(
        self: TileServer,
        session_id: str,
        layer_name: str,
        store: SQLiteStore,
    ) -> None:
        """Log a one-time warning for legacy stores missing area support."""
        status = self._get_annotation_coarse_prefilter_status(store)
        if status.get("reason") != "missing_area_column":
            return

        warning_key = (session_id, layer_name, str(store.path))
        if warning_key in self.annotation_prefilter_warnings:
            return
        self.annotation_prefilter_warnings.add(warning_key)

        prepare_hint = status.get("prepare_hint")
        if isinstance(prepare_hint, str) and prepare_hint:
            prepare_text = f" Prepare it once with: {prepare_hint}"
        else:
            prepare_text = (
                " Reopen the store from disk and call "
                "`SQLiteStore.ensure_area_column()` before interactive viewing."
            )

        logger.warning(
            "Annotation overlay %s (%s) is missing the 'area' column used for "
            "coarse-scale polygon prefiltering. Low-zoom polygon tiles will use "
            "a slower fallback path until the store is prepared.%s",
            layer_name,
            store.path,
            prepare_text,
        )

    @staticmethod
    def _annotation_geojson_debug_requested() -> bool:
        """Return whether the caller explicitly requested GeoJSON debug mode."""
        return (
            str(request.args.get("debug", "")).strip().lower()
            in ANNOTATION_GEOJSON_DEBUG_TRUE_VALUES
        )

    @staticmethod
    def _query_annotations_for_mvt(
        ann_layer: AnnotationTileGenerator,
        tile_bounds: tuple[int, int, int, int],
        where: object,
        render_hints: dict[str, float],
        representation: str,
        color_property: str | None = None,
        requested_fields: tuple[str, ...] = (),
    ) -> tuple[list[tuple[object, dict[str, object]]], bool]:
        """Query annotation candidates for an MVT tile representation."""
        projected_fields = TileServer._annotation_tile_projection_fields(
            color_property=color_property,
            requested_fields=requested_fields,
        )
        if representation in {
            ANNOTATION_MVT_OVERVIEW_REPRESENTATION,
            ANNOTATION_MVT_CENTROID_REPRESENTATION,
        }:
            if not isinstance(ann_layer.store, SQLiteStore):
                msg = "Representation not found"
                raise LookupError(msg)
            if representation == ANNOTATION_MVT_CENTROID_REPRESENTATION:
                return (
                    ann_layer.store.query_mvt_records(
                        geometry=tile_bounds,
                        where=where,
                        geometry_predicate="bbox_intersects",
                        property_fields=projected_fields,
                        centroids=True,
                    ),
                    False,
                )
            overview_hints = TileServer._get_annotation_overview_hints(
                float(render_hints["slide_units_per_pixel"]),
            )
            try:
                overview_geometries = ann_layer.store.query_renderable_geometries(
                    geometry=tile_bounds,
                    where=where,
                    geometry_predicate="bbox_intersects",
                    min_area=overview_hints["min_polygon_area"],
                    min_bbox_size=overview_hints["min_bbox_size"],
                )
            except ValueError:
                overview_geometries = ann_layer.store.query(
                    geometry=tile_bounds,
                    where=where,
                    geometry_predicate="bbox_intersects",
                    order_by_area=False,
                )

            projected_overview_geometries = (
                TileServer._project_annotation_overview_geometry(
                    overview_geometries,
                    overview_hints,
                    color_property=color_property,
                    requested_fields=requested_fields,
                )
            )
            centroid_sample_fraction = (
                ANNOTATION_MVT_OVERVIEW_CENTROID_SAMPLE_FRACTION
                if len(ann_layer.store) >= ANNOTATION_MVT_LOW_ZOOM_POINT_MIN_FEATURES
                else 1.0
            )
            annotations = ann_layer.store.query_centroids(
                geometry=tile_bounds,
                where=where,
                geometry_predicate="bbox_intersects",
                sample_fraction=centroid_sample_fraction,
            )
            annotations = TileServer._aggregate_annotation_overview(
                annotations,
                tile_bounds,
                float(render_hints["slide_units_per_pixel"]),
                color_property,
                excluded_keys=set(projected_overview_geometries),
                sample_fraction=centroid_sample_fraction,
            )
            annotations = {**annotations, **projected_overview_geometries}
            return (
                [(ann.geometry, ann.properties) for ann in annotations.values()],
                False,
            )

        prefiltered = isinstance(ann_layer.store, SQLiteStore)
        if isinstance(ann_layer.store, SQLiteStore):
            try:
                return (
                    ann_layer.store.query_mvt_records(
                        geometry=tile_bounds,
                        where=where,
                        geometry_predicate="bbox_intersects",
                        property_fields=projected_fields,
                        min_area=render_hints["min_polygon_area"],
                        min_bbox_size=render_hints["min_line_length"],
                    ),
                    prefiltered,
                )
            except ValueError as exc:
                if "without an area column" not in str(exc):
                    raise
                prefiltered = False

        annotations = ann_layer.store.query(
            geometry=tile_bounds,
            where=where,
            geometry_predicate="bbox_intersects",
            order_by_area=False,
        )
        return (
            [
                (
                    ann.geometry,
                    TileServer._project_annotation_tile_properties(
                        key,
                        ann.properties,
                        color_property=color_property,
                        requested_fields=requested_fields,
                    ),
                )
                for key, ann in annotations.items()
            ],
            prefiltered,
        )

    @staticmethod
    def _get_annotation_mvt_encode_kwargs(
        render_hints: dict[str, float],
        representation: str,
    ) -> dict[str, float]:
        """Return encoding kwargs for a specific annotation representation."""
        if representation == ANNOTATION_MVT_OVERVIEW_REPRESENTATION:
            return {
                "simplify_tolerance": (
                    render_hints["slide_units_per_pixel"]
                    * ANNOTATION_MVT_OVERVIEW_SIMPLIFICATION_PIXEL_TOLERANCE
                ),
            }
        if representation != ANNOTATION_MVT_FULL_REPRESENTATION:
            return {}
        return {
            "simplify_tolerance": render_hints["simplify_tolerance"],
            "min_line_length": render_hints["min_line_length"],
            "min_polygon_area": render_hints["min_polygon_area"],
        }

    @staticmethod
    def _parse_annotation_tile_requested_fields(value: str | None) -> tuple[str, ...]:
        """Parse optional extra tile property names from a request argument."""
        if value is None or value == "":
            return ()

        raw_fields: list[object]
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None

        if isinstance(decoded, list):
            raw_fields = decoded
        elif isinstance(decoded, str):
            raw_fields = decoded.split(",")
        else:
            raw_fields = value.split(",")

        fields: list[str] = []
        seen: set[str] = set()
        for item in raw_fields:
            if not isinstance(item, str):
                continue
            field = item.strip()
            if not field or field in seen:
                continue
            fields.append(field)
            seen.add(field)
        return tuple(fields)

    @staticmethod
    def _annotation_tile_projection_fields(
        color_property: str | None = None,
        requested_fields: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return the ordered top-level annotation fields needed in-tile."""
        fields: list[str] = []
        seen: set[str] = set()
        for field in (
            *ANNOTATION_TILE_DEFAULT_PROPERTY_FIELDS,
            color_property,
            *requested_fields,
        ):
            if not field or field == "id" or field in seen:
                continue
            fields.append(field)
            seen.add(field)
        return tuple(fields)

    @staticmethod
    def _project_annotation_tile_properties(
        key: str,
        properties: dict,
        color_property: str | None = None,
        requested_fields: tuple[str, ...] = (),
    ) -> dict[str, object]:
        """Project annotation properties down to the in-tile minimum."""
        projected: dict[str, object] = {"id": key}
        for field in TileServer._annotation_tile_projection_fields(
            color_property=color_property,
            requested_fields=requested_fields,
        ):
            if field not in properties:
                continue
            projected[field] = properties[field]
        return projected

    def _bump_annotation_revision(self: TileServer, session_id: str) -> None:
        """Invalidate cached tile URLs and annotation metadata for a session."""
        self.annotation_revisions[session_id] = (
            int(self.annotation_revisions.get(session_id, 0)) + 1
        )
        self.annotation_mvt_cache = OrderedDict(
            (key, value)
            for key, value in self.annotation_mvt_cache.items()
            if key[0] != session_id
        )
        self.annotation_metadata_cache = {
            key: value
            for key, value in self.annotation_metadata_cache.items()
            if key[0] != session_id
        }

    def _annotation_metadata_cache_key(
        self: TileServer,
        session_id: str,
        layer_name: str | None,
        event: str,
        *parts: object,
    ) -> tuple[object, ...]:
        """Return a cache key for annotation metadata derived from a store revision."""
        revision = int(self.annotation_revisions.get(session_id, 0))
        tokens = tuple(repr(part) for part in parts)
        return (session_id, layer_name, revision, event, *tokens)

    @staticmethod
    def _annotation_cache_token(value: object) -> str:
        """Return a stable digest for request arguments participating in caching."""
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            encoded = repr(value)
        return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()

    def _annotation_mvt_cache_key(
        self: TileServer,
        session_id: str,
        layer_name: str,
        representation: str,
        z: int,
        x: int,
        y: int,
        where: object,
        color_property: str | None,
        requested_fields: tuple[str, ...],
    ) -> tuple[object, ...]:
        """Return the cache key for an annotation MVT payload."""
        revision = int(self.annotation_revisions.get(session_id, 0))
        return (
            session_id,
            layer_name,
            revision,
            representation,
            self._annotation_cache_token(where),
            color_property,
            requested_fields,
            z,
            x,
            y,
        )

    def _get_annotation_mvt_cache_entry(
        self: TileServer,
        cache_key: tuple[object, ...],
    ) -> dict[str, object] | None:
        """Return a cached annotation MVT payload and refresh its LRU position."""
        cached = self.annotation_mvt_cache.get(cache_key)
        if cached is not None:
            self.annotation_mvt_cache.move_to_end(cache_key)
        return cached

    def _set_annotation_mvt_cache_entry(
        self: TileServer,
        cache_key: tuple[object, ...],
        payload: bytes,
        etag: str,
        *,
        candidates: int,
        prefiltered: bool,
        output_features: int,
    ) -> None:
        """Store an annotation MVT payload in the bounded in-memory LRU cache."""
        self.annotation_mvt_cache[cache_key] = {
            "payload": payload,
            "etag": etag,
            "candidates": candidates,
            "prefiltered": prefiltered,
            "output_features": output_features,
        }
        self.annotation_mvt_cache.move_to_end(cache_key)
        while len(self.annotation_mvt_cache) > ANNOTATION_MVT_CACHE_MAX_ENTRIES:
            self.annotation_mvt_cache.popitem(last=False)

    @staticmethod
    def _make_annotation_mvt_response(payload: bytes, etag: str) -> Response:
        """Build an annotation MVT response with standard cache headers."""
        response = Response(payload, mimetype="application/vnd.mapbox-vector-tile")
        response.headers["Cache-Control"] = (
            f"private, max-age={MVT_CACHE_MAX_AGE_SECONDS}"
        )
        response.set_etag(etag)
        response.make_conditional(request)
        return response

    def get_ann_layer(
        self: TileServer,
        session_id: str,
        layer_name: str | None = None,
    ) -> AnnotationTileGenerator | ValueError:
        """Get the annotation layer for a session_id."""
        if layer_name is not None:
            layer = self.pyramids[session_id].get(layer_name)
            if isinstance(layer, AnnotationTileGenerator):
                return layer
        for name, layer in self.pyramids[session_id].items():
            if isinstance(layer, AnnotationTileGenerator) and (
                layer_name is None or name == layer_name
            ):
                return layer
        msg = "No annotation layer found."
        raise ValueError(msg)

    def index(self: TileServer) -> str:
        """Serve the index page.

        Returns:
            flask.Response:
                The index page.

        """
        frontend_config = {
            "default_session_id": self.default_session_id,
            "initial_layers": self._serialise_layers(self._get_session_id()),
            "project": self._serialise_project(),
        }

        return render_template(
            "index.html",
            title=self.title,
            frontend_config=json.dumps(frontend_config),
        )

    def change_prop(self: TileServer) -> str:
        """Change the property to colour annotations by."""
        prop = request.form["prop"]
        session_id = self._get_session_id()
        self.renderers[session_id].score_prop = json.loads(prop)

        return "done"

    def session_id(self: TileServer) -> Response:
        """Set up a new session."""
        # respond with a random cookie to disambiguate sessions
        resp = make_response("done")
        session_id = "default" if self.default_session_id else secrets.token_urlsafe(16)
        resp.set_cookie("session_id", session_id, httponly=True)  # skipcq: PTC-W6003
        self.renderers.setdefault(session_id, copy.deepcopy(self.renderer))
        self.overlaps.setdefault(session_id, 0)
        self.layers.setdefault(session_id, {})
        self.pyramids.setdefault(session_id, {})
        self.annotation_revisions.setdefault(session_id, 0)
        return resp

    def get_project(self: TileServer) -> Response:
        """Get project discovery metadata for the OpenLayers frontend."""
        return jsonify(self._serialise_project())

    def get_project_overlays(self: TileServer) -> Response:
        """Get overlays associated with a project slide."""
        slide_path = request.args.get("slide_path")
        if slide_path is None:
            return jsonify([])
        return jsonify(self._get_project_overlays_for_slide(slide_path))

    def get_layers(self: TileServer) -> Response:
        """Get current layer metadata for the active session."""
        return jsonify(self._serialise_layers(self._get_session_id()))

    def reset(self: TileServer, session_id: str) -> str:
        """Reset the tileserver."""
        del self.layers[session_id]
        del self.pyramids[session_id]
        del self.slide_mpps[session_id]
        del self.renderers[session_id]
        del self.overlaps[session_id]
        self.annotation_revisions.pop(session_id, None)
        self.annotation_mvt_cache = OrderedDict(
            (key, value)
            for key, value in self.annotation_mvt_cache.items()
            if key[0] != session_id
        )
        self.annotation_metadata_cache = {
            key: value
            for key, value in self.annotation_metadata_cache.items()
            if key[0] != session_id
        }
        self.annotation_prefilter_warnings = {
            key for key in self.annotation_prefilter_warnings if key[0] != session_id
        }
        return "done"

    def change_slide(self: TileServer) -> str:
        """Change the slide."""
        session_id = self._get_session_id()
        slide_path = request.form["slide_path"]
        slide_path = self.decode_safe_name(slide_path)

        self.layers[session_id] = {"slide": WSIReader.open(Path(slide_path))}
        self.pyramids[session_id] = {
            "slide": ZoomifyGenerator(self.layers[session_id]["slide"], tile_size=256),
        }
        if self.layers[session_id]["slide"].info.mpp is None:
            self.layers[session_id]["slide"].info.mpp = [1, 1]
        self.slide_mpps[session_id] = self.layers[session_id]["slide"].info.mpp
        self._bump_annotation_revision(session_id)

        return "done"

    def clear_overlays(self: TileServer) -> str:
        """Clear all overlays."""
        session_id = self._get_session_id()
        slide_layer = self.layers[session_id]["slide"]
        self.layers[session_id] = {"slide": slide_layer}
        self.pyramids[session_id] = {
            "slide": ZoomifyGenerator(slide_layer, tile_size=256),
        }
        self._bump_annotation_revision(session_id)
        return "done"

    def change_mapper(self: TileServer) -> str:
        """Change the colour mapper for the overlay."""
        session_id = self._get_session_id()
        cmap = json.loads(request.form["cmap"])
        if isinstance(cmap, dict):
            cmap = dict(zip(cmap["keys"], cmap["values"], strict=False))
            self.renderers[session_id].score_fn = lambda x: x
        self.renderers[session_id].mapper = cmap
        self.renderers[session_id].function_mapper = None

        return "done"

    def change_secondary_cmap(self: TileServer) -> str:
        """Change the type-specific colour mapper for the overlay."""
        session_id = self._get_session_id()
        cmap = json.loads(request.form["cmap"])
        type_id = request.form["type_id"]
        prop = request.form["prop"]
        cmapp = self._get_cmap(cmap)

        cmap_dict = {"type": json.loads(type_id), "score_prop": prop, "mapper": cmapp}
        self.renderers[session_id].secondary_cmap = cmap_dict

        return "done"

    def update_renderer(self: TileServer, prop: str) -> str:
        """Update a property in the renderer.

        Args:
            prop (str): The property to update.

        """
        session_id = self._get_session_id()
        val = request.form["val"]
        val = json.loads(val)
        if val in ["None", "null"]:
            val = None
        self.renderers[session_id].__setattr__(prop, val)
        if prop == "blur_radius":
            self.overlaps[session_id] = int(1.5 * val)
            self.get_ann_layer(session_id).overlap = self.overlaps[session_id]
        return "done"

    def load_annotations(self: TileServer) -> str:
        """Load annotations from a dat file.

        Adds to an existing store if one is already present,
        otherwise creates a new store.

        Returns:
            str: A jsonified list of types.

        """
        session_id = self._get_session_id()
        file_path = request.form["file_path"]
        file_path = self.decode_safe_name(file_path)

        for layer in self.pyramids[session_id].values():
            if isinstance(layer, AnnotationTileGenerator):
                to_add = SQLiteStore(file_path)
                layer.store.append_many(list(to_add.values()))
                to_add.close()
                self._bump_annotation_revision(session_id)
                types = self.update_types(layer.store)
                return json.dumps(types)

        sq = SQLiteStore(file_path)

        self.pyramids[session_id]["overlay"] = AnnotationTileGenerator(
            self.layers[session_id]["slide"].info,
            sq,
            self.renderers[session_id],
            overlap=self.overlaps[session_id],
        )
        self.layers[session_id]["overlay"] = self.pyramids[session_id]["overlay"]
        self._bump_annotation_revision(session_id)
        types = self.update_types(sq)
        return json.dumps(types)

    def change_overlay(self: TileServer) -> str:
        """Change the overlay.

        If the path points to some annotations, the current overlay
        is replaced with the new one. If the path points to an image,
        it is added as a new layer.

        Returns:
            str: A jsonified list of types.

        """
        session_id = self._get_session_id()
        overlay_path = request.form["overlay_path"]
        overlay_path = self.decode_safe_name(overlay_path)

        # Get other session id
        session_ids = list(self.layers.keys())
        session_ids.remove(session_id)

        # Get the first remaining session_id (if any exist)
        other_session_id = session_ids[0] if session_ids else None

        if overlay_path.suffix in [".npy", ".mha"]:
            return self._handle_registration_overlay(
                session_id, overlay_path, other_session_id
            )

        if overlay_path.suffix in [".jpg", ".png", ".tiff", ".svs", ".ndpi", ".mrxs"]:
            return self._add_image_overlay(session_id, overlay_path)

        return self._add_annotation_overlay(session_id, overlay_path)

    def _handle_registration_overlay(
        self,
        session_id: str,
        overlay_path: Path,
        other_session_id: str | None,
    ) -> str:
        def _apply_transform(source_fp: str, target_fp: str) -> None:
            # loading a registration transformation
            self.layers[session_id]["slide"] = TransformedWSIReader(
                source_fp,
                target_img=target_fp,
                transform=overlay_path,
            )
            self.pyramids[session_id]["slide"] = ZoomifyGenerator(
                self.layers[session_id]["slide"]
            )
            self._bump_annotation_revision(session_id)

        if other_session_id is not None:
            logger.warning("Using slide in other window as target slide.")
            _apply_transform(
                self.layers[session_id]["slide"].info.file_path,
                self.layers[other_session_id]["slide"].info.file_path,
            )
            return json.dumps("slide")

        layer_keys = [
            k for k in self.layers[session_id] if k not in ["slide", "overlay"]
        ]
        layer_keys.reverse()  # try newest first
        for key in layer_keys:
            target_fp = self.layers[session_id][key].info.file_path
            if Path(target_fp).suffix in [".tiff", ".svs", ".ndpi", ".mrxs"]:
                logger.warning(
                    "Using slide as source and last overlay as target for registration."
                )
                _apply_transform(
                    self.layers[session_id]["slide"].info.file_path,
                    target_fp,
                )
                return json.dumps("slide")

        logger.warning(
            "No suitable overlay found. Using current slide as target. "
            "This may display incorrectly if dimensions differ."
        )
        _apply_transform(
            self.layers[session_id]["slide"].info.file_path,
            self.layers[session_id]["slide"].info.file_path,
        )
        return json.dumps("slide")

    def _add_image_overlay(self, session_id: str, overlay_path: Path) -> str:
        layer = overlay_path.stem
        if layer in self.layers[session_id]:
            # use full file name to disambiguate
            layer = overlay_path.name
        if overlay_path.suffix == ".tiff":
            self.layers[session_id][layer] = OpenSlideWSIReader(
                overlay_path,
                mpp=self.layers[session_id]["slide"].info.mpp[0],
            )
        elif overlay_path.suffix in [".jpg", ".png"]:
            info = self.layers[session_id]["slide"].info
            info.file_path = str(overlay_path)
            self.layers[session_id][layer] = VirtualWSIReader(
                overlay_path,
                info=info,
            )
        else:
            self.layers[session_id][layer] = WSIReader.open(overlay_path)

        self.pyramids[session_id][layer] = ZoomifyGenerator(
            self.layers[session_id][layer]
        )
        self._bump_annotation_revision(session_id)
        return json.dumps(layer)

    def _add_annotation_overlay(self, session_id: str, overlay_path: Path) -> str:
        if overlay_path.suffix == ".geojson":

            def unpack_qupath(ann: Annotation) -> Annotation:
                # Helper function to unpack QuPath measurements if present.
                props = ann.properties
                if "measurements" in props:
                    measurements = props.pop("measurements")
                    for k, v in measurements.items():
                        props[k] = v
                if "objectType" in props:
                    props["type"] = props.pop("objectType")
                return ann

            sq = SQLiteStore.from_geojson(overlay_path, transform=unpack_qupath)

        if overlay_path.suffix == ".dat":
            sq = store_from_dat(overlay_path)

        if overlay_path.suffix == ".db":
            sq = SQLiteStore(overlay_path, auto_commit=False)
        else:
            # make a temporary db for the new annotations
            tmp_path = Path(tempfile.gettempdir()) / f"temp_{session_id}.db"
            sq.dump(tmp_path)
            sq = SQLiteStore(tmp_path)

        for layer_name, layer in self.pyramids[session_id].items():
            if isinstance(layer, AnnotationTileGenerator):
                layer.store = sq
                self._warn_annotation_coarse_prefilter_unavailable(
                    session_id,
                    layer_name,
                    sq,
                )
                self._bump_annotation_revision(session_id)
                logger.info("Loaded %d annotations.", len(sq))
                types = self.update_types(sq)
                return json.dumps(types)

        self.pyramids[session_id]["overlay"] = AnnotationTileGenerator(
            self.layers[session_id]["slide"].info,
            sq,
            self.renderers[session_id],
            overlap=self.overlaps[session_id],
        )
        self.layers[session_id]["overlay"] = self.pyramids[session_id]["overlay"]
        self._warn_annotation_coarse_prefilter_unavailable(
            session_id,
            "overlay",
            sq,
        )
        self._bump_annotation_revision(session_id)
        logger.info(
            "Loaded %d annotations.", len(self.pyramids[session_id]["overlay"].store)
        )
        types = self.update_types(sq)
        return json.dumps(types)

    def get_properties(self: TileServer, ann_type: str) -> str:
        """Get all the properties of the annotations in the store.

        Args:
            ann_type (str): The type of annotations to get the properties for.

        Returns:
            str: A jsonified list of the properties.

        """
        request_start = time.perf_counter()
        session_id = self._get_session_id()
        layer_name = request.args.get("layer_name")
        where = self._annotation_type_where_clause(ann_type)
        cache_key = self._annotation_metadata_cache_key(
            session_id,
            layer_name,
            "properties",
            where,
        )
        cached = self.annotation_metadata_cache.get(cache_key)
        if cached is not None:
            result = json.dumps(cached)
            self._log_annotation_perf(
                "properties",
                time.perf_counter() - request_start,
                layer=layer_name,
                ann_type=ann_type,
                properties=len(cached),
                cache=True,
            )
            return result

        query_start = time.perf_counter()
        ann_layer = self.get_ann_layer(session_id, layer_name=layer_name)
        if isinstance(ann_layer.store, SQLiteStore):
            try:
                unique_props = ann_layer.store.property_names(where=where)
                rows = None
            except TypeError:
                ann_props = ann_layer.store.pquery(
                    select="*",
                    where=where,
                    unique=False,
                )
                props = []
                for prop_dict in ann_props.values():
                    props.extend(list(prop_dict.keys()))
                unique_props = set(props)
                rows = len(ann_props)
        else:
            ann_props = ann_layer.store.pquery(
                select="*",
                where=where,
                unique=False,
            )
            props = []
            for prop_dict in ann_props.values():
                props.extend(list(prop_dict.keys()))
            unique_props = set(props)
            rows = len(ann_props)
        query_elapsed = time.perf_counter() - query_start
        result_values = list(unique_props)
        self.annotation_metadata_cache[cache_key] = result_values
        result = json.dumps(result_values)
        self._log_annotation_perf(
            "properties",
            time.perf_counter() - request_start,
            layer=layer_name,
            ann_type=ann_type,
            rows=rows,
            properties=len(unique_props),
            cache=False,
            query_ms=f"{query_elapsed * 1000:.1f}",
        )
        return result

    def get_property_values(self: TileServer, prop: str, ann_type: str) -> str:
        """Get all the values of a property in the store.

        Args:
            prop (str): The property to get the values of.
            ann_type (str): The type of annotations to get the values for.

        Returns:
            str: A jsonified list of the values of the property.
        """
        request_start = time.perf_counter()
        session_id = self._get_session_id()
        layer_name = request.args.get("layer_name")
        where = self._annotation_type_where_clause(ann_type)
        cache_key = self._annotation_metadata_cache_key(
            session_id,
            layer_name,
            "property_values",
            prop,
            where,
        )
        cached = self.annotation_metadata_cache.get(cache_key)
        if cached is not None:
            result = json.dumps(cached)
            self._log_annotation_perf(
                "property_values",
                time.perf_counter() - request_start,
                layer=layer_name,
                ann_type=ann_type,
                property=prop,
                values=len(cached),
                cache=True,
            )
            return result

        try:
            ann_layer = self.get_ann_layer(session_id, layer_name=layer_name)
        except ValueError:
            return json.dumps([])
        query_start = time.perf_counter()
        if isinstance(ann_layer.store, SQLiteStore):
            try:
                ann_props = ann_layer.store.property_values(prop, where=where)
            except TypeError:
                ann_props = ann_layer.store.pquery(
                    select=f"props['{prop}']",
                    where=where,
                    unique=True,
                )
        else:
            ann_props = ann_layer.store.pquery(
                select=f"props['{prop}']",
                where=where,
                unique=True,
            )
        query_elapsed = time.perf_counter() - query_start
        values = list(ann_props)
        self.annotation_metadata_cache[cache_key] = values
        result = json.dumps(values)
        self._log_annotation_perf(
            "property_values",
            time.perf_counter() - request_start,
            layer=layer_name,
            ann_type=ann_type,
            property=prop,
            values=len(values),
            cache=False,
            query_ms=f"{query_elapsed * 1000:.1f}",
        )
        return result

    def get_property_summary(self: TileServer, prop: str, ann_type: str) -> Response:
        """Summarise a property for frontend legend generation."""
        request_start = time.perf_counter()
        session_id = self._get_session_id()
        layer_name = request.args.get("layer_name")
        where = self._annotation_type_where_clause(ann_type)
        extra_where = self._decode_optional_json(request.args.get("where"))
        if extra_where is not None:
            where = extra_where if where is None else f"({where}) and ({extra_where})"
        cache_key = self._annotation_metadata_cache_key(
            session_id,
            layer_name,
            "property_summary",
            prop,
            ann_type,
            where,
        )
        cached = self.annotation_metadata_cache.get(cache_key)
        if cached is not None:
            self._log_annotation_perf(
                "property_summary",
                time.perf_counter() - request_start,
                layer=layer_name,
                ann_type=ann_type,
                property=prop,
                kind=cached.get("kind"),
                values=len(cached.get("values", [])),
                filtered=extra_where is not None,
                cache=True,
            )
            return jsonify(cached)

        try:
            ann_layer = self.get_ann_layer(session_id, layer_name=layer_name)
        except ValueError:
            return jsonify({"kind": "empty", "values": []})

        query_start = time.perf_counter()
        summary = None
        if isinstance(ann_layer.store, SQLiteStore):
            try:
                summary = ann_layer.store.property_summary(prop, where=where)
            except TypeError:
                summary = None
        if summary is not None:
            query_elapsed = time.perf_counter() - query_start
            self.annotation_metadata_cache[cache_key] = summary
            self._log_annotation_perf(
                "property_summary",
                time.perf_counter() - request_start,
                layer=layer_name,
                ann_type=ann_type,
                property=prop,
                kind=summary["kind"],
                values=len(summary.get("values", [])),
                filtered=extra_where is not None,
                cache=False,
                query_ms=f"{query_elapsed * 1000:.1f}",
            )
            return jsonify(summary)

        values = list(
            ann_layer.store.pquery(
                select=f"props['{prop}']",
                where=where,
                unique=True,
            ),
        )
        query_elapsed = time.perf_counter() - query_start
        values = [value for value in values if value is not None]
        if not values:
            summary = {"kind": "empty", "values": []}
            response = jsonify(summary)
            self.annotation_metadata_cache[cache_key] = summary
            self._log_annotation_perf(
                "property_summary",
                time.perf_counter() - request_start,
                layer=layer_name,
                ann_type=ann_type,
                property=prop,
                kind="empty",
                values=0,
                filtered=extra_where is not None,
                cache=False,
                query_ms=f"{query_elapsed * 1000:.1f}",
            )
            return response

        if all(isinstance(value, (int, float)) for value in values):
            summary = {
                "kind": "numeric",
                "min": min(values),
                "max": max(values),
            }
            response = jsonify(summary)
            self.annotation_metadata_cache[cache_key] = summary
            self._log_annotation_perf(
                "property_summary",
                time.perf_counter() - request_start,
                layer=layer_name,
                ann_type=ann_type,
                property=prop,
                kind="numeric",
                values=len(values),
                filtered=extra_where is not None,
                cache=False,
                query_ms=f"{query_elapsed * 1000:.1f}",
            )
            return response

        summary = {
            "kind": "categorical",
            "values": sorted(str(value) for value in values),
        }
        response = jsonify(summary)
        self.annotation_metadata_cache[cache_key] = summary
        self._log_annotation_perf(
            "property_summary",
            time.perf_counter() - request_start,
            layer=layer_name,
            ann_type=ann_type,
            property=prop,
            kind="categorical",
            values=len(values),
            filtered=extra_where is not None,
            cache=False,
            query_ms=f"{query_elapsed * 1000:.1f}",
        )
        return response

    def commit_db(self: TileServer) -> str:
        """Commit changes to the current store.

        If the store is not already associated with a .db file,
        the save_path is used to create a new .db file.

        """
        session_id = self._get_session_id()
        save_path = request.form["save_path"]
        save_path = self.decode_safe_name(save_path)
        for layer in self.pyramids[session_id].values():
            if isinstance(layer, AnnotationTileGenerator):
                if (
                    layer.store.path.suffix == ".db"
                    and layer.store.path.name != f"temp_{session_id}.db"
                    and not str(layer.store.path.parent.name).endswith("bokeh_temp")
                ):
                    logger.info("%s*.db committed.", layer.store.path.stem)
                    layer.store.commit()
                else:
                    layer.store.commit()
                    layer.store.dump(str(save_path))
                    logger.info("db saved to %s.", save_path)
                return "done"
        return "nothing to save"

    def get_color_prop(self: TileServer) -> Response:
        """Get the property used to color annotations from renderer."""
        session_id = self._get_session_id()
        return jsonify(self.renderers[session_id].score_prop)

    def get_slide(self: TileServer) -> Response:
        """Get the slide metadata."""
        session_id = self._get_session_id()
        info = self.layers[session_id]["slide"].info.as_dict()
        info["file_path"] = str(info["file_path"])
        return jsonify(info)

    def get_mapper(self: TileServer) -> Response:
        """Get the mapper used to color annotations from renderer."""
        session_id = self._get_session_id()
        mapper = self.renderers[session_id].raw_mapper
        return jsonify(mapper)

    def get_annotations(self: TileServer) -> Response:
        """Get the annotations in the specified bounds."""
        session_id = self._get_session_id()
        bounds = json.loads(request.form["bounds"])
        where = json.loads(request.form["where"])
        annotations = self.get_ann_layer(session_id).store.query(
            geometry=bounds,
            where=where,
        )
        annotations = [
            {"geom": ann.geometry.wkt, "properties": ann.properties}
            for ann in annotations.values()
        ]
        return jsonify(annotations)

    def get_annotations_geojson(self: TileServer) -> Response:
        """Get annotations as GeoJSON with coordinates aligned to OpenLayers."""
        session_id = self._get_session_id()
        layer_name = request.args.get("layer_name")
        try:
            ann_layer = self.get_ann_layer(session_id, layer_name=layer_name)
        except ValueError:
            return jsonify({"type": "FeatureCollection", "features": []})

        bounds = self._decode_optional_json(request.args.get("bounds"))
        where = self._decode_optional_json(request.args.get("where"))
        geojson_policy = self._get_annotation_geojson_policy(ann_layer)
        debug_requested = self._annotation_geojson_debug_requested()
        if not bool(geojson_policy.get("allowed", False)):
            error_payload = {
                "error": geojson_policy.get(
                    "reason",
                    "geojson_request_not_allowed",
                ),
                "message": geojson_policy.get(
                    "message",
                    "GeoJSON is not available for this annotation overlay.",
                ),
                "feature_count": geojson_policy.get("feature_count"),
                "auto_feature_limit": geojson_policy.get("auto_feature_limit"),
                "debug_only": bool(geojson_policy.get("debug_only")),
            }
            if debug_requested and bounds is None:
                error_payload["error"] = "geojson_debug_bounds_required"
                error_payload["message"] = (
                    "Large SQLite-backed overlays only expose GeoJSON in explicit "
                    "debug mode when bounds are supplied."
                )
                return jsonify(error_payload), 409
            if not (debug_requested and bounds is not None):
                return jsonify(error_payload), 409

        annotations = ann_layer.store.query(
            geometry=bounds,
            where=where,
        )
        features = []
        for key, ann in annotations.items():
            geometry = scale_geometry(ann.geometry, xfact=1, yfact=-1, origin=(0, 0))
            features.append(
                {
                    "id": key,
                    "type": "Feature",
                    "geometry": geometry.__geo_interface__,
                    "properties": ann.properties,
                },
            )
        return jsonify({"type": "FeatureCollection", "features": features})

    def get_annotations_mvt(
        self: TileServer,
        layer: str,
        session_id: str,
        z: int,
        x: int,
        y: int,
        representation: str = ANNOTATION_MVT_FULL_REPRESENTATION,
    ) -> Response:
        """Serve an MVT tile for an annotation layer."""
        request_start = time.perf_counter()
        layer_name = self.decode_layer_name(layer)
        where = self._decode_optional_json(request.args.get("where"))
        color_property = request.args.get("cprop") or None
        requested_fields = self._parse_annotation_tile_requested_fields(
            request.args.get("fields"),
        )
        if representation not in {
            ANNOTATION_MVT_OVERVIEW_REPRESENTATION,
            ANNOTATION_MVT_FULL_REPRESENTATION,
            ANNOTATION_MVT_CENTROID_REPRESENTATION,
        }:
            return Response("Representation not found", status=404)
        try:
            ann_layer = self.get_ann_layer(session_id, layer_name=layer_name)
            grid_width, grid_height = ann_layer.tile_grid_size(z)
        except KeyError:
            return Response("Layer not found", status=404)
        except (IndexError, ValueError):
            return Response("Tile not found", status=404)

        if x < 0 or y < 0 or x >= grid_width or y >= grid_height:
            return Response("Tile not found", status=404)

        tile_bounds = self._get_annotation_tile_bounds(ann_layer, z, x, y)
        render_hints = self._get_annotation_mvt_hints(ann_layer, z)
        cache_key = self._annotation_mvt_cache_key(
            session_id,
            layer_name,
            representation,
            z,
            x,
            y,
            where,
            color_property,
            requested_fields,
        )
        cached = self._get_annotation_mvt_cache_entry(cache_key)
        if cached is not None:
            payload = cached["payload"]
            etag = cached["etag"]
            assert isinstance(payload, bytes)
            assert isinstance(etag, str)
            response = self._make_annotation_mvt_response(payload, etag)
            self._log_annotation_perf(
                "mvt_tile",
                time.perf_counter() - request_start,
                layer=layer_name,
                representation=representation,
                z=z,
                x=x,
                y=y,
                candidates=int(cached.get("candidates", 0)),
                bytes=len(payload),
                filtered=where is not None,
                prefiltered=bool(cached.get("prefiltered", False)),
                cache=True,
                query_ms="0.0",
                encode_ms="0.0",
                geometry_decode_ms="0.0",
                prepare_ms="0.0",
                mvt_geometry_ms="0.0",
                mvt_tags_ms="0.0",
                mvt_feature_ms="0.0",
                mvt_layer_ms="0.0",
                output_features=int(cached.get("output_features", 0)),
                etag_ms="0.0",
                simplify=(
                    f"{render_hints['simplify_tolerance']:.2f}"
                    if representation == ANNOTATION_MVT_FULL_REPRESENTATION
                    else "0.00"
                ),
                min_area=(
                    f"{render_hints['min_polygon_area']:.2f}"
                    if representation == ANNOTATION_MVT_FULL_REPRESENTATION
                    else "0.00"
                ),
            )
            return response

        query_start = time.perf_counter()
        try:
            annotations, prefiltered = self._query_annotations_for_mvt(
                ann_layer,
                tile_bounds,
                where,
                render_hints,
                representation,
                color_property,
                requested_fields,
            )
        except LookupError:
            return Response("Representation not found", status=404)
        except ValueError:
            annotations, prefiltered = [], False
        query_elapsed = time.perf_counter() - query_start

        mvt_timings: dict[str, float | int] = {}

        encode_start = time.perf_counter()
        payload = (
            encode_annotation_layer(
                layer_name,
                annotations,
                tile_bounds=tile_bounds,
                extent=DEFAULT_MVT_EXTENT,
                buffer=DEFAULT_MVT_BUFFER,
                timing=mvt_timings,
                **self._get_annotation_mvt_encode_kwargs(
                    render_hints,
                    representation,
                ),
            )
            if annotations
            else encode_empty_annotation_layer(
                layer_name,
                extent=DEFAULT_MVT_EXTENT,
            )
        )
        encode_elapsed = time.perf_counter() - encode_start
        etag_start = time.perf_counter()
        etag = hashlib.blake2b(payload, digest_size=16).hexdigest()
        etag_elapsed = time.perf_counter() - etag_start
        self._set_annotation_mvt_cache_entry(
            cache_key,
            payload,
            etag,
            candidates=len(annotations),
            prefiltered=prefiltered,
            output_features=int(mvt_timings.get("output_features", 0)),
        )
        response = self._make_annotation_mvt_response(payload, etag)
        self._log_annotation_perf(
            "mvt_tile",
            time.perf_counter() - request_start,
            layer=layer_name,
            representation=representation,
            z=z,
            x=x,
            y=y,
            candidates=len(annotations),
            bytes=len(payload),
            filtered=where is not None,
            query_ms=f"{query_elapsed * 1000:.1f}",
            prefiltered=prefiltered,
            cache=False,
            encode_ms=f"{encode_elapsed * 1000:.1f}",
            geometry_decode_ms=(
                f"{float(mvt_timings.get('geometry_decode_s', 0)) * 1000:.1f}"
            ),
            prepare_ms=f"{float(mvt_timings.get('prepare_s', 0)) * 1000:.1f}",
            mvt_geometry_ms=(
                f"{float(mvt_timings.get('geometry_encode_s', 0)) * 1000:.1f}"
            ),
            mvt_tags_ms=f"{float(mvt_timings.get('tag_encode_s', 0)) * 1000:.1f}",
            mvt_feature_ms=(
                f"{float(mvt_timings.get('feature_build_s', 0)) * 1000:.1f}"
            ),
            mvt_layer_ms=f"{float(mvt_timings.get('layer_build_s', 0)) * 1000:.1f}",
            output_features=int(mvt_timings.get("output_features", 0)),
            etag_ms=f"{etag_elapsed * 1000:.1f}",
            simplify=(
                f"{render_hints['simplify_tolerance']:.2f}"
                if representation == ANNOTATION_MVT_FULL_REPRESENTATION
                else "0.00"
            ),
            min_area=(
                f"{render_hints['min_polygon_area']:.2f}"
                if representation == ANNOTATION_MVT_FULL_REPRESENTATION
                else "0.00"
            ),
        )
        return response

    def get_overlay(self: TileServer) -> Response:
        """Get the overlay info."""
        session_id = self._get_session_id()
        return jsonify(str(self.get_ann_layer(session_id).store.path))

    def get_renderer(self: TileServer, prop: str) -> Response:
        """Get the requested property from the renderer."""
        session_id = self._get_session_id()
        return jsonify(getattr(self.renderers[session_id], prop))

    def get_secondary_cmap(self: TileServer) -> Response:
        """Get the secondary cmap from the renderer."""
        session_id = self._get_session_id()
        mapper = self.renderers[session_id].secondary_cmap
        mapper["mapper"] = mapper["mapper"].__class__.__name__
        return jsonify(mapper)

    def tap_query(self: TileServer, x: float, y: float) -> Response:
        """Query for annotations at a point.

        Args:
            x (float): The x coordinate.
            y (float): The y coordinate.

        Returns:
            Response: The jsonified dict of the properties of the
            smallest annotation returned from the query at the point.

        """
        session_id = self._get_session_id()
        anns = self.get_ann_layer(session_id).store.query(
            Point(x, y),
        )
        if len(anns) == 0:
            return json.dumps({})
        return jsonify(list(anns.values())[-1].properties)

    def get_annotation_details(self: TileServer) -> Response:
        """Fetch full properties for a single annotation by key."""
        session_id = self._get_session_id()
        layer_name = request.args.get("layer_name")
        key = request.args.get("key")
        if not key:
            return Response("Missing key", status=400)

        try:
            ann_layer = self.get_ann_layer(session_id, layer_name=layer_name)
        except ValueError:
            return Response("Layer not found", status=404)

        try:
            annotation = ann_layer.store[key]
        except KeyError:
            return Response("Annotation not found", status=404)

        return jsonify({"id": key, "properties": annotation.properties})

    def prop_range(self: TileServer) -> str:
        """Set the range which the color mapper will map to.

        It will create an appropriate function to map the range to the
        range [0, 1], and set the renderers score_fn to this function.

        """
        session_id = self._get_session_id()
        prop_range = json.loads(request.form["range"])
        if prop_range is None:
            self.renderers[session_id].score_fn = lambda x: x
            return "done"
        minv, maxv = prop_range
        self.renderers[session_id].score_fn = lambda x: (x - minv) / (maxv - minv)
        return "done"

    def get_channels(self: TileServer) -> Response:
        """Get the channels of the slide."""
        session_id = self._get_session_id()
        if isinstance(self.layers[session_id]["slide"].post_proc, MultichannelToRGB):
            if not self.layers[session_id]["slide"].post_proc.is_validated:
                # trigger validation of channels with small read
                _ = self.layers[session_id]["slide"].read_rect((0, 0), (100, 100))
            return jsonify(
                {
                    "channels": self.layers[session_id]["slide"].post_proc.color_dict,
                    "active": self.layers[session_id]["slide"].post_proc.channels,
                },
            )
        return jsonify({"channels": {}, "active": []})

    def set_channels(self: TileServer) -> str:
        """Set the channels of the slide."""
        session_id = self._get_session_id()
        if isinstance(self.layers[session_id]["slide"].post_proc, MultichannelToRGB):
            channels = json.loads(request.form["channels"])
            active = json.loads(request.form["active"])
            self.layers[session_id]["slide"].post_proc.color_dict = channels
            self.layers[session_id]["slide"].post_proc.channels = active
            self.layers[session_id]["slide"].post_proc.is_validated = False
        return "done"

    def set_enhance(self: TileServer) -> str:
        """Set the enhance factor of the slide."""
        session_id = self._get_session_id()
        enhance = json.loads(request.form["val"])
        if isinstance(self.layers[session_id]["slide"].post_proc, MultichannelToRGB):
            self.layers[session_id]["slide"].post_proc.enhance = enhance
        return "done"

    def sessions(self: TileServer) -> Response:
        """Retrieve a mapping of session keys to their corresponding slide file paths.

        Returns:
            Response:
                A JSON response containing a mapping of session keys
                and their respective slide file paths.

        """
        session_paths = {}
        for key, layer in self.layers.items():
            slide = layer.get("slide")
            if slide is not None:
                session_paths[key] = str(slide.info.as_dict().get("file_path", ""))
        return jsonify(session_paths)

    @staticmethod
    def healthcheck() -> Response:
        """Simple health check endpoint to verify the server is running.

        Useful for load balancers or uptime monitoring tools to check
        if the service is operational.

        Returns:
            Response:
                A JSON response with status "OK" and HTTP status code 200.

        """
        return jsonify({"status": "OK"})

    @staticmethod
    def shutdown() -> None:
        """Shutdown the tileserver."""
        sys.exit()
