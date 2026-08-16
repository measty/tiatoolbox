"""Flask routes for the efficient viewer and immutable annotation tiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flask import Blueprint, Response, jsonify, redirect, request, send_from_directory

from tiatoolbox import data
from tiatoolbox.annotation.storage import (
    PROPERTY_FILTER_MAX_BYTES,
    PROPERTY_FILTER_MAX_DEPTH,
    PROPERTY_FILTER_MAX_IN_VALUES,
    PROPERTY_FILTER_MAX_NODES,
    normalize_property_filter,
)
from tiatoolbox.visualization.annotation_tiles.source import (
    TileBudgetExceededError,
)

if TYPE_CHECKING:  # pragma: no cover
    from tiatoolbox.visualization.annotation_tiles.cache import TilePayload
    from tiatoolbox.visualization.annotation_tiles.source import Representation
    from tiatoolbox.visualization.api.services import (
        ViewerSession,
        VisualizationServices,
    )

_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_NO_CACHE = "no-cache"


def create_viewer_blueprint(  # noqa: C901, PLR0915
    services: VisualizationServices,
) -> Blueprint:
    """Create the versioned viewer/API blueprint."""
    blueprint = Blueprint("efficient_viewer", __name__)
    app_dir = data._local_sample_path(Path("visualization") / "app")  # noqa: SLF001

    def session() -> tuple[ViewerSession, bool]:
        return services.ensure_session(request.cookies.get("session_id"))

    def json_response(document, *, status: int = 200, cache: str = _NO_CACHE):  # noqa: ANN001, ANN202
        response = jsonify(document)
        response.status_code = status
        response.headers["Cache-Control"] = cache
        response.headers["X-Content-Type-Options"] = "nosniff"
        if cache == _IMMUTABLE_CACHE:
            response.add_etag()
            response.make_conditional(request)
        return response

    def request_property_filter() -> dict[str, object] | None:
        values = request.args.getlist("filter")
        if not values:
            return None
        if len(values) != 1:
            msg = "The filter query parameter may be supplied only once."
            raise ValueError(msg)
        raw_filter = values[0]
        if not raw_filter:
            return None
        if len(raw_filter.encode("utf-8")) > PROPERTY_FILTER_MAX_BYTES:
            msg = (
                "Property filter exceeds maximum encoded size "
                f"{PROPERTY_FILTER_MAX_BYTES} bytes."
            )
            raise ValueError(msg)
        try:
            parsed = json.loads(raw_filter)
        except json.JSONDecodeError as exc:
            msg = "Property filter must be valid JSON."
            raise ValueError(msg) from exc
        return normalize_property_filter(
            parsed,
            max_depth=PROPERTY_FILTER_MAX_DEPTH,
            max_nodes=PROPERTY_FILTER_MAX_NODES,
            max_in_values=PROPERTY_FILTER_MAX_IN_VALUES,
        )

    @blueprint.get("/viewer")
    def viewer_redirect() -> Response:
        return redirect("/viewer/", code=308)

    @blueprint.get("/viewer/")
    @blueprint.get("/viewer/<path:asset>")
    def viewer(asset: str = "index.html") -> Response:
        """Serve committed Vite assets with SPA fallback."""
        candidate = app_dir / asset
        if not candidate.is_file():
            asset = "index.html"
        response = send_from_directory(app_dir, asset)
        response.headers["Cache-Control"] = (
            _NO_CACHE if asset == "index.html" else _IMMUTABLE_CACHE
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/api/v1/bootstrap")
    def bootstrap() -> Response:
        current, is_new = session()
        response = json_response(services.bootstrap(current))
        if is_new:
            response.set_cookie(
                "session_id",
                current.id,
                httponly=True,
                samesite="Lax",
            )
        return response

    @blueprint.get("/api/v1/session")
    def get_session() -> Response:
        current, is_new = session()
        response = json_response(services.session_document(current))
        if is_new:
            response.set_cookie(
                "session_id",
                current.id,
                httponly=True,
                samesite="Lax",
            )
        return response

    @blueprint.put("/api/v1/session/slide")
    def select_slide() -> Response:
        current, _ = session()
        body = request.get_json(silent=True) or {}
        resource_id = body.get("resourceId")
        if not isinstance(resource_id, str):
            return json_response(
                {"error": "resourceId must be a string."},
                status=400,
            )
        return json_response(services.select_slide(current, resource_id))

    @blueprint.post("/api/v1/session/overlays")
    def add_overlay() -> Response:
        current, _ = session()
        body = request.get_json(silent=True) or {}
        resource_id = body.get("resourceId")
        if not isinstance(resource_id, str):
            return json_response(
                {"error": "resourceId must be a string."},
                status=400,
            )
        return json_response(services.add_overlay(current, resource_id), status=201)

    @blueprint.delete("/api/v1/session/overlays/<layer_id>")
    def remove_overlay(layer_id: str) -> Response:
        current, _ = session()
        services.remove_overlay(current, layer_id)
        return Response(status=204)

    @blueprint.get("/api/v1/slides/<resource_id>")
    def slide_metadata(resource_id: str) -> Response:
        return json_response(services.slide_metadata(resource_id))

    @blueprint.get("/api/v1/stores/<store_id>")
    def current_store(store_id: str) -> Response:
        current, _ = session()
        source = services.get_source(current, store_id)
        return json_response(services.store_manifest(source.store_id))

    @blueprint.get("/api/v1/stores/<store_id>/revisions/<revision>")
    def immutable_store(store_id: str, revision: str) -> Response:
        current, _ = session()
        source = services.get_source(current, store_id)
        if source.revision != revision:
            return json_response({"error": "Unknown store revision."}, status=404)
        # Geometry is revisioned from the outset, but the derived schema/LOD
        # document changes once while its background sidecar is published.
        cache = _IMMUTABLE_CACHE if source.lod.ready else _NO_CACHE
        return json_response(services.store_manifest(store_id), cache=cache)

    @blueprint.get(
        "/api/v1/stores/<store_id>/revisions/<revision>/tiles/"
        "<representation>/<int:z>/<int:x>/<int:y>.mvt",
    )
    def vector_tile(
        store_id: str,
        revision: str,
        representation: str,
        z: int,
        x: int,
        y: int,
    ) -> Response:
        current, _ = session()
        source = services.get_source(current, store_id)
        if source.revision != revision:
            return json_response({"error": "Unknown store revision."}, status=404)
        if representation not in {"auto", "aggregate", "centroid", "polygon"}:
            return json_response({"error": "Unknown tile representation."}, status=404)
        fields = tuple(
            field.strip()
            for field in request.args.get("fields", "").split(",")
            if field.strip()
        )
        payload = source.vector_tile(
            cast("Representation", representation),
            z,
            x,
            y,
            fields=fields,
            property_filter=request_property_filter(),
        )
        return tile_response(payload)

    @blueprint.get(
        "/api/v1/stores/<store_id>/revisions/<revision>/tiles/"
        "labels/<int:z>/<int:x>/<int:y>.bin",
    )
    def label_tile(
        store_id: str,
        revision: str,
        z: int,
        x: int,
        y: int,
    ) -> Response:
        current, _ = session()
        source = services.get_source(current, store_id)
        if source.revision != revision:
            return json_response({"error": "Unknown store revision."}, status=404)
        payload = source.label_tile(
            z,
            x,
            y,
            category_property=request.args.get("property"),
        )
        return tile_response(payload)

    @blueprint.get(
        "/api/v1/stores/<store_id>/revisions/<revision>/features/pick",
    )
    def pick_feature(store_id: str, revision: str) -> Response:
        current, _ = session()
        source = services.get_source(current, store_id)
        if source.revision != revision:
            return json_response({"error": "Unknown store revision."}, status=404)
        try:
            x = float(request.args["x"])
            y = float(request.args["y"])
            tolerance = float(request.args.get("tolerance", "0"))
        except (KeyError, TypeError, ValueError):
            return json_response(
                {"error": "x, y, and tolerance must be finite numbers."},
                status=400,
            )
        feature_id = source.pick(x, y, tolerance)
        if feature_id is None:
            return Response(status=204)
        return json_response({"featureId": feature_id}, cache="no-store")

    @blueprint.get(
        "/api/v1/stores/<store_id>/revisions/<revision>/features/<int:feature_id>",
    )
    def feature(
        store_id: str,
        revision: str,
        feature_id: int,
    ) -> Response:
        current, _ = session()
        source = services.get_source(current, store_id)
        if source.revision != revision:
            return json_response({"error": "Unknown store revision."}, status=404)
        document = source.feature(feature_id)
        if document is None:
            return json_response({"error": "Unknown feature."}, status=404)
        return json_response(document, cache=_IMMUTABLE_CACHE)

    @blueprint.errorhandler(KeyError)
    def handle_key_error(_error: KeyError) -> Response:
        return json_response({"error": "Unknown resource."}, status=404)

    @blueprint.errorhandler(IndexError)
    def handle_index_error(_error: IndexError) -> Response:
        return json_response({"error": "Tile coordinate is out of range."}, status=404)

    @blueprint.errorhandler(PermissionError)
    def handle_permission_error(_error: PermissionError) -> Response:
        return json_response({"error": "Resource access denied."}, status=403)

    @blueprint.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> Response:
        return json_response({"error": str(error)}, status=400)

    @blueprint.errorhandler(TileBudgetExceededError)
    def handle_tile_budget_error(error: TileBudgetExceededError) -> Response:
        return json_response({"error": str(error)}, status=422)

    @blueprint.errorhandler(RuntimeError)
    def handle_runtime_error(error: RuntimeError) -> Response:
        return json_response({"error": str(error)}, status=409)

    return blueprint


def tile_response(payload: TilePayload) -> Response:
    """Convert a renderer-neutral tile payload into an immutable response."""
    transient = payload.representation == "building"
    if (
        not transient
        and payload.etag is not None
        and request.headers.get("If-None-Match") == payload.etag
    ):
        response = Response(status=304)
    else:
        response = Response(payload.data, content_type=payload.content_type)
        if payload.content_encoding:
            response.headers["Content-Encoding"] = payload.content_encoding
    if payload.etag:
        response.headers["ETag"] = payload.etag
    response.headers["Cache-Control"] = "no-store" if transient else _IMMUTABLE_CACHE
    if transient:
        response.headers["Retry-After"] = "1"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-TIAToolbox-Representation"] = payload.representation or ""
    response.headers["X-TIAToolbox-Features"] = str(payload.feature_count)
    response.headers["X-TIAToolbox-Vertices"] = str(payload.vertex_count)
    response.headers["X-TIAToolbox-Cache"] = payload.cache_status or "unknown"
    if payload.server_timing:
        response.headers["Server-Timing"] = payload.server_timing
    return response
