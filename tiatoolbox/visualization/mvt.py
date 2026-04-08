"""Minimal Mapbox Vector Tile encoding helpers for annotation overlays."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from numbers import Integral, Real
from typing import Any

from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

DEFAULT_MVT_EXTENT = 4096
DEFAULT_MVT_BUFFER = 256

_MOVE_TO = 1
_LINE_TO = 2
_CLOSE_PATH = 7

_GEOM_UNKNOWN = 0
_GEOM_POINT = 1
_GEOM_LINESTRING = 2
_GEOM_POLYGON = 3


class _TileEnvelope:
    """Container for slide/tile bounds used during encoding."""

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        *,
        extent: int,
    ) -> None:
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.max_x = float(max_x)
        self.max_y = float(max_y)
        self.extent = int(extent)
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        self.scale_x = self.extent / width if width else 0.0
        self.scale_y = self.extent / height if height else 0.0

    @property
    def clip_box(self) -> Polygon:
        """Return the bounds as a Shapely polygon."""
        return box(self.min_x, self.min_y, self.max_x, self.max_y)


def encode_annotation_layer(
    layer_name: str,
    annotations: Iterable[tuple[BaseGeometry, dict[str, Any]]],
    *,
    tile_bounds: tuple[float, float, float, float],
    extent: int = DEFAULT_MVT_EXTENT,
    buffer: int = DEFAULT_MVT_BUFFER,
    simplify_tolerance: float = 0.0,
    min_line_length: float = 0.0,
    min_polygon_area: float = 0.0,
) -> bytes:
    """Encode annotation geometries and properties into an MVT layer.

    Args:
        layer_name:
            Name of the MVT layer.
        annotations:
            Iterable of ``(geometry, properties)`` pairs in slide pixel coordinates.
        tile_bounds:
            Tile bounds in slide pixel coordinates ``(min_x, min_y, max_x, max_y)``.
        extent:
            MVT extent for tile-local coordinates.
        buffer:
            Tile buffer in MVT coordinate space.
        simplify_tolerance:
            Tolerance, in slide pixel coordinates, used to simplify line and polygon
            geometry before MVT encoding.
        min_line_length:
            Minimum line length, in slide pixel coordinates, to keep after clipping.
        min_polygon_area:
            Minimum polygon area, in slide pixel coordinates squared, to keep after
            clipping. Thin polygons are retained if they still span a visible edge.

    Returns:
        bytes:
            Encoded Mapbox Vector Tile payload.

    """
    envelope = _TileEnvelope(*tile_bounds, extent=extent)
    clip_geometry = _buffered_clip_geometry(envelope, buffer)

    features = []
    key_index: dict[str, int] = {}
    value_index: dict[tuple[str, Any], int] = {}
    keys: list[str] = []
    values: list[bytes] = []

    for geometry, properties in annotations:
        for simple_geometry in _prepare_geometries(
            geometry,
            clip_geometry,
            simplify_tolerance=simplify_tolerance,
            min_line_length=min_line_length,
            min_polygon_area=min_polygon_area,
        ):
            encoded_geometry, geom_type = _encode_geometry(simple_geometry, envelope)
            if not encoded_geometry or geom_type == _GEOM_UNKNOWN:
                continue
            tags = _encode_tags(properties, key_index, value_index, keys, values)
            feature = bytearray()
            if tags:
                feature.extend(_field_key(2, 2))
                feature.extend(_encode_length_delimited(_encode_packed_uint32(tags)))
            feature.extend(_encode_uint32_field(3, geom_type))
            feature.extend(_field_key(4, 2))
            feature.extend(_encode_length_delimited(_encode_packed_uint32(encoded_geometry)))
            features.append(bytes(feature))

    layer = bytearray()
    layer.extend(_encode_string_field(1, layer_name))
    for feature in features:
        layer.extend(_encode_message_field(2, feature))
    for key in keys:
        layer.extend(_encode_string_field(3, key))
    for value in values:
        layer.extend(_encode_message_field(4, value))
    layer.extend(_encode_uint32_field(5, extent))
    layer.extend(_encode_uint32_field(15, 2))

    tile = bytearray()
    tile.extend(_encode_message_field(3, bytes(layer)))
    return bytes(tile)


def encode_empty_annotation_layer(
    layer_name: str,
    *,
    extent: int = DEFAULT_MVT_EXTENT,
) -> bytes:
    """Encode an empty annotation tile with stable layer metadata."""
    return encode_annotation_layer(
        layer_name,
        [],
        tile_bounds=(0.0, 0.0, float(extent), float(extent)),
        extent=extent,
        buffer=0,
    )


def _buffered_clip_geometry(envelope: _TileEnvelope, buffer: int) -> Polygon:
    buffer_x = buffer / envelope.scale_x if envelope.scale_x else 0.0
    buffer_y = buffer / envelope.scale_y if envelope.scale_y else 0.0
    return box(
        envelope.min_x - buffer_x,
        envelope.min_y - buffer_y,
        envelope.max_x + buffer_x,
        envelope.max_y + buffer_y,
    )


def _prepare_geometries(
    geometry: BaseGeometry,
    clip_geometry: Polygon,
    *,
    simplify_tolerance: float = 0.0,
    min_line_length: float = 0.0,
    min_polygon_area: float = 0.0,
) -> Iterator[Point | LineString | Polygon]:
    """Clip and split geometries into simple MVT-supported parts."""
    if geometry.is_empty:
        return

    candidate = geometry
    bounds = geometry.bounds
    clip_bounds = clip_geometry.bounds
    if (
        bounds[0] < clip_bounds[0]
        or bounds[1] < clip_bounds[1]
        or bounds[2] > clip_bounds[2]
        or bounds[3] > clip_bounds[3]
    ):
        candidate = geometry.intersection(clip_geometry)

    candidate = _simplify_geometry(candidate, simplify_tolerance)
    yield from _iter_renderable_geometries(
        candidate,
        min_line_length=min_line_length,
        min_polygon_area=min_polygon_area,
    )


def _simplify_geometry(geometry: BaseGeometry, tolerance: float) -> BaseGeometry:
    """Simplify non-point geometry prior to MVT quantisation."""
    if tolerance <= 0 or isinstance(geometry, (Point, MultiPoint)):
        return geometry
    return geometry.simplify(tolerance, preserve_topology=True)


def _iter_renderable_geometries(
    geometry: BaseGeometry,
    *,
    min_line_length: float = 0.0,
    min_polygon_area: float = 0.0,
) -> Iterator[Point | LineString | Polygon]:
    """Yield simple geometries which remain visible at the current tile scale."""
    for simple_geometry in _iter_simple_geometries(geometry):
        if _is_renderable_geometry(
            simple_geometry,
            min_line_length=min_line_length,
            min_polygon_area=min_polygon_area,
        ):
            yield simple_geometry


def _is_renderable_geometry(
    geometry: Point | LineString | Polygon,
    *,
    min_line_length: float = 0.0,
    min_polygon_area: float = 0.0,
) -> bool:
    """Cull geometries which are too small to survive tile rendering."""
    if isinstance(geometry, Point):
        return True

    if isinstance(geometry, LineString):
        return geometry.length >= min_line_length

    if isinstance(geometry, Polygon):
        if geometry.area >= min_polygon_area:
            return True
        bounds = geometry.bounds
        return max(bounds[2] - bounds[0], bounds[3] - bounds[1]) >= min_line_length

    return False


def _iter_simple_geometries(
    geometry: BaseGeometry,
) -> Iterator[Point | LineString | Polygon]:
    if geometry.is_empty:
        return

    if isinstance(geometry, Point):
        yield geometry
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from _iter_simple_geometries(part)


def _encode_geometry(
    geometry: Point | LineString | Polygon,
    envelope: _TileEnvelope,
) -> tuple[list[int], int]:
    if isinstance(geometry, Point):
        return _encode_point_geometry(geometry, envelope), _GEOM_POINT
    if isinstance(geometry, LineString):
        return _encode_linestring_geometry(geometry, envelope), _GEOM_LINESTRING
    if isinstance(geometry, Polygon):
        return _encode_polygon_geometry(geometry, envelope), _GEOM_POLYGON
    return [], _GEOM_UNKNOWN


def _encode_point_geometry(point: Point, envelope: _TileEnvelope) -> list[int]:
    x, y = _quantize_xy(point.x, point.y, envelope)
    return [_command_integer(_MOVE_TO, 1), _zigzag_encode(x), _zigzag_encode(y)]


def _encode_linestring_geometry(
    line: LineString,
    envelope: _TileEnvelope,
) -> list[int]:
    coordinates = _deduplicate_quantized(
        [_quantize_xy(x, y, envelope) for x, y in line.coords],
    )
    if len(coordinates) < 2:
        return []

    geometry = [_command_integer(_MOVE_TO, 1)]
    first_x, first_y = coordinates[0]
    geometry.extend((_zigzag_encode(first_x), _zigzag_encode(first_y)))
    geometry.append(_command_integer(_LINE_TO, len(coordinates) - 1))

    prev_x, prev_y = first_x, first_y
    for x, y in coordinates[1:]:
        geometry.extend((_zigzag_encode(x - prev_x), _zigzag_encode(y - prev_y)))
        prev_x, prev_y = x, y
    return geometry


def _encode_polygon_geometry(
    polygon: Polygon,
    envelope: _TileEnvelope,
) -> list[int]:
    geometry: list[int] = []

    exterior = _encode_ring(polygon.exterior.coords, envelope, clockwise=True)
    if exterior:
        geometry.extend(exterior)

    for interior in polygon.interiors:
        ring = _encode_ring(interior.coords, envelope, clockwise=False)
        if ring:
            geometry.extend(ring)

    return geometry


def _encode_ring(
    coordinates: Iterable[tuple[float, float]],
    envelope: _TileEnvelope,
    *,
    clockwise: bool,
) -> list[int]:
    quantized = _deduplicate_quantized(
        [_quantize_xy(x, y, envelope) for x, y in list(coordinates)[:-1]],
    )
    if len(quantized) < 3:
        return []

    area = _signed_area(quantized)
    if clockwise and area <= 0:
        quantized.reverse()
    if not clockwise and area >= 0:
        quantized.reverse()

    geometry = [_command_integer(_MOVE_TO, 1)]
    first_x, first_y = quantized[0]
    geometry.extend((_zigzag_encode(first_x), _zigzag_encode(first_y)))
    geometry.append(_command_integer(_LINE_TO, len(quantized) - 1))

    prev_x, prev_y = first_x, first_y
    for x, y in quantized[1:]:
        geometry.extend((_zigzag_encode(x - prev_x), _zigzag_encode(y - prev_y)))
        prev_x, prev_y = x, y
    geometry.append(_command_integer(_CLOSE_PATH, 1))
    return geometry


def _quantize_xy(x: float, y: float, envelope: _TileEnvelope) -> tuple[int, int]:
    local_x = int(round((x - envelope.min_x) * envelope.scale_x))
    local_y = int(round((envelope.max_y - y) * envelope.scale_y))
    return local_x, local_y


def _deduplicate_quantized(
    coordinates: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for coordinate in coordinates:
        if not output or output[-1] != coordinate:
            output.append(coordinate)
    return output


def _signed_area(coordinates: list[tuple[int, int]]) -> int:
    area = 0
    for index, (x0, y0) in enumerate(coordinates):
        x1, y1 = coordinates[(index + 1) % len(coordinates)]
        area += (x0 * y1) - (x1 * y0)
    return area


def _encode_tags(
    properties: dict[str, Any],
    key_index: dict[str, int],
    value_index: dict[tuple[str, Any], int],
    keys: list[str],
    values: list[bytes],
) -> list[int]:
    tags: list[int] = []
    for key, value in properties.items():
        if value is None:
            continue
        key_id = key_index.setdefault(key, len(keys))
        if key_id == len(keys):
            keys.append(key)

        normalised_value = _normalise_property_value(value)
        value_key = _value_cache_key(normalised_value)
        value_id = value_index.setdefault(value_key, len(values))
        if value_id == len(values):
            values.append(_encode_value(normalised_value))

        tags.extend((key_id, value_id))
    return tags


def _normalise_property_value(value: Any) -> bool | int | float | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _value_cache_key(value: bool | int | float | str) -> tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    return ("str", value)


def _encode_value(value: bool | int | float | str) -> bytes:
    payload = bytearray()
    if isinstance(value, bool):
        payload.extend(_encode_bool_field(7, value))
    elif isinstance(value, int):
        payload.extend(_encode_sint64_field(6, value))
    elif isinstance(value, float):
        payload.extend(_encode_double_field(3, value))
    else:
        payload.extend(_encode_string_field(1, value))
    return bytes(payload)


def _encode_message_field(field_number: int, payload: bytes) -> bytes:
    return _field_key(field_number, 2) + _encode_length_delimited(payload)


def _encode_string_field(field_number: int, value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _field_key(field_number, 2) + _encode_length_delimited(encoded)


def _encode_uint32_field(field_number: int, value: int) -> bytes:
    return _field_key(field_number, 0) + _encode_varint(value)


def _encode_sint64_field(field_number: int, value: int) -> bytes:
    return _field_key(field_number, 0) + _encode_varint(_zigzag_encode(value))


def _encode_bool_field(field_number: int, value: bool) -> bytes:
    return _field_key(field_number, 0) + _encode_varint(int(value))


def _encode_double_field(field_number: int, value: float) -> bytes:
    import struct

    return _field_key(field_number, 1) + struct.pack("<d", value)


def _encode_packed_uint32(values: Iterable[int]) -> bytes:
    payload = bytearray()
    for value in values:
        payload.extend(_encode_varint(value))
    return bytes(payload)


def _encode_length_delimited(payload: bytes) -> bytes:
    return _encode_varint(len(payload)) + payload


def _field_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _command_integer(command_id: int, count: int) -> int:
    return (count << 3) | command_id


def _zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 63)


def _encode_varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        msg = "Varint values must be non-negative."
        raise ValueError(msg)

    payload = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            payload.append(chunk | 0x80)
        else:
            payload.append(chunk)
            break
    return bytes(payload)
