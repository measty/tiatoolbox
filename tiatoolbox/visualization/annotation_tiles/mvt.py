"""Small, batched Mapbox Vector Tile encoder for slide-space annotations."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
from shapely import wkb as shapely_wkb
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from shapely.geometry.base import BaseGeometry

DEFAULT_EXTENT = 4096
DEFAULT_BUFFER = 128

_MOVE_TO = 1
_LINE_TO = 2
_CLOSE_PATH = 7

_GEOM_UNKNOWN = 0
_GEOM_POINT = 1
_GEOM_LINESTRING = 2
_GEOM_POLYGON = 3

_MIN_LINE_POINTS = 2
_MIN_RING_POINTS = 3
_POLYGON_OFFSET_ARRAYS = 2
_VARINT_DATA_MASK = 0x7F
_VARINT_CONTINUATION = 0x80


@dataclass(frozen=True, slots=True)
class TileFeature:
    """A renderer-facing feature in baseline slide coordinates."""

    feature_id: int | None
    geometry: BaseGeometry | bytes
    properties: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MVTEncodeResult:
    """Encoded MVT bytes and budget/diagnostic counters."""

    data: bytes
    input_features: int
    output_features: int
    output_vertices: int
    budget_exceeded: bool


@dataclass(frozen=True, slots=True)
class _Envelope:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    extent: int

    @property
    def scale_x(self) -> float:
        return self.extent / (self.max_x - self.min_x)

    @property
    def scale_y(self) -> float:
        return self.extent / (self.max_y - self.min_y)


def encode_mvt(  # noqa: PLR0912
    layer_name: str,
    features: Sequence[TileFeature] | Iterable[TileFeature],
    *,
    tile_bounds: tuple[float, float, float, float],
    extent: int = DEFAULT_EXTENT,
    buffer: int = DEFAULT_BUFFER,
    simplify_tolerance: float = 0,
    max_features: int | None = None,
    max_vertices: int | None = None,
) -> MVTEncodeResult:
    """Encode features into one MVT layer.

    Geometry decoding, clipping and simplification use Shapely 2 vectorised
    operations. Protobuf command construction remains a compact Python loop and is
    intentionally hidden behind this interface so a native encoder can replace it.
    """
    if not layer_name:
        msg = "MVT layer name must not be empty."
        raise ValueError(msg)
    if extent <= 0:
        msg = "MVT extent must be positive."
        raise ValueError(msg)

    feature_list = list(features)
    envelope = _Envelope(*map(float, tile_bounds), extent=extent)
    if envelope.max_x <= envelope.min_x or envelope.max_y <= envelope.min_y:
        msg = "Tile bounds must have positive width and height."
        raise ValueError(msg)

    geometries = _prepare_geometries(
        feature_list,
        envelope,
        buffer=buffer,
        simplify_tolerance=simplify_tolerance,
    )
    polygon_batch = _encode_polygon_batch(geometries, envelope)

    encoded_features: list[bytes] = []
    keys: list[str] = []
    values: list[bytes] = []
    key_index: dict[str, int] = {}
    value_index: dict[tuple[str, Any], int] = {}
    output_vertices = 0

    for feature_index, (tile_feature, geometry) in enumerate(
        zip(feature_list, geometries, strict=True),
    ):
        batched = polygon_batch.get(feature_index)
        if batched is not None:
            encoded_parts = [batched]
        else:
            if geometry is None or geometry.is_empty:
                continue
            encoded_parts = [
                _encode_geometry(simple_geometry, envelope)
                for simple_geometry in _homogeneous_parts(geometry)
            ]
        for part_index, (commands, geometry_type, vertex_count) in enumerate(
            encoded_parts,
        ):
            if not commands or geometry_type == _GEOM_UNKNOWN:
                continue
            if max_features is not None and len(encoded_features) >= max_features:
                return _build_result(
                    layer_name,
                    encoded_features,
                    keys,
                    values,
                    extent,
                    len(feature_list),
                    output_vertices,
                    budget_exceeded=True,
                )
            if (
                max_vertices is not None
                and output_vertices + vertex_count > max_vertices
            ):
                return _build_result(
                    layer_name,
                    encoded_features,
                    keys,
                    values,
                    extent,
                    len(feature_list),
                    output_vertices,
                    budget_exceeded=True,
                )

            tags = _encode_tags(
                tile_feature.properties,
                key_index,
                value_index,
                keys,
                values,
            )
            feature_message = bytearray()
            # A GeometryCollection may produce multiple MVT features. Keep the ID on
            # the first part only so IDs remain unique inside a tile layer.
            if tile_feature.feature_id is not None and part_index == 0:
                feature_message.extend(_encode_varint_field(1, tile_feature.feature_id))
            if tags:
                feature_message.extend(_field_key(2, 2))
                feature_message.extend(_length_delimited(_packed_uint32(tags)))
            feature_message.extend(_encode_varint_field(3, geometry_type))
            feature_message.extend(_field_key(4, 2))
            feature_message.extend(_length_delimited(_packed_uint32(commands)))
            encoded_features.append(bytes(feature_message))
            output_vertices += vertex_count

    return _build_result(
        layer_name,
        encoded_features,
        keys,
        values,
        extent,
        len(feature_list),
        output_vertices,
        budget_exceeded=False,
    )


def encode_empty_mvt(layer_name: str = "annotations") -> bytes:
    """Return a valid empty vector tile."""
    return encode_mvt(
        layer_name,
        [],
        tile_bounds=(0, 0, DEFAULT_EXTENT, DEFAULT_EXTENT),
    ).data


def _prepare_geometries(
    features: Sequence[TileFeature],
    envelope: _Envelope,
    *,
    buffer: int,
    simplify_tolerance: float,
) -> list[BaseGeometry | None]:
    if not features:
        return []
    raw_geometries = [feature.geometry for feature in features]
    if all(isinstance(geometry, bytes) for geometry in raw_geometries):
        # Shapely's array ufunc avoids one Python wrapper call per annotation.
        array = shapely.from_wkb(np.asarray(raw_geometries, dtype=object))
    else:
        array = np.asarray(
            [
                shapely_wkb.loads(geometry) if isinstance(geometry, bytes) else geometry
                for geometry in raw_geometries
            ],
            dtype=object,
        )

    buffer_x = buffer / envelope.scale_x
    buffer_y = buffer / envelope.scale_y
    clip_bounds = (
        envelope.min_x - buffer_x,
        envelope.min_y - buffer_y,
        envelope.max_x + buffer_x,
        envelope.max_y + buffer_y,
    )
    bounds = shapely.bounds(array)
    needs_clip = (
        (bounds[:, 0] < clip_bounds[0])
        | (bounds[:, 1] < clip_bounds[1])
        | (bounds[:, 2] > clip_bounds[2])
        | (bounds[:, 3] > clip_bounds[3])
    )
    if np.any(needs_clip):
        array[needs_clip] = shapely.clip_by_rect(array[needs_clip], *clip_bounds)
    if simplify_tolerance > 0:
        non_points = ~np.isin(
            shapely.get_type_id(array),
            [shapely.GeometryType.POINT, shapely.GeometryType.MULTIPOINT],
        )
        if np.any(non_points):
            array[non_points] = shapely.simplify(
                array[non_points],
                simplify_tolerance,
                # At this representation the tolerance is sub-display-pixel.
                # GEOS' topology-preserving variant is an order of magnitude
                # slower on cell segmentation and retains detail the client
                # cannot display. Exact selected geometry remains unsimplified
                # and comes from the feature endpoint.
                preserve_topology=False,
            )
    return [None if item is None else item for item in array.tolist()]


def _homogeneous_parts(geometry: BaseGeometry) -> Iterator[BaseGeometry]:
    if isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from _homogeneous_parts(child)
        return
    yield geometry


def _encode_geometry(
    geometry: BaseGeometry,
    envelope: _Envelope,
) -> tuple[list[int], int, int]:
    if isinstance(geometry, (Point, MultiPoint)):
        return _encode_points(geometry, envelope)
    if isinstance(geometry, (LineString, MultiLineString)):
        return _encode_lines(geometry, envelope)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return _encode_polygons(geometry, envelope)
    return [], _GEOM_UNKNOWN, 0


def _quantize(
    coordinates: Iterable[tuple[float, float]],
    envelope: _Envelope,
) -> list[tuple[int, int]]:
    return [
        (
            round((float(x) - envelope.min_x) * envelope.scale_x),
            round((float(y) - envelope.min_y) * envelope.scale_y),
        )
        for x, y, *_ in coordinates
    ]


def _deduplicate(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for point in points:
        if not result or result[-1] != point:
            result.append(point)
    return result


def _encode_points(
    geometry: Point | MultiPoint,
    envelope: _Envelope,
) -> tuple[list[int], int, int]:
    source_points = [geometry] if isinstance(geometry, Point) else list(geometry.geoms)
    points = _deduplicate(
        _quantize(((point.x, point.y) for point in source_points), envelope),
    )
    if not points:
        return [], _GEOM_UNKNOWN, 0
    commands = [_command(_MOVE_TO, len(points))]
    cursor_x = cursor_y = 0
    for x, y in points:
        commands.extend((_zigzag(x - cursor_x), _zigzag(y - cursor_y)))
        cursor_x, cursor_y = x, y
    return commands, _GEOM_POINT, len(points)


def _encode_lines(
    geometry: LineString | MultiLineString,
    envelope: _Envelope,
) -> tuple[list[int], int, int]:
    lines = [geometry] if isinstance(geometry, LineString) else list(geometry.geoms)
    commands: list[int] = []
    cursor_x = cursor_y = 0
    vertex_count = 0
    for line in lines:
        points = _deduplicate(_quantize(line.coords, envelope))
        if len(points) < _MIN_LINE_POINTS:
            continue
        x, y = points[0]
        commands.extend(
            (_command(_MOVE_TO, 1), _zigzag(x - cursor_x), _zigzag(y - cursor_y)),
        )
        cursor_x, cursor_y = x, y
        commands.append(_command(_LINE_TO, len(points) - 1))
        for x, y in points[1:]:
            commands.extend((_zigzag(x - cursor_x), _zigzag(y - cursor_y)))
            cursor_x, cursor_y = x, y
        vertex_count += len(points)
    return commands, _GEOM_LINESTRING if commands else _GEOM_UNKNOWN, vertex_count


def _encode_polygons(
    geometry: Polygon | MultiPolygon,
    envelope: _Envelope,
) -> tuple[list[int], int, int]:
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    commands: list[int] = []
    cursor_x = cursor_y = 0
    vertex_count = 0
    for polygon in polygons:
        rings = [(polygon.exterior.coords, True)]
        rings.extend((ring.coords, False) for ring in polygon.interiors)
        for coordinates, exterior in rings:
            points = _deduplicate(_quantize(coordinates, envelope))
            if len(points) > 1 and points[0] == points[-1]:
                points.pop()
            if len(points) < _MIN_RING_POINTS:
                continue
            signed_area = _signed_area(points)
            # Numeric slide coordinates have positive Y down on screen. Positive
            # shoelace area therefore displays clockwise, as required by MVT.
            if (exterior and signed_area < 0) or (not exterior and signed_area > 0):
                points.reverse()
            x, y = points[0]
            commands.extend(
                (_command(_MOVE_TO, 1), _zigzag(x - cursor_x), _zigzag(y - cursor_y)),
            )
            cursor_x, cursor_y = x, y
            commands.append(_command(_LINE_TO, len(points) - 1))
            for x, y in points[1:]:
                commands.extend((_zigzag(x - cursor_x), _zigzag(y - cursor_y)))
                cursor_x, cursor_y = x, y
            commands.append(_command(_CLOSE_PATH, 1))
            vertex_count += len(points)
    return commands, _GEOM_POLYGON if commands else _GEOM_UNKNOWN, vertex_count


def _encode_polygon_batch(
    geometries: Sequence[BaseGeometry | None],
    envelope: _Envelope,
) -> dict[int, tuple[list[int], int, int]]:
    """Encode simple polygons through Shapely's contiguous coordinate arrays.

    Cell-segmentation tiles commonly contain thousands of small polygons. Using
    ``exterior``, ``interiors`` and ``coords`` on every Shapely object spends
    more time in Python wrappers than in protobuf encoding. The ragged-array API
    extracts all rings once while preserving their feature boundaries.
    """
    feature_indices = [
        index
        for index, geometry in enumerate(geometries)
        if isinstance(geometry, Polygon) and not geometry.is_empty
    ]
    if not feature_indices:
        return {}
    polygons = np.asarray(
        [geometries[index] for index in feature_indices],
        dtype=object,
    )
    output: dict[int, tuple[list[int], int, int]] = {}

    # ``to_ragged_array`` first extracts every ring, which is comparatively
    # expensive for the overwhelmingly common cell-segmentation case of one
    # exterior ring and no holes. ``get_coordinates(..., return_index=True)``
    # gives us feature boundaries directly and avoids creating ring objects.
    interior_counts = np.asarray(shapely.get_num_interior_rings(polygons))
    simple_polygon_indices = np.flatnonzero(interior_counts == 0)
    if simple_polygon_indices.size:
        simple_coordinates, owners = shapely.get_coordinates(
            polygons[simple_polygon_indices],
            return_index=True,
        )
        quantized = _quantize_array(simple_coordinates, envelope)
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(owners[1:] != owners[:-1]) + 1,
            ),
        )
        stops = np.concatenate(
            (starts[1:], np.asarray([len(owners)], dtype=np.int64)),
        )
        for start, stop in zip(starts, stops, strict=True):
            owner = int(owners[int(start)])
            polygon_index = int(simple_polygon_indices[owner])
            commands: list[int] = []
            _, _, vertex_count = _append_polygon_ring(
                commands,
                quantized[int(start) : int(stop)],
                exterior=True,
                cursor_x=0,
                cursor_y=0,
            )
            if commands:
                output[feature_indices[polygon_index]] = (
                    commands,
                    _GEOM_POLYGON,
                    vertex_count,
                )

    complex_polygon_indices = np.flatnonzero(interior_counts != 0)
    if not complex_polygon_indices.size:
        return output
    geometry_type, coordinates, offsets = shapely.to_ragged_array(
        polygons[complex_polygon_indices],
    )
    if (  # pragma: no cover
        geometry_type != shapely.GeometryType.POLYGON
        or len(offsets) != _POLYGON_OFFSET_ARRAYS
    ):
        return {}
    ring_offsets, polygon_offsets = offsets
    quantized = _quantize_array(coordinates, envelope)

    for subset_index, polygon_index in enumerate(complex_polygon_indices):
        feature_index = feature_indices[int(polygon_index)]
        commands: list[int] = []
        cursor_x = cursor_y = 0
        vertex_count = 0
        first_ring = int(polygon_offsets[subset_index])
        last_ring = int(polygon_offsets[subset_index + 1])
        for ring_index in range(first_ring, last_ring):
            start = int(ring_offsets[ring_index])
            stop = int(ring_offsets[ring_index + 1])
            cursor_x, cursor_y, ring_vertices = _append_polygon_ring(
                commands,
                quantized[start:stop],
                exterior=ring_index == first_ring,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
            )
            vertex_count += ring_vertices
        if commands:
            output[feature_index] = (commands, _GEOM_POLYGON, vertex_count)
    return output


def _quantize_array(coordinates: np.ndarray, envelope: _Envelope) -> np.ndarray:
    """Quantize a contiguous Shapely coordinate array into tile coordinates."""
    return np.rint(
        (coordinates[:, :2] - np.asarray([envelope.min_x, envelope.min_y]))
        * np.asarray([envelope.scale_x, envelope.scale_y]),
    ).astype(np.int64)


def _append_polygon_ring(
    commands: list[int],
    quantized: np.ndarray,
    *,
    exterior: bool,
    cursor_x: int,
    cursor_y: int,
) -> tuple[int, int, int]:
    """Append one quantized MVT polygon ring and return its final cursor."""
    points = _deduplicate(quantized.tolist())
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < _MIN_RING_POINTS:
        return cursor_x, cursor_y, 0
    signed_area = _signed_area(points)
    if (exterior and signed_area < 0) or (not exterior and signed_area > 0):
        points.reverse()
    x, y = points[0]
    commands.extend(
        (
            _command(_MOVE_TO, 1),
            _zigzag(x - cursor_x),
            _zigzag(y - cursor_y),
        ),
    )
    cursor_x, cursor_y = x, y
    commands.append(_command(_LINE_TO, len(points) - 1))
    for x, y in points[1:]:
        commands.extend((_zigzag(x - cursor_x), _zigzag(y - cursor_y)))
        cursor_x, cursor_y = x, y
    commands.append(_command(_CLOSE_PATH, 1))
    return cursor_x, cursor_y, len(points)


def _signed_area(points: Sequence[tuple[int, int]]) -> float:
    return 0.5 * sum(
        (x1 * y2) - (x2 * y1)
        for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]), strict=True)
    )


def _encode_tags(
    properties: Mapping[str, Any],
    key_index: dict[str, int],
    value_index: dict[tuple[str, Any], int],
    keys: list[str],
    values: list[bytes],
) -> list[int]:
    tags: list[int] = []
    for raw_key, raw_value in properties.items():
        if raw_value is None:
            continue
        key = str(raw_key)
        if key not in key_index:
            key_index[key] = len(keys)
            keys.append(key)
        normalised, encoded = _encode_value(raw_value)
        value_key = (type(normalised).__name__, normalised)
        if value_key not in value_index:
            value_index[value_key] = len(values)
            values.append(encoded)
        tags.extend((key_index[key], value_index[value_key]))
    return tags


def _encode_value(value: object) -> tuple[str | int | float | bool, bytes]:
    if isinstance(value, (np.bool_, bool)):
        normalised = bool(value)
        return normalised, _encode_varint_field(7, int(normalised))
    if isinstance(value, Integral):
        normalised = int(value)
        if normalised >= 0:
            return normalised, _encode_varint_field(5, normalised)
        return normalised, _encode_varint_field(6, _zigzag(normalised))
    if isinstance(value, Real) and math.isfinite(float(value)):
        normalised = float(value)
        return normalised, _field_key(3, 1) + struct.pack("<d", normalised)
    if isinstance(value, str):
        return value, _encode_string_field(1, value)
    normalised = json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    return normalised, _encode_string_field(1, normalised)


def _build_result(
    layer_name: str,
    features: Sequence[bytes],
    keys: Sequence[str],
    values: Sequence[bytes],
    extent: int,
    input_features: int,
    output_vertices: int,
    *,
    budget_exceeded: bool,
) -> MVTEncodeResult:
    layer = bytearray(_encode_string_field(1, layer_name))
    for feature in features:
        layer.extend(_encode_message_field(2, feature))
    for key in keys:
        layer.extend(_encode_string_field(3, key))
    for value in values:
        layer.extend(_encode_message_field(4, value))
    layer.extend(_encode_varint_field(5, extent))
    layer.extend(_encode_varint_field(15, 2))
    tile = _encode_message_field(3, bytes(layer))
    return MVTEncodeResult(
        data=tile,
        input_features=input_features,
        output_features=len(features),
        output_vertices=output_vertices,
        budget_exceeded=budget_exceeded,
    )


def _command(command_id: int, count: int) -> int:
    return (count << 3) | command_id


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 63)


def _field_key(field_number: int, wire_type: int) -> bytes:
    value = (field_number << 3) | wire_type
    if value <= _VARINT_DATA_MASK:
        return bytes((value,))
    return _varint(value)


def _varint(value: int) -> bytes:
    if value < 0:
        msg = "Varints must be non-negative."
        raise ValueError(msg)
    if value <= _VARINT_DATA_MASK:
        return bytes((value,))
    encoded = bytearray()
    while value > _VARINT_DATA_MASK:
        encoded.append((value & _VARINT_DATA_MASK) | _VARINT_CONTINUATION)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _length_delimited(value: bytes) -> bytes:
    return _varint(len(value)) + value


def _packed_uint32(values: Iterable[int]) -> bytes:
    encoded = bytearray()
    for raw_value in values:
        value = int(raw_value)
        if value < 0:
            msg = "Packed uint32 values must be non-negative."
            raise ValueError(msg)
        while value > _VARINT_DATA_MASK:
            encoded.append((value & _VARINT_DATA_MASK) | _VARINT_CONTINUATION)
            value >>= 7
        encoded.append(value)
    return bytes(encoded)


def _encode_varint_field(field_number: int, value: int) -> bytes:
    return _field_key(field_number, 0) + _varint(value)


def _encode_string_field(field_number: int, value: str) -> bytes:
    return _field_key(field_number, 2) + _length_delimited(value.encode())


def _encode_message_field(field_number: int, value: bytes) -> bytes:
    return _field_key(field_number, 2) + _length_delimited(value)
