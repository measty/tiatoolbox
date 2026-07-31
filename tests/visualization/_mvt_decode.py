"""Independent, test-only decoder for the MVT subset emitted by the viewer."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ProtoValue = int | bytes
Scalar = str | int | float | bool


@dataclass(frozen=True, slots=True)
class DecodedFeature:
    """One independently decoded MVT feature."""

    feature_id: int | None
    geometry_type: int
    paths: tuple[tuple[tuple[int, int], ...], ...]
    properties: dict[str, Scalar]


@dataclass(frozen=True, slots=True)
class DecodedLayer:
    """One independently decoded MVT layer."""

    name: str
    extent: int
    version: int
    features: tuple[DecodedFeature, ...]


def decode_mvt(data: bytes, layer_name: str = "annotations") -> DecodedLayer:
    """Decode one named layer without using the production encoder internals."""
    layer_messages = [
        _as_bytes(value)
        for number, wire_type, value in _protobuf_fields(data)
        if number == 3 and wire_type == 2
    ]
    layers = [_decode_layer(message) for message in layer_messages]
    for layer in layers:
        if layer.name == layer_name:
            return layer
    message = f"MVT layer {layer_name!r} not found."
    raise AssertionError(message)


def _decode_layer(data: bytes) -> DecodedLayer:
    fields = list(_protobuf_fields(data))
    names = [
        _as_bytes(value).decode()
        for number, wire_type, value in fields
        if number == 1 and wire_type == 2
    ]
    assert len(names) == 1
    keys = [
        _as_bytes(value).decode()
        for number, wire_type, value in fields
        if number == 3 and wire_type == 2
    ]
    values = [
        _decode_value(_as_bytes(value))
        for number, wire_type, value in fields
        if number == 4 and wire_type == 2
    ]
    extents = [
        _as_int(value)
        for number, wire_type, value in fields
        if number == 5 and wire_type == 0
    ]
    versions = [
        _as_int(value)
        for number, wire_type, value in fields
        if number == 15 and wire_type == 0
    ]
    feature_messages = [
        _as_bytes(value)
        for number, wire_type, value in fields
        if number == 2 and wire_type == 2
    ]
    assert len(extents) == 1
    assert len(versions) == 1
    return DecodedLayer(
        name=names[0],
        extent=extents[0],
        version=versions[0],
        features=tuple(
            _decode_feature(message, keys, values) for message in feature_messages
        ),
    )


def _decode_feature(
    data: bytes,
    keys: list[str],
    values: list[Scalar],
) -> DecodedFeature:
    fields = list(_protobuf_fields(data))
    ids = [
        _as_int(value)
        for number, wire_type, value in fields
        if number == 1 and wire_type == 0
    ]
    types = [
        _as_int(value)
        for number, wire_type, value in fields
        if number == 3 and wire_type == 0
    ]
    geometry = [
        _as_bytes(value)
        for number, wire_type, value in fields
        if number == 4 and wire_type == 2
    ]
    packed_tags = [
        _as_bytes(value)
        for number, wire_type, value in fields
        if number == 2 and wire_type == 2
    ]
    tags = [] if not packed_tags else list(_packed_varints(packed_tags[0]))
    assert len(ids) <= 1
    assert len(types) == 1
    assert len(geometry) == 1
    assert len(tags) % 2 == 0
    properties = {
        keys[tags[index]]: values[tags[index + 1]] for index in range(0, len(tags), 2)
    }
    return DecodedFeature(
        feature_id=None if not ids else ids[0],
        geometry_type=types[0],
        paths=_decode_geometry(geometry[0]),
        properties=properties,
    )


def _decode_geometry(data: bytes) -> tuple[tuple[tuple[int, int], ...], ...]:
    commands = list(_packed_varints(data))
    paths: list[list[tuple[int, int]]] = []
    cursor_x = cursor_y = 0
    offset = 0
    current: list[tuple[int, int]] | None = None
    while offset < len(commands):
        command = commands[offset]
        offset += 1
        command_id = command & 0x7
        count = command >> 3
        if command_id == 1:  # MoveTo
            for _ in range(count):
                cursor_x += _unzigzag(commands[offset])
                cursor_y += _unzigzag(commands[offset + 1])
                offset += 2
                current = [(cursor_x, cursor_y)]
                paths.append(current)
        elif command_id == 2:  # LineTo
            assert current is not None
            for _ in range(count):
                cursor_x += _unzigzag(commands[offset])
                cursor_y += _unzigzag(commands[offset + 1])
                offset += 2
                current.append((cursor_x, cursor_y))
        elif command_id == 7:  # ClosePath
            assert count == 1
            assert current is not None
            if current[-1] != current[0]:
                current.append(current[0])
        else:
            message = f"Unsupported MVT geometry command {command_id}."
            raise AssertionError(message)
    return tuple(tuple(path) for path in paths)


def _decode_value(data: bytes) -> Scalar:
    fields = list(_protobuf_fields(data))
    assert len(fields) == 1
    number, wire_type, value = fields[0]
    if number == 1 and wire_type == 2:
        return _as_bytes(value).decode()
    if number == 2 and wire_type == 5:
        return struct.unpack("<f", _as_bytes(value))[0]
    if number == 3 and wire_type == 1:
        return struct.unpack("<d", _as_bytes(value))[0]
    if number in {4, 5} and wire_type == 0:
        return _as_int(value)
    if number == 6 and wire_type == 0:
        return _unzigzag(_as_int(value))
    if number == 7 and wire_type == 0:
        return bool(_as_int(value))
    message = f"Unsupported MVT value field {number}/{wire_type}."
    raise AssertionError(message)


def _protobuf_fields(data: bytes) -> Iterator[tuple[int, int, ProtoValue]]:
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 0x7
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            value, offset = data[offset : offset + 8], offset + 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            value, offset = data[offset : offset + length], offset + length
        elif wire_type == 5:
            value, offset = data[offset : offset + 4], offset + 4
        else:
            message = f"Unsupported protobuf wire type {wire_type}."
            raise AssertionError(message)
        yield number, wire_type, value


def _packed_varints(data: bytes) -> Iterator[int]:
    offset = 0
    while offset < len(data):
        value, offset = _read_varint(data, offset)
        yield value


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _unzigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _as_bytes(value: ProtoValue) -> bytes:
    assert isinstance(value, bytes)
    return value


def _as_int(value: ProtoValue) -> int:
    assert isinstance(value, int)
    return value
