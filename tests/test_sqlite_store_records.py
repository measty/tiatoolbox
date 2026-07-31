"""Tests for read-only and thin-record SQLiteStore access."""

from __future__ import annotations

import sqlite3
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from tiatoolbox.annotation import Annotation, SQLiteStore
from tiatoolbox.annotation.storage import (
    PROPERTY_FILTER_MAX_DEPTH,
    PROPERTY_FILTER_MAX_IN_VALUES,
    AnnotationRecord,
    canonical_property_filter,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_store(path: Path) -> None:
    """Create a small file-backed annotation store."""
    store = SQLiteStore(path)
    store.append_many(
        [
            Annotation(
                box(0, 0, 4, 4),
                {"class": "small", "score": 0.25, "unused": True},
            ),
            Annotation(
                box(1, 1, 11, 11),
                {"class": "large", "score": 0.75, "unused": False},
            ),
            Annotation(Point(2, 2), {"class": "point"}),
            Annotation(box(100, 100, 110, 110), {"class": "far"}),
        ],
        keys=["small", "large", "point", "far"],
    )
    store.close()


def test_sqlite_store_read_only_is_non_mutating_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    """A read-only store uses SQLite protections and does no close-time writes."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    before = path.read_bytes()

    store = SQLiteStore(path, read_only=True)
    assert store.con.execute("PRAGMA query_only").fetchone() == (1,)
    assert len(store) == 4
    assert set(store.query((0, 0, 20, 20))) == {"small", "large", "point"}

    with pytest.raises(sqlite3.OperationalError, match=r"readonly|read-only"):
        store.con.execute("CREATE TABLE forbidden (id INTEGER)")
    with pytest.raises(PermissionError, match="read_only=True"):
        store.append(Annotation(Point(0, 0), {}))
    with pytest.raises(PermissionError, match="read_only=True"):
        store.commit()

    store.close()
    store.close()
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="closed"):
        len(store)


def test_sqlite_store_read_only_requires_existing_file(tmp_path: Path) -> None:
    """Read-only mode rejects in-memory and missing connection targets."""
    with pytest.raises(ValueError, match="file-backed"):
        SQLiteStore(":memory:", read_only=True)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SQLiteStore(tmp_path / "missing.db", read_only=True)
    empty_path = tmp_path / "empty.db"
    empty_path.touch()
    with pytest.raises(ValueError, match="non-empty"):
        SQLiteStore(empty_path, read_only=True)


def test_sqlite_store_registered_functions_do_not_delay_cleanup(tmp_path: Path) -> None:
    """SQLite callbacks do not keep an otherwise unreferenced store alive."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path)
    store_reference = weakref.ref(store)

    del store

    assert store_reference() is None
    path.unlink()


def test_sqlite_store_main_thread_closes_worker_connection(tmp_path: Path) -> None:
    """A joined worker's connection can be closed by the service thread."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path, read_only=True)

    def query_from_worker() -> tuple[int, int]:
        return threading.get_ident(), len(
            list(store.query_records((0, 0, 20, 20))),
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_id, record_count = executor.submit(query_from_worker).result()

    assert record_count == 3
    assert worker_id in store.cons
    assert worker_id != threading.get_ident()
    store.close()
    assert store.cons == {}
    path.unlink()


def test_query_records_returns_thin_streamed_records(tmp_path: Path) -> None:
    """Thin records contain IDs, bounds, requested properties, and raw WKB."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path, read_only=True)

    records_iterator = store.query_records(
        Polygon.from_bounds(-1, -1, 20, 20),
        selected_properties=("class", "missing"),
    )
    assert iter(records_iterator) is records_iterator
    records = list(records_iterator)

    assert all(isinstance(record, AnnotationRecord) for record in records)
    assert {record.key for record in records} == {"small", "large", "point"}
    assert len({record.id for record in records}) == 3
    assert all(isinstance(record.id, int) for record in records)
    assert all(set(record.properties) == {"class"} for record in records)
    assert all(record.wkb is not None for record in records)
    assert {shapely.from_wkb(record.wkb).geom_type for record in records} == {
        "Point",
        "Polygon",
    }

    by_key = {record.key: record for record in records}
    assert by_key["small"].object_type == "Polygon"
    assert by_key["small"].bounds == pytest.approx((0, 0, 4, 4))
    assert by_key["small"].area == pytest.approx(16)
    assert by_key["point"].object_type == "Point"
    assert by_key["point"].bounds == pytest.approx((2, 2, 2, 2))
    assert by_key["point"].area == 0
    store.close()


def test_query_records_options_and_no_area_ordering(tmp_path: Path) -> None:
    """Geometry/properties can be omitted and limits do not add area sorting."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path)
    statements: list[str] = []
    store.con.set_trace_callback(statements.append)

    records = list(
        store.query_records(
            (-1, -1, 20, 20),
            include_geometry=False,
            min_area=20,
            limit=1,
        ),
    )

    assert len(records) == 1
    assert records[0].key == "large"
    assert records[0].wkb is None
    assert records[0].properties == {}
    record_queries = [
        statement for statement in statements if "FROM rtree" in statement
    ]
    assert len(record_queries) == 1
    assert "ORDER BY" not in record_queries[0].upper()
    store.close()


def test_query_records_legacy_store_without_area(tmp_path: Path) -> None:
    """Legacy stores return an absent area and reject area filtering."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path)
    store.remove_area_column()

    records = list(store.query_records((0, 0, 20, 20), include_geometry=False))
    assert records
    assert all(record.area is None for record in records)
    with pytest.raises(ValueError, match="without an area column"):
        list(store.query_records((0, 0, 20, 20), min_area=1))
    store.close()


@pytest.mark.parametrize(
    ("property_filter", "expected_keys"),
    [
        ({"op": "eq", "property": "class", "value": "small"}, {"small"}),
        (
            {"op": "ne", "property": "class", "value": "small"},
            {"large", "point"},
        ),
        ({"op": "lt", "property": "score", "value": 0.5}, {"small"}),
        ({"op": "lte", "property": "score", "value": 0.25}, {"small"}),
        ({"op": "gt", "property": "score", "value": 0.25}, {"large"}),
        ({"op": "gte", "property": "score", "value": 0.75}, {"large"}),
        (
            {
                "op": "in",
                "property": "class",
                "values": ["point", "small"],
            },
            {"point", "small"},
        ),
        (
            {
                "op": "and",
                "args": [
                    {"op": "gte", "property": "score", "value": 0.25},
                    {
                        "op": "in",
                        "property": "class",
                        "values": ["small", "large"],
                    },
                ],
            },
            {"small", "large"},
        ),
        (
            {
                "op": "or",
                "args": [
                    {"op": "eq", "property": "class", "value": "point"},
                    {"op": "gt", "property": "score", "value": 0.5},
                ],
            },
            {"point", "large"},
        ),
    ],
)
def test_query_records_safe_property_filter_semantics(
    tmp_path: Path,
    property_filter: dict[str, object],
    expected_keys: set[str],
) -> None:
    """The bounded AST provides typed comparison and boolean semantics."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path, read_only=True)

    records = store.query_records(
        (0, 0, 20, 20),
        include_geometry=False,
        property_filter=property_filter,
    )

    assert {record.key for record in records} == expected_keys
    store.close()


def test_query_records_property_filter_is_parameterized(tmp_path: Path) -> None:
    """SQL-like property names and values remain inert bound parameters."""
    path = tmp_path / "annotations.db"
    _make_store(path)
    store = SQLiteStore(path, read_only=True)
    bounds = (0, 0, 20, 20)

    value_injection = {
        "op": "eq",
        "property": "class",
        "value": "' OR 1=1 --",
    }
    property_injection = {
        "op": "eq",
        "property": "class') OR 1=1 --",
        "value": "small",
    }

    assert not list(
        store.query_records(
            bounds,
            include_geometry=False,
            property_filter=value_injection,
        ),
    )
    assert not list(
        store.query_records(
            bounds,
            include_geometry=False,
            property_filter=property_injection,
        ),
    )
    assert len(store) == 4
    with pytest.raises(ValueError, match="Unsupported"):
        list(
            store.query_records(
                bounds,
                property_filter={
                    "op": "eq) OR 1=1 --",
                    "property": "class",
                    "value": "small",
                },
            ),
        )
    store.close()


def test_property_filter_limits_and_canonicalization() -> None:
    """Equivalent boolean filters canonicalize and hostile shapes are bounded."""
    class_filter = {"op": "eq", "property": "class", "value": "small"}
    score_filter = {"op": "gte", "property": "score", "value": 0.25}
    first = {"op": "and", "args": [class_filter, score_filter]}
    second = {"args": [score_filter, class_filter], "op": "and"}
    assert canonical_property_filter(first) == canonical_property_filter(second)

    deep_filter: dict[str, object] = class_filter
    for _ in range(PROPERTY_FILTER_MAX_DEPTH):
        deep_filter = {"op": "and", "args": [deep_filter]}
    with pytest.raises(ValueError, match="maximum depth"):
        canonical_property_filter(deep_filter)
    with pytest.raises(ValueError, match="between 1"):
        canonical_property_filter(
            {
                "op": "in",
                "property": "class",
                "values": list(range(PROPERTY_FILTER_MAX_IN_VALUES + 1)),
            },
        )
    with pytest.raises(ValueError, match="JSON scalars"):
        canonical_property_filter(
            {"op": "eq", "property": "class", "value": {"raw": "SQL"}},
        )
