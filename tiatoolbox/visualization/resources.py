"""Opaque, root-confined resource registry for the modern viewer API."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ResourceKind = Literal["slide", "annotation", "raster-overlay"]

_SLIDE_EXTENSIONS = {
    ".svs",
    ".ndpi",
    ".mrxs",
    ".tif",
    ".tiff",
    ".jp2",
    ".dcm",
    ".dicom",
    ".zarr",
}
_ANNOTATION_EXTENSIONS = {".db", ".geojson", ".dat"}
_RASTER_OVERLAY_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".svs",
    ".ndpi",
    ".mrxs",
}


@dataclass(frozen=True, slots=True)
class ViewerResource:
    """A server-side file mapping exposed to clients only by opaque ID."""

    id: str
    name: str
    kind: ResourceKind
    path: Path
    revision: str

    def public_dict(self) -> dict[str, str]:
        """Return path-free resource metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "revision": self.revision,
        }


class ResourceRegistry:
    """Discover resources under explicitly configured roots."""

    def __init__(
        self,
        *,
        slide_roots: list[str | Path] | tuple[str | Path, ...] = (),
        overlay_roots: list[str | Path] | tuple[str | Path, ...] = (),
        recursive: bool = False,
    ) -> None:
        """Create a registry and scan configured roots."""
        self.slide_roots = self._normalise_roots(slide_roots)
        self.overlay_roots = self._normalise_roots(overlay_roots)
        self.recursive = recursive
        self._resources: dict[str, ViewerResource] = {}
        self.refresh()

    def refresh(self) -> None:
        """Rescan configured roots without exposing paths."""
        resources: dict[str, ViewerResource] = {}
        for root in self.slide_roots:
            for path in self._files(root):
                if path.suffix.lower() in _SLIDE_EXTENSIONS:
                    resource = self._resource(path, "slide")
                    resources[resource.id] = resource
        for root in self.overlay_roots:
            for path in self._files(root):
                suffix = path.suffix.lower()
                if suffix in _ANNOTATION_EXTENSIONS:
                    kind: ResourceKind = "annotation"
                elif suffix in _RASTER_OVERLAY_EXTENSIONS:
                    kind = "raster-overlay"
                else:
                    continue
                resource = self._resource(path, kind)
                resources[resource.id] = resource
        self._resources = resources

    def get(self, resource_id: str, kind: ResourceKind | None = None) -> ViewerResource:
        """Resolve an opaque ID and optionally enforce its kind."""
        resource = self._resources.get(resource_id)
        if resource is None or (kind is not None and resource.kind != kind):
            msg = "Unknown viewer resource."
            raise KeyError(msg)
        # Re-resolve before use so a symlink replacement cannot escape a root.
        resolved = resource.path.resolve(strict=True)
        roots = self.slide_roots if resource.kind == "slide" else self.overlay_roots
        if not any(resolved.is_relative_to(root) for root in roots):
            msg = "Viewer resource is outside configured roots."
            raise PermissionError(msg)
        return resource

    def catalog(self) -> dict[str, list[dict[str, str]]]:
        """Return public resources grouped by function."""
        resources = sorted(self._resources.values(), key=lambda item: item.name.lower())
        return {
            "slides": [
                item.public_dict() for item in resources if item.kind == "slide"
            ],
            "overlays": [
                item.public_dict() for item in resources if item.kind != "slide"
            ],
        }

    def associated_overlays(self, slide_id: str) -> list[dict[str, str]]:
        """Return overlays whose filename stem begins with the slide stem."""
        slide = self.get(slide_id, "slide")
        stem = slide.path.stem.casefold()
        return [
            resource.public_dict()
            for resource in self._resources.values()
            if resource.kind != "slide"
            and resource.path.stem.casefold().startswith(stem)
        ]

    def _resource(self, path: Path, kind: ResourceKind) -> ViewerResource:
        path = path.resolve(strict=True)
        identity = f"{kind}\0{os.path.normcase(str(path))}"
        digest = hashlib.blake2b(
            identity.encode(),
            digest_size=12,
        ).hexdigest()
        return ViewerResource(
            id=digest,
            name=path.name,
            kind=kind,
            path=path,
            revision=file_revision(path),
        )

    def _files(self, root: Path):  # noqa: ANN202
        iterator = root.rglob("*") if self.recursive else root.iterdir()
        return (path for path in iterator if path.is_file())

    @staticmethod
    def _normalise_roots(
        roots: list[str | Path] | tuple[str | Path, ...],
    ) -> tuple[Path, ...]:
        output = []
        for root in roots:
            path = Path(root).resolve(strict=True)
            if not path.is_dir():
                msg = f"Viewer resource root is not a directory: {path}"
                raise NotADirectoryError(msg)
            output.append(path)
        return tuple(output)


def file_revision(path: str | Path) -> str:
    """Return a cheap immutable-URL revision for a local file.

    The fingerprint deliberately avoids hashing an entire multi-gigabyte WSI or
    store at viewer startup. Size and nanosecond mtime change for ordinary atomic
    updates; schema/cache builders may additionally persist a full content hash.
    """
    path = Path(path)
    stat = path.stat()
    if path.suffix.lower() == ".db":
        with path.open("rb") as handle:
            header = handle.read(100)
        if header.startswith(b"SQLite format 3\x00"):
            # SQLite's file-change counter, database page count, schema cookie,
            # and version-valid-for number change on ordinary committed edits.
            # Unlike mtime, these bytes survive a copy and still detect a
            # same-size edit whose timestamp was deliberately restored.
            structural = header[24:44] + header[92:96]
            token = b"sqlite:" + str(stat.st_size).encode() + b":" + structural
            return hashlib.blake2b(token, digest_size=12).hexdigest()
    token = f"{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.blake2b(token, digest_size=12).hexdigest()
