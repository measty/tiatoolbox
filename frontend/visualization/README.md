# TIAToolbox visualization frontend

This directory contains the source for the TypeScript/OpenLayers viewer. The
production build is committed under `tiatoolbox/data/visualization/app` so that
Python wheels and source distributions do not need Node.js at install time.

## Development

Use Node.js 22.12 or newer (Node.js 24 is used for development):

```console
npm ci
npm run dev
```

The Vite server proxies `/api` and legacy `/tileserver` requests to
`http://127.0.0.1:5000`. Run the TIAToolbox visualization API on that port.

Before committing changes:

```console
npm run typecheck
npm test
npm run build
```

The final command replaces the committed package data in
`tiatoolbox/data/visualization/app`. The application is designed to be served
at `/viewer/`; Vite emits relative asset URLs so it can also run beneath another
prefix.

## API contract

The viewer starts with `GET /api/v1/bootstrap`. Its `catalog` contains opaque
slide and overlay resource IDs, while `session` contains the currently loaded
slide, annotation stores, and raster layers. The browser changes that state
through `PUT /api/v1/session/slide`, `POST /api/v1/session/overlays`, and
`DELETE /api/v1/session/overlays/{loadedId}`. Selecting a slide resets its
dependent overlays.

Catalog overlay IDs identify importable resources. Once imported, an annotation
store has a separate loaded store ID used by vector-tile, feature-detail, status,
and removal requests. The frontend keeps these identities separate. A store in
`building` or `not-built` LOD state is polled at `GET /api/v1/stores/{loadedId}`
until it becomes `ready` or `failed`; its status-bearing tile URL then replaces
the temporary overview source.

The client accepts snake-case or camel-case JSON keys, but the canonical
contract is:

- Slide: `id`, `name`, `revision`, `dimensions`, `mpp`, `objectivePower`,
  `tileMatrix`, `tileUrl`, and `associatedOverlays`.
- Store: `id`, `name`, `revision`, `featureCount`, `bounds`, `geometryTypes`,
  `properties`, `representations`, `budgets`, `lodStatus`, `tileMatrix`, and
  `urls` (`tiles`, representation templates, labels, and exact feature).
- Raster layer: `id`, `name`, `revision`, `layerName`, `dimensions`, and
  `tileUrl`.
- Exact feature: a GeoJSON Feature whose numeric MVT ID is `id` and whose
  canonical annotation key is `properties._tiatoolbox.canonicalId`.

Resource IDs are opaque. Filesystem paths must not be exposed to the client.
Vector-tile templates use `{z}`, `{x}`, and `{y}`. Exact feature templates use
`{featureId}` (the earlier `{fid}` alias is also accepted). A raster template can
use ordinary XYZ placeholders or Zoomify's
`{TileGroup}/{z}-{x}-{y}` form.

Coordinates are baseline WSI pixels with a top-left origin, x increasing right,
and y increasing down. OpenLayers uses `[x, -y]`; z=0 is the overview and
z=`max_zoom` is baseline resolution. Annotation MVTs use an extent of 4096 and
must align with the slide's advertised tile grid.

The `auto` representation advertises a contiguous `store-zoom` policy. Every
tile at one source zoom uses the same base family and the same deterministic
screen-area rule; for a max-zoom-9 cellular store the default ranges are
aggregate z0-z3, centroid z4-z6, and polygon z7-z9. In the aggregate and
centroid ranges, a polygon whose projected area is at least 36 CSS pixels
squared remains a simplified polygon. This preserves visible glands and other
larger structures while cell-scale objects still use the bounded base family.
Promoted annotations are removed from their aggregate/centroid representation,
so they are neither duplicated nor selected according to tile density.
Canvas and WebGL continue requesting only the `auto` URL. They use separate,
band-limited OpenLayers sources at the advertised boundaries so a loaded
aggregate parent cannot temporarily fill a missing centroid tile (or a
centroid parent a polygon tile).
Polygon MVTs use a deterministic largest-first feature order, so Canvas and
WebGL paint smaller cell-scale polygons above larger structural polygons in
every tile. The tile-contract token in annotation URLs changes when encoded
tile semantics change, without requiring the reusable LOD sidecar to rebuild.

Session slide and raster-overlay tile templates carry a committed slide
generation as well as the opaque resource ID and revision. Every slide
selection advances that generation, including reselecting the same slide, so
browser caches and in-flight requests from an older OpenLayers map cannot feed
tiles into the replacement map. Concurrent slide opens commit in request order:
a slower superseded selection cannot overwrite the newest choice.

Client styling and filtering assume that each MVT contains the properties
listed as styleable in `properties`. Palette, visibility, opacity, and value-range
changes do not replace the tile source or issue new requests. Selecting a new
data property requests only the fields needed for that presentation rather than
embedding the full schema in every tile. Exact GeoJSON geometry is loaded
only after picking a compact MVT feature ID. Aggregate overview cells have no
authoritative feature ID and are intentionally not selectable.

Each annotation layer's **Low zoom** control can retain those aggregate dots
(the default) or hide only aggregate point primitives. Promoted screen-visible
polygons remain visible in either mode, and suppressing aggregates is a local
style change: it does not request raw annotations at overview levels or change
the server's bounded representation policy. The setting is independent per
comparison view and is included in exported viewer configurations. The detail
transition itself is intentionally not movable below the manifest boundary:
doing that without a different precomputed LOD would reintroduce expensive
overview store queries.

Centroids render as fill-only dots; polygon outlines do not apply to them. The
**Dot size** setting controls the radius at the most detailed centroid zoom,
with the renderer reducing it linearly to half that radius at the lowest
centroid zoom. Aggregate dots retain their fixed overview size.

The per-feature colour option is exposed only when the manifest contains an
exact `color` property. Legacy float/byte RGB arrays and strict CSS hex/RGB
values are normalized by the backend; invalid or absent values use a neutral
fallback. Imported direct-colour presentations also fall back safely when the
selected store does not support them.

Use `?renderer=canvas` or `?renderer=webgl` to force a renderer during
benchmarking. `auto` uses WebGL when available and falls back to Canvas. The
WebGL vector-tile API is experimental in OpenLayers 10.9.0 and remains isolated
behind the annotation renderer interface.

OpenLayers 10.9.0 has a renderer crash when an empty vector tile reaches the
WebGL tile-mask pass. `patches/ol+10.9.0.patch` is an exact runtime backport of
[OpenLayers PR #17503](https://github.com/openlayers/openlayers/pull/17503) and
is applied by `patch-package` after installation. Remove the patch when the
OpenLayers dependency includes that fix, but retain the empty-batch regression
test. The `pretest` and `prebuild` checks also execute the patched runtime
contract, so an install performed with lifecycle scripts disabled cannot
silently produce an unsafe viewer bundle.

The OpenLayers map queue caps concurrent tile loads at 16 across raster and
vector sources. Repeated requests for the same annotation tile supersede older
fetches, transient failures and empty HTTP responses receive two short retries,
renderer disposal aborts all outstanding client work, and a representation-band
change aborts outgoing browser requests. The backend independently limits each
annotation source to one active cold tile build while cache hits bypass that
limit. This bounds pressure, but an aborted HTTP request does not cooperatively
stop SQLite/encoding work already executing in Flask.

Each map view has a collapsible overview map backed by the same slide source and
projection. **Save view as PNG** waits for a complete render and composites the
main map canvases in layer order, including visible raster, vector, and selection
layers. Tile sources use anonymous CORS mode so export remains origin-safe; the
viewer reports a clear error if an external server does not permit canvas use.

## Viewer configuration

**Export JSON** and **Import JSON** round-trip the phase-3 core viewer state. The
document schema is `tiatoolbox.viewer`, currently at version `1`:

```json
{
  "schema": "tiatoolbox.viewer",
  "version": 1,
  "slideResourceId": "opaque-slide-id",
  "overlayResourceIds": ["opaque-overlay-id"],
  "renderer": "auto",
  "compare": { "enabled": false, "linked": true },
  "views": {
    "primary": [
      {
        "kind": "annotation",
        "resourceId": "opaque-overlay-id",
        "presentation": {}
      }
    ],
    "secondary": []
  }
}
```

`overlayResourceIds` records session load order. Each view array independently
records layer order and either the complete annotation presentation (including
categorical visibility, numeric range filters, and legacy per-feature `color`
mode, plus low-zoom aggregate visibility) or raster visibility/opacity.
Imports validate IDs against the current server catalog, reject unsupported
versions and oversized documents, deduplicate IDs, and bound untrusted numeric,
colour, property-name, and category values. The format contains opaque resource
IDs only and never filesystem paths or loaded-store cache IDs.

Version 1 deliberately excludes center, zoom/resolution, and rotation. Those
values currently belong to the live OpenLayers `View` instances inside
`SlideMap`; applying them safely requires a projection-aware camera snapshot and
restore contract coordinated with linked-view initialization. They should be
added in a later schema version with that explicit contract, rather than raced
against map construction in version 1.
