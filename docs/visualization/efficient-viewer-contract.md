# Efficient viewer phase 0-4 contract

This document is the acceptance contract for the TIAToolbox visualization
rewrite through phase 4. It describes externally visible behaviour and hard
performance invariants. It deliberately does not prescribe a particular HTTP
framework, vector-tile encoder, or GPU renderer.

The terms **must**, **should**, and **may** are normative. A phase is complete
only when all of its **must** requirements have automated coverage or a recorded
exception approved by the maintainers.

## Scope and phases

| Phase | Outcome                                                                                                                            |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Freeze the functional contract, fixtures, measurements, and benchmark scenarios.                                                   |
| 1     | Introduce a renderer-neutral, revisioned backend for slide and annotation data while retaining the legacy viewer as a fallback.    |
| 2     | Compare candidate Canvas and WebGL renderers with the same protocol and select a default plus fallback.                            |
| 3     | Deliver the core JavaScript viewer and the important existing viewer interactions.                                                 |
| 4     | Deliver bounded-cost LOD, caching, native or batched encoding, dense-segmentation data tiles, cancellation, and performance gates. |

Phases 0-4 cover the efficient core viewer. Model inference, editing, graph
visualization, multichannel mixing, and registration are intentionally deferred
until that core has been assessed on representative hardware and data.

## Required functionality through phase 4

The following functionality must be available in the new viewer before phase 4
is considered complete.

| Area               | Required behaviour                                                                                                                                                                      | Acceptance evidence                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Application        | Start one user-facing viewer URL, report actionable load errors, and cleanly release readers/workers on shutdown.                                                                       | API integration and repeated start/stop tests.                          |
| Slides             | Open and switch supported WSI files without reopening the same slide separately in the UI process.                                                                                      | Reader-count/resource test and browser test.                            |
| Navigation         | Pan, zoom, reset, fit-to-slide, rotate, fullscreen, overview map, slide coordinates, and physical scale where metadata permit it.                                                       | Browser interaction tests at a fixed viewport.                          |
| Base imagery       | Display WSI raster tiles without seams or a vertical-axis inversion and preserve existing associated-image/raster-overlay use cases.                                                    | Known-coordinate and visual-regression tests.                           |
| Layer model        | Display multiple raster and annotation layers with independent order, visibility, and opacity.                                                                                          | State unit tests and browser tests with at least two annotation layers. |
| Store discovery    | Return bounds, count, geometry types, property schema, categorical summaries, numeric ranges/histograms, and a stable revision without materializing the entire store for each request. | Manifest API and query-path tests.                                      |
| Annotation styling | Support categorical colors, continuous colormaps, direct color properties, type visibility, fill/outline controls, edge width, and layer opacity without rebuilding geometry tiles.     | Renderer tests and a zero-network style-change assertion.               |
| Filtering          | Support common categorical/numeric filters in the client and selective server-side store filters when they materially reduce the result.                                                | Filter semantic tests and cache-key tests.                              |
| Inspection         | Hover or click an annotation, resolve a compact tile ID to its canonical UUID, and display exact source properties and geometry.                                                        | Known-ID browser/API test.                                              |
| Linked views       | Synchronize view position/zoom/rotation across two views while keeping layer state independent.                                                                                         | Browser test.                                                           |
| Configuration      | Load initial slide, overlays, view, styles, and filters from a documented configuration representation.                                                                                 | Configuration round-trip test.                                          |
| Export             | Capture a screenshot containing the base image and visible overlays.                                                                                                                    | Browser test with a non-empty image assertion.                          |
| Compatibility      | Keep the legacy raster-annotation path usable as a fallback during migration.                                                                                                           | Existing TileServer tests remain green until formal deprecation.        |
| Renderer fallback  | Run with the stable Canvas/hybrid renderer when WebGL is missing, disabled, or fails initialization.                                                                                    | Forced-capability-failure browser test.                                 |
| WebGL              | Offer a GPU renderer if phase-2 measurements show a material end-to-end benefit without semantic differences. The backend protocol must not depend on it.                               | Renderer-parity and fixed-hardware performance reports.                 |

Blur and other effects that cannot initially be reproduced by the vector/GPU
path may use the raster fallback. The UI must make the chosen representation
predictable; it must not silently change annotation semantics.

## Explicitly deferred functionality

The following items are valuable but do not block phase 4. They remain supported
by the legacy viewer until their later migration where practical.

| Deferred area        | Later work                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Model execution      | Asynchronous inference jobs, progress, cancellation, and result-layer creation.                                           |
| Annotation editing   | Drawing, point/box selection, exact geometry editing, transactional commit, conflict handling, and save/export workflows. |
| Graph overlays       | Tiled nodes/edges, graph-specific selection, and graph hover details.                                                     |
| Multichannel slides  | Client shader channel mixing, per-channel ranges/colors, and enhancement controls.                                        |
| Registration         | Interactive transforms, linked registered overlays, and transform persistence.                                            |
| Specialized analyses | HoVer-Net, NuClick, SAM, or other model-specific controls.                                                                |

Deferral does not authorize removal from the legacy viewer during phases 0-4.

## Protocol and representation contract

### Coordinate system

- API bounds and exact annotation geometries must use level-0 slide pixel
  coordinates.
- Tile matrix orientation, origin, extent, buffer, and zoom-to-downsample mapping
  must be documented in the API schema and covered with asymmetric fixtures.
- Quantization must be deterministic. Adjacent tiles must agree at shared edges.
- MVT feature IDs must be compact unsigned integers that are unique within a
  store revision. Canonical UUIDs are retrieved from the detail endpoint instead
  of repeated in every tile.

### Revisions and immutability

- Every store representation must be addressed by a stable store identifier and
  revision.
- Identical source content and preprocessing settings must produce the same
  revision. A geometry, property, schema, or representation-affecting setting
  change must produce a different revision.
- Revisioned tile URLs must be immutable and safe for long-lived HTTP caches.
- Viewing an existing SQLiteStore must use read-only/query-only connections and
  must not change its bytes, timestamps, journals, or optimization state.
- Mutable edits should eventually use a separate WAL-backed delta layer; they
  must not invalidate an immutable dense base store after every edit.

### Representation selection

The automatic representation is a revisioned, whole-store zoom schedule. A
given source zoom must use one deterministic representation rule for every
tile, independent of tile density, requested style fields, and filters. The
manifest advertises the base schedule as contiguous `store-zoom` ranges and
the `auto` endpoint remains the authority that applies it. At aggregate and
centroid zooms, larger polygons may remain simplified polygons when their
stored area projects to at least a fixed screen-area threshold. This decision
is a pure function of feature area and zoom, is advertised in the manifest,
and must remove the feature from its base representation to avoid duplicates.
Within a store tile, polygon features are emitted from largest to smallest with
a stable ID tie-break, so smaller structures are painted above larger
structures consistently across independently queried tiles.

| Regime                             | Preferred representation                                                                      |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| Whole-slide                        | Persisted density/count/class-composition aggregates.                                         |
| Intermediate                       | Centroids, or optional class/scalar data tiles.                                               |
| Mixed-scale structures             | Simplified polygons for screen-visible structures over the scheduled aggregate/centroid base. |
| Cell-detail                        | Simplified or full polygon MVT within budget.                                                 |
| Selected/editing                   | Exact source geometry in a small ordinary vector layer.                                       |
| Dense non-overlapping segmentation | Optional class, scalar, and annotation-ID data-tile pyramid, with polygons at close zoom.     |

Normal feature/vertex/byte targets are calibrated for pathology annotations;
larger absolute safety limits remain bounded. A tile that exceeds an absolute
limit must return an explicit 422 failure without silently changing geometry family. This
prefers an occasional missing/error tile over a misleading centroid/aggregate
or polygon/centroid patchwork. The default schedule must be derivable from the
slide matrix and inexpensive manifest settings; it may not require a whole-
store density scan before the first useful view.

### Styling and filtering

- Presentation-only state, including palette, opacity, fill, polygon outline,
  centroid-dot size, aggregate visibility, and class visibility, must not be
  part of geometry tile URLs or cache keys. Aggregate suppression must retain
  promoted polygon structures at overview zooms.
- Common property values needed for client styling may be included in tiles, but
  payload properties must be explicitly allow-listed.
- Server-side filters form part of a tile cache key only when they alter the
  returned geometry/properties.
- A style-only interaction must issue zero annotation geometry/data requests.

### Cache and request lifecycle

- Cache keys must include store ID, revision, representation policy and
  budgets, matrix/category contract, schema/property projection,
  geometry-affecting filter, and tile coordinates.
- Public tile URLs must include the tile-contract revision independently of the
  LOD revision, so renderer-affecting byte changes invalidate browser/CDN tile
  caches without forcing an otherwise unnecessary LOD rebuild.
- Memory caches must be bounded by bytes. Persistent caches must be revisioned
  and atomically published.
- Concurrent misses for one key should be coalesced into one build.
- Active cold builds must be bounded per annotation source so a viewport burst
  cannot turn SQLite/encoder contention into lower total throughput.
- Obsolete viewport requests must be abortable. Cancelled queued work should not
  enter expensive encoding, and stale results must not replace current tiles.
- Responses must provide stable ETags and honour conditional requests.

## Hard correctness gates

These gates run in normal pull-request CI and must not depend on wall-clock
timing.

1. **No overview full scan.** An interactive overview request must use persisted
   aggregate/LOD data and may not query or hash every source annotation.
1. **Bounded output.** Every successful tile respects the configured feature,
   vertex, and byte limits and reports the representation actually used;
   failures emit no partial or cross-family tile bytes.
1. **Deterministic LOD.** Dense and sparse tiles at the same source zoom use the
   same advertised base family and feature-area promotion rule; neither
   preflight nor post-encode overflow may change one tile to a cheaper family.
1. **Style independence.** Client-supported style changes cause no annotation
   tile request and do not change tile cache keys.
1. **Identity.** Every displayed/picked compact ID resolves to the exact canonical
   annotation for that store revision.
1. **Geometry.** Independently decoded tiles have the correct Y orientation,
   clipping, holes/rings, buffer behaviour, and tile-edge continuity.
1. **Read-only viewing.** Repeated and concurrent viewing leaves an immutable
   SQLiteStore unchanged and releases all connections.
1. **Cache correctness.** Hits are byte-identical, conditional requests return
   304, revisions invalidate old content, and simultaneous misses are
   coalesced.
1. **Renderer semantics.** Canvas and any selected WebGL renderer expose the same
   visible feature set, styles, filters, and picked IDs within documented
   rasterization tolerances.
1. **Fallback.** A WebGL initialization/context failure falls back to Canvas
   without losing state or requiring a server restart.
1. **Resource stability.** Repeated view creation/destruction and rapid
   navigation do not leak readers, connections, workers, or unbounded cache
   memory.

## Performance measurements and gates

Wall-clock measurements are split from ordinary unit tests:

- Pull requests enforce structural limits and publish benchmark results.
- A dedicated CPU job compares a change with its merge base on the same runner.
- Browser FPS, long-frame, GPU-memory, and sustained-navigation gates run on
  fixed, recorded hardware. Software-rendered hosted CI is used for semantics,
  not GPU performance claims.

Each result must record the commit, OS, CPU, Python, SQLite, geometry/encoder
versions, browser, GPU renderer, dataset fingerprint, tile budgets, and cache
state. Report query, geometry preparation, encoding, compression, total latency,
candidate/features/vertices, response bytes, cache status, and batch throughput.
Use at least two warmups and twenty samples for a p95 claim.

A timing regression becomes a hard failure only when both conditions hold:

1. It is more than 20-25% slower than the merge-base result on the same worker.
1. It exceeds the agreed absolute service-level target.

Initial engineering targets for the supplied 627,761-polygon store are:

| Scenario                                                  |                       Initial target |
| --------------------------------------------------------- | -----------------------------------: |
| Warm manifest                                             |                        10 ms or less |
| Persisted overview, application-cold / warm               |                75 ms / 10 ms or less |
| Dense approximately 1,760-feature polygon tile, p50 / p95 |               80 ms / 150 ms or less |
| Dense approximately 224-feature tile, p50 / p95           |                25 ms / 50 ms or less |
| Memory-cache response p95                                 |                         5 ms or less |
| Dense compressed tile without UUID strings                |                       60 KiB or less |
| Client style interaction                                  | 50 ms or less and zero tile requests |
| Picking                                                   |                       100 ms or less |
| First complete local overlay viewport                     |                        1.5 s or less |
| Fixed-GPU steady pan                                      |                    55 FPS or greater |
| Settled heap growth after ten minutes                     |                          20% or less |

These values are starting targets, not portable assertions for shared CI. They
must be reviewed after the native/batched encoder and renderer spike are
available.

## Phase completion gates

### Phase 0

- This contract is reviewed.
- Small, deterministic, and large dataset profiles are documented.
- The backend benchmark runner emits machine-readable results without importing
  an implementation-specific service module.

### Phase 1

- Manifest, details, revisioned tile/data endpoints, read-only connections,
  compression, ETag, and cache semantics satisfy the correctness gates.
- The legacy raster endpoint remains functional.

### Phase 2

- Canvas, candidate WebGL, and any deck.gl adapter consume identical protocol
  responses in a fixed scenario.
- The renderer decision records end-to-end decode, preparation, upload, frame,
  picking, memory, and fallback results. Backend timings alone are insufficient.

### Phase 3

- Every item in the required-functionality table has a passing unit, API, or
  browser test.
- Linked views and multiple annotation layers work without shared-style state.

### Phase 4

- Overview work is independent of total annotation count.
- Detail tiles, data tiles, caching, request cancellation, and renderer fallback
  pass all hard gates.
- The small sample passes pull-request browser tests and the supplied large store
  has a recorded benchmark report against the initial targets.
- Core performance and memory are judged acceptable before specialized features
  begin migration.
