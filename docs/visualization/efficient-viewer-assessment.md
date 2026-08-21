# Efficient visualization rewrite: phase 0-4 assessment

This document records the state and rationale of the efficient visualization
work through phase 4. It complements the normative
[phase contract](efficient-viewer-contract.md): the contract describes the
target, while this assessment distinguishes implemented functionality,
deliberate exceptions, and measurements still needed before making the new
viewer the default.

## Executive assessment

The important architectural change is not Bokeh-to-OpenLayers by itself. It is
moving annotation rendering from a server-generated image into a bounded,
revisioned data pipeline. WSI and raster overlays remain raster tiles;
annotation stores are exposed through scheduled vector representations, compact
identifiers, exact-detail endpoints, and caches. This removes presentation
choices such as colour, opacity, or category visibility from the expensive
geometry-generation path.

The implementation is ready for wider functional and fixed-hardware assessment
as an opt-in viewer. Integrated tests and a manual browser pass now cover the
core workflow, and CPU-side measurements on the supplied 627,761-polygon store
show that the bounded vector path is viable. It is not yet justified as a
replacement for every Bokeh workflow. In particular, the browser pass proves
functional Canvas/WebGL parity, but no controlled fixed-GPU result establishes
a WebGL frame-rate or end-to-end speed advantage. WebGL is therefore a
candidate renderer, with Canvas retained as the semantic baseline and fallback.

## Architecture delivered

```text
OpenLayers application (/viewer/)
  |-- WSI and image overlays ------------------> raster tile routes
  |-- session/catalogue ------------------------> /api/v1
  `-- annotation layers ------------------------> immutable MVT/data-tile URLs
                                                    |
                          read-only AnnotationStore query
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
          aggregate LOD           centroids          clipped polygons
                 |                    |                    |
                 +-------- hard feature/vertex/byte budgets
                                      |
                    memory + persistent tile caches
                                      |
                      Canvas or experimental WebGL
```

The principal design properties are:

- The existing Bokeh viewer remains the default compatibility path. The new
  viewer is selected explicitly with `--ui openlayers` and is served from a
  single origin.
- `/api/v1` uses opaque, root-confined resource IDs and session-scoped loaded
  layers. Local filesystem paths are not part of the browser contract.
- SQLite-backed stores are opened read-only for visualization. Queries stream
  thin records instead of materialising full `Annotation` objects when that is
  unnecessary.
- A persisted manifest and LOD sidecar provide counts, bounds, property
  summaries, histograms, and pre-aggregated overview data. LOD construction is
  asynchronous and its status is visible to the client.
- Annotation URLs include a stable store revision and a representation-pipeline
  version. SQLite revisions use structural database-header state rather than
  modification time alone. Successful immutable manifests, feature details,
  and tiles support conditional ETags; temporary `building` responses are not
  cached as immutable content.
- The automatic path uses one immutable base geometry family and deterministic
  feature-area rule per store and source zoom. For the supplied max-zoom-9
  slide the base schedule is aggregate z0-z3, centroid z4-z6, and polygon
  z7-z9; tile density, style fields, and filters cannot alter it. In the first
  two ranges, polygons projecting to at least 36 CSS pixels squared remain
  simplified polygons, preserving larger tissue structures without turning
  dense and sparse tiles into different LOD regimes or duplicating features.
- Pathology-calibrated operating targets are 32,000 centroid features and 256
  KiB compressed. Larger absolute feature, vertex, and byte limits remain as
  safety stops. Exceeding one produces an explicit error instead of silently
  degrading a single tile and creating a misleading patchwork; the HTTP API
  reports this as a non-retryable 422 response.
- Ready low-zoom categorical filters use the persisted LOD index. Unsupported
  overview filters fail fast instead of falling back to a whole-store scan.
- MVTs carry compact numeric feature IDs and only requested style properties.
  Exact geometry, canonical annotation key, and complete properties are fetched
  after selection. Aggregate overview cells intentionally have no authoritative
  annotation ID and cannot be selected as if they were cells.
- Tile construction is protected by single-flight work coalescing, at most one
  active cold builders per annotation source, and bounded memory/persistent
  caches. The browser caps annotation requests at 16 and aborts superseded
  requests where possible; already-running Flask work is not cooperatively
  cancellable.
- The browser renderer is behind a small adapter. Canvas and WebGL consume the
  same coordinates, manifests, tile URLs, style state, and inspection API; the
  backend does not know which renderer is active.

This shape addresses the dominant scaling failure of the old path: every style
variant no longer requires SQLite query, geometry conversion, OpenCV drawing,
image encoding, transfer, and image decoding again.

## Phase status

| Phase                       | Current assessment                                                                                                                                                                                                                                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0: contract and baselines   | Delivered. The functional contract, dataset tiers, HTTP scenarios, and initial large-store observations are recorded.                                                                                                                                                                                                                                    |
| 1: renderer-neutral backend | Delivered for assessment. The versioned API, resources, sessions, raster tiles, read-only store access, manifests, exact feature lookup, and revision-aware responses are implemented while the Bokeh route remains available.                                                                                                                           |
| 2: renderer comparison      | Candidate implementations and manual semantic smoke tests are delivered, but renderer selection is not concluded. Canvas and WebGL can be forced against the same protocol; controlled frame-time, GPU-memory, and context-loss results remain outstanding.                                                                                              |
| 3: core JavaScript viewer   | The main viewing workflow is delivered. It intentionally does not yet have complete Bokeh feature parity; exceptions are listed below.                                                                                                                                                                                                                   |
| 4: bounded dense-store path | Delivered for assessment: uniform store-zoom LOD, background overview construction, budgets, compact data, caching, bounded client/server cold work, and batched/custom vector encoding. Initial integrated large-store CPU measurements are recorded below; broader representative-store, fixed-host browser, memory, and endurance acceptance remains. |

## Core functionality preserved in the OpenLayers viewer

The current viewer supports:

- discovering, opening, and switching slides, with tiled WSI imagery;
- loading multiple annotation stores and raster overlays;
- independent layer visibility and opacity;
- constant, categorical, and numeric colour modes, category visibility,
  numeric range filtering, fill opacity, polygon outline width, centroid-dot
  sizing, and independent low-zoom aggregate suppression;
- changing client-side presentation without replacing geometry tile sources;
- pan, zoom, rotation, fit-to-slide, fullscreen, pointer coordinates, and a
  physical scale line where metadata allow it;
- annotation selection with bounded local/server picking followed by exact
  feature inspection;
- one or two views, optional linked pan/zoom/rotation, and independent layer
  presentation in each view;
- explicit `Auto`, `Canvas`, and `WebGL` renderer choices;
- a collapsible overview map, composited PNG export, and versioned JSON
  configuration import/export;
- direct per-feature `color` values when the store advertises that property;
  and
- visible background LOD status with source refresh when an immutable ready
  revision becomes available.

The server also has a normalized, allow-listed property-filter representation
and includes filter state in relevant cache keys. The new UI deliberately uses
structured categorical and numeric controls rather than accepting the legacy
free-form AnnotationStore expression language.

## Known exceptions and deferred specialist functionality

The following are current parity gaps rather than performance regressions:

- no grid overlay or baseline-pixel versus MPP axis toggle (coordinates, an
  overview map, and a metric scale line are present);
- no secondary type-specific colour rule or general free-form AnnotationStore
  filter text box;
- version-1 configuration intentionally excludes camera center, resolution,
  and rotation;
- combined raster/vector ordering is raster-first and has no reorder UI, even
  though order within each kind is preserved in configuration;
- no user-facing equivalent of every legacy blur or `max-scale` setting; LOD
  and representation selection replace the common dense-overlay use case, and
  effects such as blur can continue through the raster fallback;
- aggregate overviews preserve the configured categorical property, but cannot
  preserve arbitrary numeric styling semantics; overview dots can be hidden
  per layer, but the UI does not otherwise expose the server's selected
  polygon/centroid/aggregate representation;
- the structured server-filter API is not selected by the UI when it could
  reduce geometry transfer; and
- dense label/data-tile service support is present in the backend but needs
  browser acceptance before being treated as a finished user workflow.

These should be prioritized after phase-4 performance and core usability are
validated. They do not require reopening the protocol boundary.

The following specialized areas were deliberately deferred and remain reasons
to retain the Bokeh viewer during migration:

- model execution and model-specific controls such as HoVer-Net, NuClick, and
  SAM;
- annotation drawing, editing, committing, and saving;
- graph overlays and graph-specific inspection;
- multichannel image mixing and channel controls;
- registration transforms and registration-specific comparison controls; and
- WSI-native label/macro associated images (external raster overlays are
  already supported).

Two operational hardening items also remain before a long-running hosted
deployment: session/source/reader/cache lifetime needs TTL or reference-counted
eviction, and a runtime `webglcontextlost` event does not yet migrate a live map
to Canvas with state preserved. Initialization and constructor failures do
fall back safely.

## WebGL decision

WebGL is useful enough to keep, but current evidence does not support declaring
it universally faster or making an unconditional performance promise.

It is most likely to help when many vector primitives remain visible after LOD,
when the user pans continuously, or when styles change repeatedly and GPU-side
buffer reuse dominates. It does not make SQLite queries, geometry clipping,
MVT encoding, compression, or network transfer faster. For cold tiles those
server stages may dominate, and aggressive LOD can make Canvas sufficiently
cheap at overview scales.

`Auto` currently checks for a WebGL implementation without a reported major
performance caveat and tries the OpenLayers WebGL vector-tile layer. That policy
is provisional while the entire OpenLayers viewer is opt-in; it should be
revisited before the viewer becomes the default. Creation failure falls back to
Canvas. Runtime context loss is not yet handled. The OpenLayers 10.9.0 WebGL
vector-tile API is experimental, so the adapter boundary and Canvas fallback
should be retained even if later measurements favour WebGL.

The phase-2 decision should compare both renderers on the same browser build,
GPU, viewport, view path, tile cache state, layer state, and dataset. Record
frame-time distributions and long frames during pan/zoom, time to first useful
overlay, style-change latency, request count/bytes, memory, correctness, and
recovery from a lost/unavailable WebGL context. Average FPS alone is not an
adequate acceptance measure. The manual browser pass verified that both paths
render the same WSI/raster/vector workflows, but no FPS or WebGL speed-up claim
has been established for this work.

## Supplied large-store observations

The supplied tier-D case is
`04-09-142.454.EX.1A.db` with its matching slide. The store is approximately
404.6 MiB and contains 627,761 polygons and 27,736,168 coordinates. Observed
vertices per polygon were mean 44.18, median 39, p95 65, p99 263, and maximum
783\. The slide is 106,496 x 85,248 pixels at 40x with mean MPP approximately
0.22706. The observed `type` counts were 303,511 (0), 49,190 (1), 242,899 (2),
6,053 (3), and 26,108 (4).

Exploratory measurements of the old raster path provide motivation, not
acceptance numbers for the integrated rewrite. On one dense downsample-8 tile
with about 1,760 candidates, representative stages were approximately 2.8 ms
for the sorted database fetch, 3.3 ms decompression, 3.2 ms JSON parsing,
17.5 ms object materialisation, 77.7 ms OpenCV drawing, and 13.4 ms WebP
encoding. A separate run had roughly 75 ms drawing and 37 ms image encoding.
An empty whole-slide downsample-512 request still took about 1.07 s because it
scanned broadly. A property query was about 948 ms and property-dictionary
enumeration about 5.5 s.

An early dynamic Python MVT experiment showed why vector tiles need LOD and
caching rather than merely changing the response format:

| Request            |             Features / payload |          Cold |    Warm cache |
| ------------------ | -----------------------------: | ------------: | ------------: |
| dense z6 x32 y13   | 1,759 / 118,456 B uncompressed |  about 236 ms | about 0.48 ms |
| detail z8 x113 y57 |                 224 / 17,535 B | about 35.7 ms | about 0.47 ms |
| naive overview z2  |                   98 / 3.4 KiB | about 1.685 s |  about 0.5 ms |

The z6 cold result spent about 12.3 ms in query and 223 ms in encoding; the z8
result spent about 1.9 ms in query and 33.2 ms in encoding. Repeating a UUID in
each feature also mattered: the example tile was about 118 KiB with UUID and
type, 58.7 KiB with integer ID and type, and 42.9 KiB with type alone.

The conclusions from that prototype are deliberately narrow: naive cold MVT
generation was not automatically faster than raster drawing; cache hits were
very cheap; compact IDs and minimal properties materially reduced payload; and
low-zoom requests needed precomputed aggregates rather than a source scan.

### Integrated phase-4 observations

The current implementation was then exercised against the same 627,761-polygon
store on a Windows development machine. These are local observations rather
than portable CI thresholds. "Cold tile" below means a new application tile
key with no memory/persistent tile hit; the operating-system page cache was not
cleared.

| Request                           |                                    Output | Server / local HTTP time |
| --------------------------------- | ----------------------------------------: | -----------------------: |
| persisted overview z2 x0 y1       |        200 aggregates, 6,494 B compressed |           14.7 / 17.5 ms |
| dense categorical z6 x32 y13      | 1,843 features, 12,540 vertices, 35,401 B |           73.3 / 80.0 ms |
| detail categorical z8 x113 y57    |     250 features, 4,206 vertices, 8,087 B |           12.9 / 16.0 ms |
| dense numeric-property z6 x32 y13 |                       52,243 B compressed |           76.3 / 79.3 ms |

For the dense categorical request, `Server-Timing` attributed approximately
1.2 ms to selection, 23.2 ms to the indexed query, 47.3 ms to encoding, and
1.4 ms to compression. A 20-tile uncached build sample produced p50 69.7 ms,
p95 94.0 ms, and mean 81.5 ms; it also contained a single approximately 278 ms
outlier. Median query, encode, and gzip stages were about 23.7, 44.3, and 1.5 ms.
This is comfortably below the initial 150 ms dense-tile p95 target for that
sample, but must still be repeated against merge base on a fixed worker.

Persisted LOD construction took approximately 13.8 seconds in isolation;
viewer startup, slide open, and LOD preparation together took approximately
17.8 seconds. Once cached, sequential HTTP samples centered around 2.0-2.6 ms.
The standard-library HTTP probe opens short-lived connections on this Windows
host and showed periodic 15-25 ms connection/scheduling outliers, so its warm
p95 was 15-23 ms even though direct memory-cache lookup was approximately
0.04 ms. A 32-request, concurrency-8 viewport batch was approximately 10-11 ms
p50 and 11-12 ms p95 per request. A fixed-host keep-alive run is still needed
before treating the 5 ms memory-cache HTTP target as accepted.

The aggregate-to-detail handoff was subsequently moved down by one source
level for the supplied max-zoom-9 slide: persisted aggregates cover z0-z3 and
z4 begins the centroid range. An earlier tile-local policy produced 112
centroid responses and 31 aggregate responses across the 143 z4 tiles. That
measurement exposed a correctness problem rather than a useful optimization:
dense and sparse neighbours formed a visually ambiguous patchwork. The current
policy therefore makes every z4 tile use the centroid base rule and raises the
normal centroid envelope to cover the observed buffered maximum of 22,096
cells (the z4 mean was 4,991 and p95 19,043) without adding a density pre-scan.
The later screen-area promotion for larger structures is deterministic across
the store and is not a return to tile-local density switching.

A fresh complete z4 sweep after the uniform-policy change requested all 143
tiles with the UI's `type` field. Every response reported `centroid`; none
reported aggregate or an absolute-limit error. The largest response contained
21,983 points and was 144,972 bytes compressed, below both the normal 32,000-
point/256-KiB envelope and the absolute safety ceiling.

The mixed-scale colorectal store
`B-1914637_01-03_HE_20221018_ihc.db` contains 669,524 polygons. A cold LOD
build with 36-pixel-squared promotion took 13.1 seconds and produced a 1.8-MiB
sidecar. Only 5,718 annotations are promotion candidates at any zoom; the
eligible count rises from 762 at z5 to 2,505 at z6 and 5,718 at z8. A sampled
z5 tile encoded 218 aggregates and 84 structure polygons in 42 ms cold. A
dense sampled z6 tile encoded 26,072 centroids and 67 structure polygons in
1.28 seconds cold, then returned from memory cache in under 0.3 ms. Manual
Canvas and WebGL checks confirmed that gland boundaries survive at both z5 and
z6 while smaller objects follow the aggregate/centroid base schedule, with no
browser warnings or errors.

A deliberately cold 40-distinct-tile experiment showed that unconstrained
parallelism is counterproductive on this path: representative wall times were
about 7.2 seconds with one builder, 12.3 seconds with two, 18.9 seconds with
four, and 24 seconds with 8-16 competing workers. A follow-up viewport run also
found that one builder beat two for first-tile, first-eight-tile, and complete
viewport latency. The service therefore admits one active cold build per
source. Large uncached teleports remain an important
fixed-host workload, and queued WSGI work is not yet cooperatively cancelled
when a browser request is abandoned.

A cold browser pass then traversed every source level from z1 through z7. At
the z4 boundary the WebGL layer showed a temporary blank while cold centroid
tiles arrived, followed by a visually uniform centroid field. At z7 it again
showed a blank rather than substituting centroid parents, then rendered
polygons. Canvas produced the same z4 centroid field, and neither renderer
logged a warning or error. This validates the band-limited source handoff in a
real render, in addition to its tile-grid unit tests.

The manual browser pass covered both Canvas and WebGL, the overview map, slide
switching, a raster overlay, categorical and direct-colour controls, and linked
view behavior. A style-only fill-opacity change caused zero annotation geometry
requests. PNG export produced a non-empty 644,357-byte composite, and JSON
export produced a 1,405-byte version-1 document containing opaque resource IDs,
layer order, renderer, comparison state, and full per-view presentation. These
checks establish functional integration, not frame-time, memory, or endurance
performance.

A subsequent WebGL regression pass reproduced a frozen/offset overlay as the
OpenLayers 10.9.0 empty-vector-tile crash fixed upstream by PR 17503. The exact
upstream change is temporarily backported in the frontend dependency, and a
pre-test/pre-build contract check prevents an unpatched bundle. After also
removing FIFO request eviction and terminalizing cancelled custom tile loads,
rapid pan/zoom across populated and empty regions, WebGL/Canvas renderer
switches during loading, and authoritative feature picking completed with no
new browser warnings or errors on the supplied 627,761-polygon store.

The follow-up browser pass also verified per-layer suppression of aggregate
dots until the first bounded detail zoom. Slide raster URLs now use a fresh
selection generation, resource ID, and revision; traversing the previously
visited zoom levels while switching between both supplied WSIs and back showed
no cross-slide tiles or browser warnings. Server traces confirmed distinct
generations 1, 2, and 3 for that sequence.

The deterministic verification run comprised 344 modern visualization/storage
tests plus 37 legacy TileServer tests, all passing; Ruff also passed. The
frontend passed type checking, 49 Vitest tests, and a production build. Browser
checks are currently manual rather than an automated end-to-end CI gate.

## Benchmark and acceptance method

Use the scenarios and runner described in the
[benchmark README](../../benchmarks/visualization/README.md). Keep results in
three separate groups:

1. application-cold backend requests after a restart/empty application cache;
1. warm backend requests, including a parallel viewport batch; and
1. browser interaction runs for Canvas and WebGL with identical navigation and
   presentation changes.

For latency percentiles, use at least two warmups and 20 measured samples.
Record the commit, TIAToolbox/Python/SQLite/geometry/encoder versions, OS, CPU,
RAM, browser, GPU and driver, dataset fingerprint, tile budgets, cache capacity,
LOD readiness, concurrency, and viewport. "Cold" must state which application
caches were cleared; it must not imply the operating-system filesystem cache
was dropped unless that was explicitly controlled.

Capture per-tile candidate/output feature counts, input/output vertices,
representation, payload bytes, cache outcome, and query/decode/clip/encode/
compression timings where available. Also verify invariants: overview requests
do not scan the full source store, hard budgets are respected, aggregate cells
are not pickable, style-only changes do not refetch geometry, store access is
read-only, and revision/ETag semantics prevent stale cache reuse.

Ordinary CI should enforce deterministic correctness and budget invariants, not
narrow wall-clock thresholds. A performance gate should run the change and its
merge base on the same fixed worker and fail only when both a meaningful
relative regression and an absolute service-level limit are exceeded. The
tier-D store belongs in manual or scheduled fixed-host acceptance rather than
the normal test dependency set.

## Decision point before specialized migration

The initial tier-D CPU results justify proceeding with wider evaluation of the
opt-in viewer; they do not yet justify changing the default UI or promising a
WebGL benefit. Next, automate the browser semantics smoke tests and run the
fixed-GPU Canvas/WebGL, first-viewport, picking, ten-minute heap, and resource-
lifetime matrix. Use those results to choose the `Auto` policy and address any
measured bottleneck--store query, LOD build, encoding, transfer, or rendering--
without changing the renderer-neutral API prematurely. Once those gates hold,
prioritize the specialist parity gaps above.
