import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiClient, isAbortError } from "../api/client";
import type {
  BootstrapManifest,
  FeatureDetail,
  LoadedRaster,
  LoadedStore,
  RasterLayerManifest,
  RasterPresentation,
  ResourceSummary,
  SlideManifest,
  StoreManifest,
} from "../api/types";
import { Inspector } from "../components/Inspector";
import { LayerPanel } from "../components/LayerPanel";
import { RasterLayerPanel } from "../components/RasterLayerPanel";
import {
  defaultPresentation,
  presentationForStore,
  refreshProvisionalPresentation,
  type LayerPresentation,
} from "../domain/style-spec";
import {
  createViewerConfig,
  MAX_VIEWER_CONFIG_BYTES,
  parseViewerConfig,
  serialiseViewerConfig,
  type ViewerConfigCatalog,
  type ViewerConfigLayer,
  type ViewerConfigV1,
} from "../domain/viewer-config";
import { SlideMap, type FeaturePick } from "../map/SlideMap";
import { ViewLinkController } from "../map/view-link-controller";
import type { RendererPreference } from "../renderers/annotation-renderer";

import "./app.css";

type LayerView = "primary" | "secondary";

type SessionOverlay =
  | {
      kind: "annotation";
      resourceId: string;
      layerId: string;
      manifest: StoreManifest;
    }
  | {
      kind: "raster";
      resourceId: string;
      layerId: string;
      manifest: RasterLayerManifest;
    };

interface AnnotationViewLayer {
  resourceId: string;
  loaded: LoadedStore;
}

interface RasterViewLayer {
  resourceId: string;
  loaded: LoadedRaster;
}

const DEFAULT_RASTER_PRESENTATION: RasterPresentation = {
  visible: true,
  opacity: 0.65,
};

function initialRenderer(): RendererPreference {
  const value = new URLSearchParams(window.location.search).get("renderer");
  return value === "canvas" || value === "webgl" ? value : "auto";
}

export function App() {
  const [api] = useState(() => new ApiClient());
  const [linkController] = useState(() => new ViewLinkController());
  const [bootstrap, setBootstrap] = useState<BootstrapManifest | null>(null);
  const [slideId, setSlideId] = useState("");
  const [slide, setSlide] = useState<SlideManifest | null>(null);
  const [loadedOverlays, setLoadedOverlays] = useState<
    Record<string, SessionOverlay>
  >({});
  const [primaryOverlayIds, setPrimaryOverlayIds] = useState<string[]>([]);
  const [secondaryOverlayIds, setSecondaryOverlayIds] = useState<string[]>([]);
  const [primaryPresentations, setPrimaryPresentations] = useState<
    Record<string, LayerPresentation>
  >({});
  const [secondaryPresentations, setSecondaryPresentations] = useState<
    Record<string, LayerPresentation>
  >({});
  const [primaryRasterPresentations, setPrimaryRasterPresentations] = useState<
    Record<string, RasterPresentation>
  >({});
  const [secondaryRasterPresentations, setSecondaryRasterPresentations] = useState<
    Record<string, RasterPresentation>
  >({});
  const [pendingOverlayIds, setPendingOverlayIds] = useState<string[]>([]);
  const [activeLayerView, setActiveLayerView] = useState<LayerView>("primary");
  const [renderer, setRenderer] = useState<RendererPreference>(initialRenderer);
  const [viewCount, setViewCount] = useState<1 | 2>(1);
  const [linked, setLinked] = useState(true);
  const [selection, setSelection] = useState<FeatureDetail | null>(null);
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const selectionGenerationRef = useRef(0);
  const slideGenerationRef = useRef(0);
  const configInputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [configNotice, setConfigNotice] = useState<string | null>(null);

  const storesRef = useRef<Record<string, StoreManifest>>({});
  storesRef.current = Object.fromEntries(
    Object.values(loadedOverlays).flatMap((overlay) =>
      overlay.kind === "annotation"
        ? [[overlay.manifest.id, overlay.manifest]]
        : [],
    ),
  );

  useEffect(() => {
    void api
      .bootstrap()
      .then((manifest) => {
        const restored = restoreSessionOverlays(manifest);
        const restoredIds = Object.keys(restored);
        const annotationPresentations: Record<string, LayerPresentation> = {};
        const rasterPresentations: Record<string, RasterPresentation> = {};
        for (const [resourceId, overlay] of Object.entries(restored)) {
          if (overlay.kind === "annotation") {
            annotationPresentations[resourceId] = defaultPresentation(
              overlay.manifest,
            );
          } else {
            rasterPresentations[resourceId] = DEFAULT_RASTER_PRESENTATION;
          }
        }
        setBootstrap(manifest);
        setLoadedOverlays(restored);
        setPrimaryOverlayIds(restoredIds);
        setSecondaryOverlayIds(restoredIds);
        setPrimaryPresentations(annotationPresentations);
        setSecondaryPresentations(annotationPresentations);
        setPrimaryRasterPresentations(rasterPresentations);
        setSecondaryRasterPresentations(rasterPresentations);
        if (manifest.session.slide) {
          setSlide(manifest.session.slide);
          setSlideId(manifest.session.slide.id);
        } else {
          setSlideId(manifest.defaultSlideId ?? manifest.slides[0]?.id ?? "");
        }
      })
      .catch((reason: unknown) => {
        if (!isAbortError(reason)) setError(errorMessage(reason));
      })
      .finally(() => setLoading(false));
    return () => {
      api.dispose();
      linkController.dispose();
    };
  }, [api, linkController]);

  useEffect(() => {
    if (!bootstrap || !slideId || slide?.id === slideId) return;
    const generation = ++slideGenerationRef.current;
    setLoading(true);
    setError(null);
    clearSelectionState(api, selectionGenerationRef, {
      setSelection,
      setSelectionLoading,
      setSelectionError,
    });
    void api
      .selectSlide(slideId)
      .then((manifest) => {
        if (generation !== slideGenerationRef.current) return;
        setSlide(manifest);
        setLoadedOverlays({});
        setPrimaryOverlayIds([]);
        setSecondaryOverlayIds([]);
        setPrimaryPresentations({});
        setSecondaryPresentations({});
        setPrimaryRasterPresentations({});
        setSecondaryRasterPresentations({});
      })
      .catch((reason: unknown) => {
        if (!isAbortError(reason) && generation === slideGenerationRef.current) {
          setError(errorMessage(reason));
          if (slide?.id) setSlideId(slide.id);
        }
      })
      .finally(() => {
        if (generation === slideGenerationRef.current) setLoading(false);
      });
  }, [api, bootstrap, slide?.id, slideId]);

  useEffect(() => {
    linkController.setEnabled(linked);
  }, [linkController, linked]);

  useEffect(() => {
    if (loading) return;
    const pending = Object.entries(loadedOverlays).flatMap(
      ([resourceId, overlay]) =>
        overlay.kind === "annotation" &&
        (overlay.manifest.lodStatus === "building" ||
          overlay.manifest.lodStatus === "not-built")
          ? [{ resourceId, storeId: overlay.manifest.id }]
          : [],
    );
    if (pending.length === 0) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      const results = await Promise.allSettled(
        pending.map(async ({ resourceId, storeId }) => ({
          resourceId,
          storeId,
          manifest: await api.store(storeId),
        })),
      );
      if (cancelled) return;
      for (const result of results) {
        if (result.status !== "fulfilled") continue;
        const { resourceId, storeId, manifest } = result.value;
        const previous = loadedOverlays[resourceId];
        if (
          previous?.kind !== "annotation" ||
          previous.manifest.id !== storeId
        ) {
          continue;
        }
        refreshPresentationState(
          resourceId,
          previous.manifest,
          manifest,
          setPrimaryPresentations,
        );
        refreshPresentationState(
          resourceId,
          previous.manifest,
          manifest,
          setSecondaryPresentations,
        );
      }
      setLoadedOverlays((current) => {
        const next = { ...current };
        for (const result of results) {
          if (result.status !== "fulfilled") continue;
          const { resourceId, storeId, manifest } = result.value;
          const existing = current[resourceId];
          if (existing?.kind !== "annotation" || existing.manifest.id !== storeId) {
            continue;
          }
          next[resourceId] = { ...existing, manifest };
        }
        return next;
      });
      const failure = results.find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failure && !isAbortError(failure.reason)) {
        setError(`Unable to refresh annotation metadata: ${errorMessage(failure.reason)}`);
      }
      timer = window.setTimeout(() => void poll(), 1000);
    };
    timer = window.setTimeout(() => void poll(), 1000);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
      for (const { storeId } of pending) api.abort(`store:${storeId}`);
    };
  }, [api, loadedOverlays, loading]);

  useEffect(() => {
    if (viewCount === 1) setActiveLayerView("primary");
  }, [viewCount]);

  const primaryAnnotations = useMemo(
    () =>
      annotationViewLayers(
        primaryOverlayIds,
        loadedOverlays,
        primaryPresentations,
      ),
    [loadedOverlays, primaryOverlayIds, primaryPresentations],
  );
  const secondaryAnnotations = useMemo(
    () =>
      annotationViewLayers(
        secondaryOverlayIds,
        loadedOverlays,
        secondaryPresentations,
      ),
    [loadedOverlays, secondaryOverlayIds, secondaryPresentations],
  );
  const primaryRasters = useMemo(
    () =>
      rasterViewLayers(
        primaryOverlayIds,
        loadedOverlays,
        primaryRasterPresentations,
      ),
    [loadedOverlays, primaryOverlayIds, primaryRasterPresentations],
  );
  const secondaryRasters = useMemo(
    () =>
      rasterViewLayers(
        secondaryOverlayIds,
        loadedOverlays,
        secondaryRasterPresentations,
      ),
    [loadedOverlays, secondaryOverlayIds, secondaryRasterPresentations],
  );

  const activeOverlayIds =
    activeLayerView === "primary" ? primaryOverlayIds : secondaryOverlayIds;
  const activeAnnotations =
    activeLayerView === "primary" ? primaryAnnotations : secondaryAnnotations;
  const activeRasters =
    activeLayerView === "primary" ? primaryRasters : secondaryRasters;

  const overlayCatalog = useMemo(() => {
    if (!bootstrap) return [];
    const catalog = [...bootstrap.overlays];
    const knownIds = new Set(catalog.map((resource) => resource.id));
    for (const overlay of Object.values(loadedOverlays)) {
      if (knownIds.has(overlay.resourceId)) continue;
      catalog.push({
        id: overlay.resourceId,
        name: overlay.manifest.name,
        kind: overlay.kind === "annotation" ? "annotation" : "raster-overlay",
        revision: overlay.manifest.revision,
      });
    }
    const associatedIds = new Set(
      slide?.associatedOverlays.map((resource) => resource.id) ?? [],
    );
    return catalog.sort(
      (left, right) =>
        Number(associatedIds.has(right.id)) - Number(associatedIds.has(left.id)) ||
        left.name.localeCompare(right.name),
    );
  }, [bootstrap, loadedOverlays, slide?.associatedOverlays]);

  const updateAnnotationPresentation = useCallback(
    (resourceId: string, presentation: LayerPresentation) => {
      const setter =
        activeLayerView === "primary"
          ? setPrimaryPresentations
          : setSecondaryPresentations;
      setter((current) => ({ ...current, [resourceId]: presentation }));
    },
    [activeLayerView],
  );

  const updateRasterPresentation = useCallback(
    (resourceId: string, presentation: RasterPresentation) => {
      const setter =
        activeLayerView === "primary"
          ? setPrimaryRasterPresentations
          : setSecondaryRasterPresentations;
      setter((current) => ({ ...current, [resourceId]: presentation }));
    },
    [activeLayerView],
  );

  const pick = useCallback(
    async ({ storeId, fid }: FeaturePick) => {
      const store = storesRef.current[storeId];
      if (!store) return;
      const generation = ++selectionGenerationRef.current;
      setSelectionLoading(true);
      setSelectionError(null);
      try {
        const detail = await api.feature(store, fid);
        if (generation === selectionGenerationRef.current) setSelection(detail);
      } catch (reason) {
        if (!isAbortError(reason) && generation === selectionGenerationRef.current) {
          setSelectionError(errorMessage(reason));
        }
      } finally {
        if (generation === selectionGenerationRef.current) {
          setSelectionLoading(false);
        }
      }
    },
    [api],
  );

  const clearSelection = useCallback(() => {
    clearSelectionState(api, selectionGenerationRef, {
      setSelection,
      setSelectionLoading,
      setSelectionError,
    });
  }, [api]);

  const applyConfiguration = async (config: ViewerConfigV1) => {
    slideGenerationRef.current += 1;
    clearSelection();
    setPendingOverlayIds(config.overlayResourceIds);
    const nextSlide = await api.selectSlide(config.slideResourceId);
    const nextLoaded: Record<string, SessionOverlay> = {};
    const failures: string[] = [];
    for (const resourceId of config.overlayResourceIds) {
      try {
        const result = await api.addOverlay(resourceId);
        nextLoaded[resourceId] =
          result.kind === "annotation"
            ? {
                kind: "annotation",
                resourceId,
                layerId: result.store.id,
                manifest: result.store,
              }
            : {
                kind: "raster",
                resourceId,
                layerId: result.layer.id,
                manifest: result.layer,
              };
      } catch (reason) {
        if (isAbortError(reason)) throw reason;
        failures.push(`${resourceId}: ${errorMessage(reason)}`);
      }
    }

    const primary = importedViewState(config.views.primary, nextLoaded);
    const secondary = importedViewState(config.views.secondary, nextLoaded);
    setSlide(nextSlide);
    setSlideId(config.slideResourceId);
    setLoadedOverlays(nextLoaded);
    setPrimaryOverlayIds(primary.overlayResourceIds);
    setSecondaryOverlayIds(secondary.overlayResourceIds);
    setPrimaryPresentations(primary.annotationPresentations);
    setSecondaryPresentations(secondary.annotationPresentations);
    setPrimaryRasterPresentations(primary.rasterPresentations);
    setSecondaryRasterPresentations(secondary.rasterPresentations);
    setRenderer(config.renderer);
    setViewCount(config.compare.enabled ? 2 : 1);
    setLinked(config.compare.linked);
    setActiveLayerView("primary");
    if (failures.length > 0) {
      throw new Error(`Some overlays could not be loaded: ${failures.join("; ")}`);
    }
  };

  const exportConfiguration = () => {
    if (!bootstrap || !slideId) return;
    try {
      const stableIds = new Set(bootstrap.overlays.map((resource) => resource.id));
      const overlayResourceIds = uniqueIds([
        ...primaryOverlayIds,
        ...secondaryOverlayIds,
        ...Object.keys(loadedOverlays),
      ]).filter((id) => loadedOverlays[id]);
      if (overlayResourceIds.some((id) => !stableIds.has(id))) {
        throw new Error(
          "Configuration cannot include a session-only overlay that is absent from the resource catalog.",
        );
      }
      const config = createViewerConfig({
        slideResourceId: slideId,
        overlayResourceIds,
        renderer,
        compare: { enabled: viewCount === 2, linked },
        views: {
          primary: configViewLayers(
            primaryOverlayIds,
            loadedOverlays,
            primaryPresentations,
            primaryRasterPresentations,
          ),
          secondary: configViewLayers(
            secondaryOverlayIds,
            loadedOverlays,
            secondaryPresentations,
            secondaryRasterPresentations,
          ),
        },
      });
      downloadConfiguration(
        serialiseViewerConfig(config),
        configurationFilename(slide?.name ?? "viewer"),
      );
      setConfigNotice("Configuration exported.");
      setError(null);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const importConfiguration = async (file: File) => {
    if (!bootstrap) return;
    setLoading(true);
    setError(null);
    setConfigNotice(null);
    try {
      if (file.size > MAX_VIEWER_CONFIG_BYTES) {
        throw new TypeError("Viewer configuration exceeds the 1 MB limit.");
      }
      const config = parseViewerConfig(
        await file.text(),
        configCatalog(bootstrap),
      );
      await applyConfiguration(config);
      setConfigNotice("Configuration imported.");
    } catch (reason) {
      if (!isAbortError(reason)) setError(errorMessage(reason));
    } finally {
      setPendingOverlayIds([]);
      setLoading(false);
    }
  };

  const toggleAvailableOverlay = async (resourceId: string) => {
    const activeIds =
      activeLayerView === "primary" ? primaryOverlayIds : secondaryOverlayIds;
    const otherIds =
      viewCount === 2
        ? activeLayerView === "primary"
          ? secondaryOverlayIds
          : primaryOverlayIds
        : [];
    const setActiveIds =
      activeLayerView === "primary"
        ? setPrimaryOverlayIds
        : setSecondaryOverlayIds;

    if (activeIds.includes(resourceId)) {
      setActiveIds((current) => current.filter((id) => id !== resourceId));
      clearSelection();
      if (otherIds.includes(resourceId)) return;
      const overlay = loadedOverlays[resourceId];
      if (!overlay) return;
      setPendingOverlayIds((current) => [...current, resourceId]);
      try {
        await api.removeOverlay(overlay.layerId);
        setPrimaryOverlayIds((current) =>
          current.filter((id) => id !== resourceId),
        );
        setSecondaryOverlayIds((current) =>
          current.filter((id) => id !== resourceId),
        );
        setLoadedOverlays((current) => {
          const next = { ...current };
          delete next[resourceId];
          return next;
        });
        removePresentation(resourceId, overlay.kind, {
          setPrimaryPresentations,
          setSecondaryPresentations,
          setPrimaryRasterPresentations,
          setSecondaryRasterPresentations,
        });
      } catch (reason) {
        if (!isAbortError(reason)) setError(errorMessage(reason));
      } finally {
        setPendingOverlayIds((current) =>
          current.filter((id) => id !== resourceId),
        );
      }
      return;
    }

    const existing = loadedOverlays[resourceId];
    if (existing) {
      setActiveIds((current) => [...current, resourceId]);
      ensurePresentation(
        resourceId,
        existing,
        activeLayerView,
        {
          setPrimaryPresentations,
          setSecondaryPresentations,
          setPrimaryRasterPresentations,
          setSecondaryRasterPresentations,
        },
      );
      return;
    }

    setPendingOverlayIds((current) => [...current, resourceId]);
    setError(null);
    try {
      const result = await api.addOverlay(resourceId);
      const overlay: SessionOverlay =
        result.kind === "annotation"
          ? {
              kind: "annotation",
              resourceId,
              layerId: result.store.id,
              manifest: result.store,
            }
          : {
              kind: "raster",
              resourceId,
              layerId: result.layer.id,
              manifest: result.layer,
            };
      setLoadedOverlays((current) => ({ ...current, [resourceId]: overlay }));
      setActiveIds((current) =>
        current.includes(resourceId) ? current : [...current, resourceId],
      );
      ensurePresentation(
        resourceId,
        overlay,
        activeLayerView,
        {
          setPrimaryPresentations,
          setSecondaryPresentations,
          setPrimaryRasterPresentations,
          setSecondaryRasterPresentations,
        },
      );
    } catch (reason) {
      if (!isAbortError(reason)) setError(errorMessage(reason));
    } finally {
      setPendingOverlayIds((current) =>
        current.filter((id) => id !== resourceId),
      );
    }
  };

  if (loading && !bootstrap) return <div className="splash">Loading viewer…</div>;
  if (error && !bootstrap) {
    return (
      <div className="splash splash--error">
        <h1>Unable to start the viewer</h1>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="viewer-app">
      <header className="app-header">
        <div>
          <span className="eyebrow">TIAToolbox</span>
          <h1>{bootstrap?.title ?? "Viewer"}</h1>
        </div>
        <div className="toolbar">
          <label>
            Renderer
            <select
              value={renderer}
              onChange={(event) =>
                setRenderer(event.target.value as RendererPreference)
              }
            >
              <option value="auto">Auto</option>
              <option value="canvas">Canvas</option>
              <option value="webgl">WebGL</option>
            </select>
          </label>
          <button
            type="button"
            className={viewCount === 2 ? "active" : ""}
            onClick={() => setViewCount((current) => (current === 1 ? 2 : 1))}
          >
            {viewCount === 1 ? "Compare" : "Single view"}
          </button>
          {viewCount === 2 && (
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={linked}
                onChange={(event) => setLinked(event.target.checked)}
              />
              Link views
            </label>
          )}
        </div>
      </header>

      <aside className="sidebar">
        <section>
          <h2>Slide</h2>
          <select
            value={slideId}
            disabled={loading}
            onChange={(event) => setSlideId(event.target.value)}
          >
            {bootstrap?.slides.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
          {slide && (
            <p className="resource-meta">
              {slide.width.toLocaleString()} × {slide.height.toLocaleString()} px
              {slide.mpp ? ` · ${slide.mpp[0].toFixed(3)} µm/px` : ""}
            </p>
          )}
        </section>

        <section>
          <h2>Available overlays</h2>
          {viewCount === 2 && (
            <div className="view-tabs" role="group" aria-label="Layer controls for view">
              <button
                type="button"
                className={activeLayerView === "primary" ? "active" : ""}
                onClick={() => setActiveLayerView("primary")}
              >
                View 1
              </button>
              <button
                type="button"
                className={activeLayerView === "secondary" ? "active" : ""}
                onClick={() => setActiveLayerView("secondary")}
              >
                View 2
              </button>
            </div>
          )}
          <div className="resource-list">
            {overlayCatalog.map((entry) => (
              <label key={entry.id} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={activeOverlayIds.includes(entry.id)}
                  disabled={loading || !slide || pendingOverlayIds.includes(entry.id)}
                  onChange={() => void toggleAvailableOverlay(entry.id)}
                />
                <span>{entry.name}</span>
                <small>
                  {entry.kind === "raster-overlay" ? "raster" : "vector"}
                </small>
              </label>
            ))}
            {overlayCatalog.length === 0 && <p>No overlays available.</p>}
          </div>
        </section>

        <section>
          <h2>Layers · {activeLayerView === "primary" ? "View 1" : "View 2"}</h2>
          {activeRasters.map(({ resourceId, loaded }) => (
            <RasterLayerPanel
              key={resourceId}
              layer={loaded.manifest}
              presentation={loaded.presentation}
              onChange={(next) => updateRasterPresentation(resourceId, next)}
              onRemove={() => void toggleAvailableOverlay(resourceId)}
            />
          ))}
          {activeAnnotations.map(({ resourceId, loaded }) => (
            <LayerPanel
              key={resourceId}
              store={loaded.manifest}
              presentation={loaded.presentation}
              onChange={(next) => updateAnnotationPresentation(resourceId, next)}
              onRemove={() => void toggleAvailableOverlay(resourceId)}
            />
          ))}
          {activeOverlayIds.some((id) => pendingOverlayIds.includes(id)) && (
            <p className="resource-meta">Loading overlay metadata…</p>
          )}
        </section>
        <section>
          <h2>Configuration</h2>
          <div className="config-actions">
            <button
              type="button"
              disabled={loading || !slide}
              onClick={exportConfiguration}
            >
              Export JSON
            </button>
            <button
              type="button"
              disabled={loading}
              onClick={() => configInputRef.current?.click()}
            >
              Import JSON
            </button>
            <input
              ref={configInputRef}
              hidden
              type="file"
              accept=".json,application/json"
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (file) void importConfiguration(file);
              }}
            />
          </div>
          <p className="resource-meta">
            Version 1 saves resource IDs, layer order, renderer, comparison, and
            per-view styles. Camera position is not included.
          </p>
          {configNotice && (
            <p className="config-notice" role="status">
              {configNotice}
            </p>
          )}
        </section>
        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button type="button" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}
      </aside>

      <main className={`map-grid map-grid--${viewCount}`}>
        {slide && (
          <SlideMap
            id="primary"
            slide={slide}
            stores={primaryAnnotations.map((entry) => entry.loaded)}
            rasters={primaryRasters.map((entry) => entry.loaded)}
            rendererPreference={renderer}
            selection={selection}
            linkController={linkController}
            onPick={pick}
            onClearSelection={clearSelection}
          />
        )}
        {slide && viewCount === 2 && (
          <SlideMap
            id="secondary"
            slide={slide}
            stores={secondaryAnnotations.map((entry) => entry.loaded)}
            rasters={secondaryRasters.map((entry) => entry.loaded)}
            rendererPreference={renderer}
            selection={selection}
            linkController={linkController}
            onPick={pick}
            onClearSelection={clearSelection}
          />
        )}
        {!slide && <div className="map-placeholder">Select a slide to begin.</div>}
      </main>

      <Inspector
        detail={selection}
        loading={selectionLoading}
        error={selectionError}
        onClose={clearSelection}
      />
    </div>
  );
}

function configViewLayers(
  resourceIds: string[],
  overlays: Record<string, SessionOverlay>,
  annotationPresentations: Record<string, LayerPresentation>,
  rasterPresentations: Record<string, RasterPresentation>,
): ViewerConfigLayer[] {
  const output: ViewerConfigLayer[] = [];
  for (const resourceId of resourceIds) {
    const overlay = overlays[resourceId];
    if (!overlay) continue;
    output.push(
      overlay.kind === "annotation"
        ? {
            kind: "annotation",
            resourceId,
            presentation:
              annotationPresentations[resourceId] ??
              defaultPresentation(overlay.manifest),
          }
        : {
            kind: "raster",
            resourceId,
            presentation:
              rasterPresentations[resourceId] ?? DEFAULT_RASTER_PRESENTATION,
          },
    );
  }
  return output;
}

interface ImportedViewState {
  overlayResourceIds: string[];
  annotationPresentations: Record<string, LayerPresentation>;
  rasterPresentations: Record<string, RasterPresentation>;
}

function importedViewState(
  layers: ViewerConfigLayer[],
  overlays: Record<string, SessionOverlay>,
): ImportedViewState {
  const output: ImportedViewState = {
    overlayResourceIds: [],
    annotationPresentations: {},
    rasterPresentations: {},
  };
  for (const layer of layers) {
    const overlay = overlays[layer.resourceId];
    if (!overlay || overlay.kind !== layer.kind) continue;
    output.overlayResourceIds.push(layer.resourceId);
    if (layer.kind === "annotation") {
      if (overlay.kind !== "annotation") continue;
      output.annotationPresentations[layer.resourceId] = presentationForStore(
        overlay.manifest,
        layer.presentation,
      );
    } else {
      output.rasterPresentations[layer.resourceId] = layer.presentation;
    }
  }
  return output;
}

function configCatalog(bootstrap: BootstrapManifest): ViewerConfigCatalog {
  const overlayKinds = Object.create(null) as Record<
    string,
    "annotation" | "raster"
  >;
  for (const resource of bootstrap.overlays) {
    if (resource.kind === "annotation") overlayKinds[resource.id] = "annotation";
    if (resource.kind === "raster-overlay") overlayKinds[resource.id] = "raster";
  }
  return {
    slideResourceIds: bootstrap.slides.map((resource) => resource.id),
    overlayKinds,
  };
}

function configurationFilename(slideName: string): string {
  const stem = slideName.replace(/\.[^.]+$/, "");
  const safe = stem
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${safe || "tiatoolbox-viewer"}-viewer.json`;
}

function downloadConfiguration(document: string, filename: string): void {
  const url = URL.createObjectURL(
    new Blob([document], { type: "application/json;charset=utf-8" }),
  );
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  window.document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function uniqueIds(values: string[]): string[] {
  return [...new Set(values)];
}

function restoreSessionOverlays(
  bootstrap: BootstrapManifest,
): Record<string, SessionOverlay> {
  const restored: Record<string, SessionOverlay> = {};
  const claimedResourceIds = new Set<string>();
  for (const store of bootstrap.session.annotationStores) {
    const resourceId = matchCatalogResource(
      bootstrap.overlays,
      store.name,
      store.revision,
      "annotation",
      claimedResourceIds,
    ) ?? `loaded:annotation:${store.id}`;
    claimedResourceIds.add(resourceId);
    restored[resourceId] = {
      kind: "annotation",
      resourceId,
      layerId: store.id,
      manifest: store,
    };
  }
  for (const raster of bootstrap.session.rasterLayers) {
    const resourceId =
      bootstrap.overlays.find((resource) => resource.id === raster.id)?.id ??
      matchCatalogResource(
        bootstrap.overlays,
        raster.name,
        raster.revision,
        "raster-overlay",
        claimedResourceIds,
      ) ??
      `loaded:raster:${raster.id}`;
    claimedResourceIds.add(resourceId);
    restored[resourceId] = {
      kind: "raster",
      resourceId,
      layerId: raster.id,
      manifest: raster,
    };
  }
  return restored;
}

function matchCatalogResource(
  resources: ResourceSummary[],
  name: string,
  revision: string | undefined,
  kind: "annotation" | "raster-overlay",
  claimed: Set<string>,
): string | undefined {
  const candidates = resources.filter(
    (resource) =>
      !claimed.has(resource.id) &&
      resource.kind !== (kind === "annotation" ? "raster-overlay" : "annotation") &&
      resource.name === name,
  );
  return (
    candidates.find(
      (resource) => !revision || !resource.revision || resource.revision === revision,
    ) ?? candidates[0]
  )?.id;
}

function annotationViewLayers(
  resourceIds: string[],
  overlays: Record<string, SessionOverlay>,
  presentations: Record<string, LayerPresentation>,
): AnnotationViewLayer[] {
  return resourceIds.flatMap((resourceId) => {
    const overlay = overlays[resourceId];
    const presentation = presentations[resourceId];
    return overlay?.kind === "annotation" && presentation
      ? [{ resourceId, loaded: { manifest: overlay.manifest, presentation } }]
      : [];
  });
}

function rasterViewLayers(
  resourceIds: string[],
  overlays: Record<string, SessionOverlay>,
  presentations: Record<string, RasterPresentation>,
): RasterViewLayer[] {
  return resourceIds.flatMap((resourceId) => {
    const overlay = overlays[resourceId];
    const presentation = presentations[resourceId];
    return overlay?.kind === "raster" && presentation
      ? [{ resourceId, loaded: { manifest: overlay.manifest, presentation } }]
      : [];
  });
}

interface PresentationSetters {
  setPrimaryPresentations: React.Dispatch<
    React.SetStateAction<Record<string, LayerPresentation>>
  >;
  setSecondaryPresentations: React.Dispatch<
    React.SetStateAction<Record<string, LayerPresentation>>
  >;
  setPrimaryRasterPresentations: React.Dispatch<
    React.SetStateAction<Record<string, RasterPresentation>>
  >;
  setSecondaryRasterPresentations: React.Dispatch<
    React.SetStateAction<Record<string, RasterPresentation>>
  >;
}

function ensurePresentation(
  resourceId: string,
  overlay: SessionOverlay,
  view: LayerView,
  setters: PresentationSetters,
): void {
  if (overlay.kind === "annotation") {
    const setter =
      view === "primary"
        ? setters.setPrimaryPresentations
        : setters.setSecondaryPresentations;
    setter((current) => ({
      ...current,
      [resourceId]: current[resourceId] ?? defaultPresentation(overlay.manifest),
    }));
    return;
  }
  const setter =
    view === "primary"
      ? setters.setPrimaryRasterPresentations
      : setters.setSecondaryRasterPresentations;
  setter((current) => ({
    ...current,
    [resourceId]: current[resourceId] ?? DEFAULT_RASTER_PRESENTATION,
  }));
}

function removePresentation(
  resourceId: string,
  kind: SessionOverlay["kind"],
  setters: PresentationSetters,
): void {
  const remove = <T,>(current: Record<string, T>): Record<string, T> => {
    const next = { ...current };
    delete next[resourceId];
    return next;
  };
  if (kind === "annotation") {
    setters.setPrimaryPresentations(remove);
    setters.setSecondaryPresentations(remove);
  } else {
    setters.setPrimaryRasterPresentations(remove);
    setters.setSecondaryRasterPresentations(remove);
  }
}

function refreshPresentationState(
  resourceId: string,
  previousStore: StoreManifest,
  nextStore: StoreManifest,
  setter: React.Dispatch<
    React.SetStateAction<Record<string, LayerPresentation>>
  >,
): void {
  setter((current) => {
    const presentation = current[resourceId];
    if (!presentation) return current;
    const refreshed = refreshProvisionalPresentation(
      presentation,
      previousStore,
      nextStore,
    );
    return refreshed === presentation
      ? current
      : { ...current, [resourceId]: refreshed };
  });
}

interface SelectionSetters {
  setSelection: React.Dispatch<React.SetStateAction<FeatureDetail | null>>;
  setSelectionLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setSelectionError: React.Dispatch<React.SetStateAction<string | null>>;
}

function clearSelectionState(
  api: ApiClient,
  generation: React.MutableRefObject<number>,
  setters: SelectionSetters,
): void {
  generation.current += 1;
  api.abort("feature-detail");
  setters.setSelection(null);
  setters.setSelectionLoading(false);
  setters.setSelectionError(null);
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}
