import "ol/ol.css";

import { useEffect, useMemo, useRef, useState } from "react";

import FullScreen from "ol/control/FullScreen.js";
import OverviewMap from "ol/control/OverviewMap.js";
import ScaleLine from "ol/control/ScaleLine.js";
import { defaults as defaultControls } from "ol/control/defaults.js";
import TileLayer from "ol/layer/Tile.js";
import OlMap from "ol/Map.js";
import { unByKey } from "ol/Observable.js";
import type Projection from "ol/proj/Projection.js";
import ImageTile from "ol/source/ImageTile.js";
import Zoomify from "ol/source/Zoomify.js";
import type TileGrid from "ol/tilegrid/TileGrid.js";
import View from "ol/View.js";

import type {
  FeatureDetail,
  FeatureId,
  LoadedRaster,
  LoadedStore,
  RasterLayerManifest,
  SlideManifest,
} from "../api/types";
import { isAbortError } from "../api/client";
import { requiredTileProperties } from "../domain/style-spec";
import { createSelectionLayer, updateSelectionLayer } from "../layers/selection-layer";
import type {
  AnnotationLayerHandle,
  RendererPreference,
} from "../renderers/annotation-renderer";
import { createRenderer } from "../renderers/openlayers/create-renderer";
import {
  type ManagedLayerEntry,
  reconcileKeyedLayers,
} from "./keyed-layer-reconciler";
import { renderMapToPng } from "./map-export";
import { createSlideProjection, createSlideTileGrid, mapToSlide, slideExtent } from "./projection";
import type { ViewLinkController } from "./view-link-controller";
import { slideViewTransform } from "./view-link-controller";

export interface FeaturePick {
  storeId: string;
  fid: FeatureId;
}

const RASTER_LAYER_ORDER_BASE = 100;
const ANNOTATION_LAYER_ORDER_BASE = 1_000;

interface SlideMapProps {
  id: string;
  slide: SlideManifest;
  stores: LoadedStore[];
  rasters: LoadedRaster[];
  rendererPreference: RendererPreference;
  selection: FeatureDetail | null;
  linkController: ViewLinkController;
  onPick(pick: FeaturePick): void;
  onClearSelection(): void;
}

export function SlideMap({
  id,
  slide,
  stores,
  rasters,
  rendererPreference,
  selection,
  linkController,
  onPick,
  onClearSelection,
}: SlideMapProps) {
  const targetRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<OlMap | null>(null);
  const slideIdRef = useRef<string | null>(null);
  const pickEpochRef = useRef(0);
  const handlesRef = useRef(
    new Map<string, ManagedLayerEntry<AnnotationLayerHandle>>(),
  );
  const rasterLayersRef = useRef(
    new Map<string, ManagedLayerEntry<TileLayer<Zoomify | ImageTile>>>(),
  );
  const selectionLayerRef = useRef<ReturnType<typeof createSelectionLayer> | null>(
    null,
  );
  if (!selectionLayerRef.current) selectionLayerRef.current = createSelectionLayer();
  const selectionLayer = selectionLayerRef.current;
  const [mapInstance, setMapInstance] = useState<OlMap | null>(null);
  const [coordinate, setCoordinate] = useState<[number, number] | null>(null);
  const [rendererSummary, setRendererSummary] = useState("none");
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const projection = useMemo(() => createSlideProjection(slide), [slide]);
  const tileGrid = useMemo(() => createSlideTileGrid(slide), [slide]);
  useEffect(() => {
    const target = targetRef.current;
    if (!target) return;
    const slideChanged = slideIdRef.current !== slide.id;
    slideIdRef.current = slide.id;
    const rasterSource = createRasterSource(slide, projection, tileGrid);
    const view = new View({
      projection,
      resolutions: tileGrid.getResolutions(),
      extent: [...slideExtent(slide)],
      constrainOnlyCenter: true,
      smoothResolutionConstraint: true,
      showFullExtent: true,
    });
    const map = new OlMap({
      target,
      maxTilesLoading: 16,
      layers: [new TileLayer({ source: rasterSource, properties: { role: "slide" } })],
      view,
      controls: defaultControls({ attribution: false, rotate: true }).extend([
        new FullScreen(),
        new ScaleLine({ units: "metric", bar: true, minWidth: 120 }),
        new OverviewMap({
          className: "ol-overviewmap tiatoolbox-overview",
          collapsed: true,
          layers: [
            new TileLayer({
              source: rasterSource,
              properties: { role: "overview-slide" },
            }),
          ],
          rotateWithView: true,
          tipLabel: "Toggle overview map",
          view: new View({
            projection,
            resolutions: tileGrid.getResolutions(),
            extent: [...slideExtent(slide)],
            constrainOnlyCenter: true,
            showFullExtent: true,
          }),
        }),
      ]),
    });
    map.addLayer(selectionLayer);
    mapRef.current = map;
    setMapInstance(map);
    const unregister = linkController.register(id, view, slideViewTransform, {
      restoreLastState: !slideChanged,
    });
    const needsInitialFit =
      slideChanged ||
      view.getCenter() === undefined ||
      view.getResolution() === undefined;
    const pointerKey = map.on("pointermove", (event) => {
      if (event.dragging) return;
      const slideCoordinate = mapToSlide(event.coordinate as [number, number]);
      setCoordinate(slideCoordinate);
    });
    const clickKey = map.on("singleclick", async (event) => {
      const epoch = ++pickEpochRef.current;
      const entries = [...handlesRef.current.entries()].reverse();
      for (const [storeId, entry] of entries) {
        try {
          const result = await entry.value.pick(
            event.pixel,
            mapToSlide(event.coordinate as [number, number]),
            view.getResolution(),
          );
          if (epoch !== pickEpochRef.current) return;
          if (result) {
            onPick({ storeId, fid: result.fid });
            return;
          }
        } catch (error) {
          if (epoch !== pickEpochRef.current || isAbortError(error)) return;
          console.warn("Annotation hit detection failed.", error);
        }
      }
      if (epoch === pickEpochRef.current) onClearSelection();
    });
    requestAnimationFrame(() => {
      map.updateSize();
      if (needsInitialFit) {
        view.fit([...slideExtent(slide)], {
          size: map.getSize(),
          padding: [24, 24, 24, 24],
        });
      }
    });
    return () => {
      ++pickEpochRef.current;
      unregister();
      unByKey([pointerKey, clickKey]);
      for (const { value: handle } of handlesRef.current.values()) {
        handle.detach(map);
        handle.dispose();
      }
      handlesRef.current.clear();
      for (const { value: layer } of rasterLayersRef.current.values()) {
        map.removeLayer(layer);
        layer.getSource()?.clear();
        layer.dispose();
      }
      rasterLayersRef.current.clear();
      map.removeLayer(selectionLayer);
      map.setTarget(undefined);
      map.dispose();
      rasterSource.clear();
      rasterSource.dispose();
      mapRef.current = null;
      setMapInstance(null);
    };
  }, [
    id,
    linkController,
    onClearSelection,
    onPick,
    projection,
    selectionLayer,
    slide,
    tileGrid,
  ]);

  useEffect(() => {
    if (!mapInstance || mapInstance !== mapRef.current) return;
    const layers = rasterLayersRef.current;
    reconcileKeyedLayers(layers, rasters, {
      key: ({ manifest }) => manifest.id,
      identity: ({ manifest }) => rasterSourceIdentity(manifest),
      create: (raster) => {
        const layer = new TileLayer({
          source: createOverlayRasterSource(
            raster.manifest,
            slide,
            projection,
            tileGrid,
          ),
          properties: { role: "raster-overlay", layerId: raster.manifest.id },
        });
        mapInstance.addLayer(layer);
        return layer;
      },
      update: (layer, raster, order) => {
        layer.setVisible(raster.presentation.visible);
        layer.setOpacity(raster.presentation.opacity);
        layer.setZIndex(RASTER_LAYER_ORDER_BASE + order);
      },
      dispose: (layer) => {
        mapInstance.removeLayer(layer);
        layer.getSource()?.clear();
        layer.dispose();
      },
    });
  }, [mapInstance, projection, rasters, slide, tileGrid]);

  useEffect(() => {
    if (!mapInstance || mapInstance !== mapRef.current) return;
    ++pickEpochRef.current;
    const handles = handlesRef.current;
    reconcileKeyedLayers(handles, stores, {
      key: ({ manifest }) => manifest.id,
      identity: (store) => annotationSourceIdentity(store, rendererPreference),
      create: (store) => {
        const handle = createRenderer(rendererPreference, {
          slide,
          store: store.manifest,
          projection,
          tileGrid,
          presentation: store.presentation,
        });
        handle.attach(mapInstance);
        return handle;
      },
      update: (handle, store, order) => {
        handle.setOrder(ANNOTATION_LAYER_ORDER_BASE + order);
        handle.setPresentation(store.presentation);
      },
      dispose: (handle) => {
        handle.detach(mapInstance);
        handle.dispose();
      },
    });
    const rendererKinds = new Set(
      [...handles.values()].map(({ value }) => value.capabilities.kind),
    );
    setRendererSummary([...rendererKinds].join(" + ") || "none");
  }, [mapInstance, projection, rendererPreference, slide, stores, tileGrid]);

  useEffect(() => {
    updateSelectionLayer(selectionLayer, selection);
  }, [selection, selectionLayer]);

  const reset = () => {
    const map = mapRef.current;
    if (!map) return;
    map.getView().fit([...slideExtent(slide)], {
      size: map.getSize(),
      padding: [24, 24, 24, 24],
      duration: 180,
    });
  };

  const exportPng = async () => {
    const map = mapRef.current;
    if (!map || exporting) return;
    setExporting(true);
    setExportError(null);
    try {
      const blob = await renderMapToPng(map);
      downloadBlob(blob, screenshotName(slide.name, id));
    } catch (error) {
      setExportError(exportErrorMessage(error));
    } finally {
      setExporting(false);
    }
  };

  return (
    <section className="slide-map" aria-label={`${slide.name} view`}>
      <div ref={targetRef} className="slide-map__target" />
      <div className="slide-map__status">
        <span>{rendererSummary}</span>
        <span>
          {coordinate
            ? `${Math.round(coordinate[0])}, ${Math.round(coordinate[1])}`
            : "—, —"}
        </span>
      </div>
      <div className="slide-map__actions">
        <button type="button" onClick={reset}>
          Fit slide
        </button>
        <button type="button" disabled={exporting} onClick={() => void exportPng()}>
          {exporting ? "Preparing PNG…" : "Save view as PNG"}
        </button>
      </div>
      {exportError && (
        <div className="slide-map__export-error" role="alert">
          {exportError}
        </div>
      )}
    </section>
  );
}

function screenshotName(slideName: string, viewId: string): string {
  const slideStem = slideName.replace(/\.[^.]+$/, "");
  const safe = `${slideStem}-${viewId}`
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${safe || "tiatoolbox-view"}.png`;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportErrorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "SecurityError") {
    return "PNG export was blocked by a cross-origin tile. Serve image tiles with CORS enabled.";
  }
  return error instanceof Error ? error.message : "Unable to export this view.";
}

function annotationSourceIdentity(
  store: LoadedStore,
  rendererPreference: RendererPreference,
): string {
  const autoRepresentation = store.manifest.representations.find(
    (representation) => representation.kind === "auto",
  );
  return JSON.stringify({
    rendererPreference,
    id: store.manifest.id,
    revision: store.manifest.revision,
    lodStatus: store.manifest.lodStatus,
    tileUrl: store.manifest.tileUrlTemplates.auto ?? autoRepresentation?.urlTemplate,
    pickUrl: store.manifest.pickUrl,
    overlaps: store.manifest.overlaps,
    policy: autoRepresentation?.policy,
    fields: requiredTileProperties(store.presentation),
  });
}

function rasterSourceIdentity(raster: RasterLayerManifest): string {
  return JSON.stringify({
    id: raster.id,
    revision: raster.revision,
    layerName: raster.layerName,
    width: raster.width,
    height: raster.height,
    tileUrl: raster.rasterTileUrlTemplate,
  });
}

function createRasterSource(
  slide: SlideManifest,
  projection: Projection,
  tileGrid: TileGrid,
): Zoomify | ImageTile {
  if (
    slide.rasterTileUrlTemplate.includes("{TileGroup}") ||
    slide.rasterTileUrlTemplate.includes("{tileIndex}")
  ) {
    return new Zoomify({
      url: slide.rasterTileUrlTemplate,
      size: [slide.width, slide.height],
      tileSize: slide.tileSize,
      projection,
      extent: [...slideExtent(slide)],
      crossOrigin: "anonymous",
      zDirection: -1,
      transition: 0,
    });
  }
  return new ImageTile({
    url: slide.rasterTileUrlTemplate,
    tileGrid,
    tileSize: slide.tileSize,
    projection,
    crossOrigin: "anonymous",
    wrapX: false,
    zDirection: -1,
    transition: 0,
  });
}

function createOverlayRasterSource(
  raster: RasterLayerManifest,
  slide: SlideManifest,
  projection: Projection,
  tileGrid: TileGrid,
): Zoomify | ImageTile {
  if (
    raster.rasterTileUrlTemplate.includes("{TileGroup}") ||
    raster.rasterTileUrlTemplate.includes("{tileIndex}")
  ) {
    return new Zoomify({
      url: raster.rasterTileUrlTemplate,
      size: [raster.width, raster.height],
      tileSize: slide.tileSize,
      projection,
      extent: [...slideExtent(slide)],
      crossOrigin: "anonymous",
      zDirection: -1,
      transition: 0,
    });
  }
  return new ImageTile({
    url: raster.rasterTileUrlTemplate,
    tileGrid,
    tileSize: slide.tileSize,
    projection,
    crossOrigin: "anonymous",
    wrapX: false,
    zDirection: -1,
    transition: 0,
  });
}
