import type { FeatureLike } from "ol/Feature.js";
import type { EventsKey } from "ol/events.js";
import VectorTileLayer from "ol/layer/VectorTile.js";
import type Map from "ol/Map.js";
import { unByKey } from "ol/Observable.js";
import type { Pixel } from "ol/pixel.js";
import type TileGrid from "ol/tilegrid/TileGrid.js";

import {
  annotationLayerMinZoom,
  matchesPresentation,
  presentationAllowsPicking,
  type LayerPresentation,
} from "../../domain/style-spec";
import type { StoreManifest } from "../../api/types";
import type {
  AnnotationLayerHandle,
  AnnotationRendererContext,
  PickResult,
} from "../annotation-renderer";
import { createCanvasStyleFunction } from "./style-compiler";
import {
  createManagedVectorTileSource,
  type ManagedVectorTileSource,
} from "./vector-tile-source";
import { pickFeatureOnServer } from "./server-pick";

const AGGREGATE_PROPERTY = "tiatoolbox_aggregate";

/** Convert only a locally rendered, authoritative annotation into a pick. */
export function localFeaturePickResult(
  feature: FeatureLike,
  presentation: LayerPresentation,
): PickResult | null {
  const properties = feature.getProperties();
  if (!matchesPresentation(properties, presentation)) return null;
  if (
    presentation.overviewMode === "hidden" &&
    properties[AGGREGATE_PROPERTY] === true
  ) {
    return null;
  }
  // Overview aggregate cells intentionally have no MVT feature ID and are not
  // authoritative annotations, so never treat one of their properties as an ID.
  const id = feature.getId();
  if (typeof id !== "number" && typeof id !== "string") return null;
  return { fid: id, properties };
}

export class CanvasMvtRenderer implements AnnotationLayerHandle {
  readonly capabilities = {
    kind: "canvas" as const,
    localPick: true,
    dynamicStyle: true,
    gpu: false,
  };

  private readonly layer: VectorTileLayer;
  private readonly managed: ManagedVectorTileSource;
  private readonly requests;
  private readonly store: StoreManifest;
  private readonly tileGrid: TileGrid;
  private presentation: LayerPresentation;
  private map: Map | null = null;
  private resolutionKey: EventsKey | undefined;

  constructor(context: AnnotationRendererContext) {
    const managed = createManagedVectorTileSource(context);
    this.managed = managed;
    this.requests = managed.requests;
    this.store = context.store;
    this.tileGrid = context.tileGrid;
    this.presentation = context.presentation;
    this.layer = new VectorTileLayer({
      source: managed.source,
      renderMode: "hybrid",
      renderBuffer: 4,
      preload: 0,
      updateWhileAnimating: false,
      updateWhileInteracting: false,
      style: createCanvasStyleFunction(
        context.presentation,
        context.store,
        context.tileGrid,
      ),
      visible: context.presentation.visible,
      opacity: context.presentation.opacity,
      minZoom: annotationLayerMinZoom(context.store, context.presentation),
      properties: { storeId: context.store.id, renderer: "canvas" },
    });
  }

  attach(map: Map): void {
    this.map = map;
    const view = map.getView();
    this.synchronizeSource(view.getResolution());
    this.resolutionKey = view.on("change:resolution", () => {
      this.synchronizeSource(view.getResolution());
    });
    map.addLayer(this.layer);
  }

  detach(map: Map): void {
    if (this.resolutionKey) unByKey(this.resolutionKey);
    this.resolutionKey = undefined;
    map.removeLayer(this.layer);
    if (this.map === map) this.map = null;
  }

  setOrder(order: number): void {
    this.layer.setZIndex(order);
  }

  setPresentation(presentation: LayerPresentation): void {
    this.presentation = presentation;
    this.layer.setVisible(presentation.visible);
    this.layer.setOpacity(presentation.opacity);
    this.layer.setMinZoom(annotationLayerMinZoom(this.store, presentation));
    this.layer.setStyle(
      createCanvasStyleFunction(presentation, this.store, this.tileGrid),
    );
  }

  async pick(
    pixel: Pixel,
    slideCoordinate?: readonly [number, number],
    resolution?: number,
  ): Promise<PickResult | null> {
    if (!presentationAllowsPicking(this.presentation)) return null;
    const features = (await this.layer.getFeatures(pixel)) as FeatureLike[];
    const localResult =
      features
        .map((feature) => localFeaturePickResult(feature, this.presentation))
        .find((result): result is PickResult => result !== null) ??
      this.map?.forEachFeatureAtPixel(
        pixel,
        (candidate, layer) =>
          layer === this.layer
            ? localFeaturePickResult(candidate, this.presentation) ?? undefined
            : undefined,
        {
          hitTolerance: 4,
          layerFilter: (layer) => layer === this.layer,
        },
      );
    if (!localResult) {
      return pickFeatureOnServer(
        this.store,
        slideCoordinate,
        resolution,
        this.presentation,
        this.requests,
      );
    }
    return localResult;
  }

  dispose(): void {
    if (this.resolutionKey) unByKey(this.resolutionKey);
    this.resolutionKey = undefined;
    this.requests.abortAll();
    this.layer.dispose();
    this.managed.disposeSources();
  }

  private synchronizeSource(resolution: number | undefined): void {
    const source = this.managed.sourceForResolution(resolution);
    if (this.layer.getSource() === source) return;
    this.requests.abortAll();
    this.layer.setSource(source);
  }
}
