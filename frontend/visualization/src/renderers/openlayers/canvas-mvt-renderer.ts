import type { FeatureLike } from "ol/Feature.js";
import VectorTileLayer from "ol/layer/VectorTile.js";
import type Map from "ol/Map.js";
import type { Pixel } from "ol/pixel.js";

import {
  annotationLayerMinZoom,
  type LayerPresentation,
} from "../../domain/style-spec";
import type { StoreManifest } from "../../api/types";
import type {
  AnnotationLayerHandle,
  AnnotationRendererContext,
  PickResult,
} from "../annotation-renderer";
import { createCanvasStyleFunction } from "./style-compiler";
import { createManagedVectorTileSource } from "./vector-tile-source";
import { pickFeatureOnServer } from "./server-pick";

export class CanvasMvtRenderer implements AnnotationLayerHandle {
  readonly capabilities = {
    kind: "canvas" as const,
    localPick: true,
    dynamicStyle: true,
    gpu: false,
  };

  private readonly layer: VectorTileLayer;
  private readonly requests;
  private readonly store: StoreManifest;
  private map: Map | null = null;

  constructor(context: AnnotationRendererContext) {
    const managed = createManagedVectorTileSource(context);
    this.requests = managed.requests;
    this.store = context.store;
    this.layer = new VectorTileLayer({
      source: managed.source,
      renderMode: "hybrid",
      renderBuffer: 4,
      preload: 0,
      updateWhileAnimating: false,
      updateWhileInteracting: false,
      style: createCanvasStyleFunction(context.presentation),
      visible: context.presentation.visible,
      opacity: context.presentation.opacity,
      minZoom: annotationLayerMinZoom(context.store, context.presentation),
      properties: { storeId: context.store.id, renderer: "canvas" },
    });
  }

  attach(map: Map): void {
    this.map = map;
    map.addLayer(this.layer);
  }

  detach(map: Map): void {
    map.removeLayer(this.layer);
    if (this.map === map) this.map = null;
  }

  setPresentation(presentation: LayerPresentation): void {
    this.layer.setVisible(presentation.visible);
    this.layer.setOpacity(presentation.opacity);
    this.layer.setMinZoom(annotationLayerMinZoom(this.store, presentation));
    this.layer.setStyle(createCanvasStyleFunction(presentation));
  }

  async pick(
    pixel: Pixel,
    slideCoordinate?: readonly [number, number],
    resolution?: number,
  ): Promise<PickResult | null> {
    const features = (await this.layer.getFeatures(pixel)) as FeatureLike[];
    const feature =
      features[0] ??
      this.map?.forEachFeatureAtPixel(
        pixel,
        (candidate, layer) => (layer === this.layer ? candidate : undefined),
        {
          hitTolerance: 4,
          layerFilter: (layer) => layer === this.layer,
        },
      );
    if (!feature) {
      return pickFeatureOnServer(
        this.store,
        slideCoordinate,
        resolution,
        this.requests,
      );
    }
    const properties = feature.getProperties();
    // Overview aggregate cells intentionally have no MVT feature ID and are not
    // authoritative annotations, so never treat one of their properties as an ID.
    const id = feature.getId();
    if (typeof id !== "number" && typeof id !== "string") {
      return pickFeatureOnServer(
        this.store,
        slideCoordinate,
        resolution,
        this.requests,
      );
    }
    return { fid: id, properties };
  }

  dispose(): void {
    this.requests.abortAll();
    this.layer.getSource()?.clear();
    this.layer.dispose();
  }
}
