import type { EventsKey } from "ol/events.js";
import WebGLVectorTileLayer from "ol/layer/WebGLVectorTile.js";
import type Map from "ol/Map.js";
import { unByKey } from "ol/Observable.js";
import type { Pixel } from "ol/pixel.js";
import type RenderFeature from "ol/render/Feature.js";
import type VectorTileSource from "ol/source/VectorTile.js";

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
import { compileWebGlStyle } from "./style-compiler";
import {
  createManagedVectorTileSource,
  type ManagedVectorTileSource,
} from "./vector-tile-source";
import { pickFeatureOnServer } from "./server-pick";

export class WebGlMvtRenderer implements AnnotationLayerHandle {
  readonly capabilities = {
    kind: "webgl" as const,
    // OpenLayers' experimental WebGLVectorTile hit buffer is not reliable in
    // all embedded Chromium/WebView builds. Close-detail picks use the bounded
    // authoritative spatial endpoint instead.
    localPick: false,
    dynamicStyle: true,
    gpu: true,
  };

  private readonly layer: WebGLVectorTileLayer<
    VectorTileSource<RenderFeature>,
    RenderFeature
  >;
  private readonly managed: ManagedVectorTileSource;
  private readonly requests;
  private readonly store: StoreManifest;
  private structureKey: string;
  private resolutionKey: EventsKey | undefined;

  constructor(context: AnnotationRendererContext) {
    const managed = createManagedVectorTileSource(context);
    this.managed = managed;
    this.requests = managed.requests;
    this.store = context.store;
    const compiled = compileWebGlStyle(context.presentation, context.store);
    this.structureKey = compiled.structureKey;
    this.layer = new WebGLVectorTileLayer<
      VectorTileSource<RenderFeature>,
      RenderFeature
    >({
      source: managed.source,
      style: compiled.rules,
      variables: compiled.variables,
      visible: context.presentation.visible,
      opacity: context.presentation.opacity,
      minZoom: annotationLayerMinZoom(context.store, context.presentation),
      // Close-detail selection uses the authoritative server endpoint. Avoid
      // per-feature hit attributes and buffers for full-slide segmentations.
      disableHitDetection: true,
      properties: { storeId: context.store.id, renderer: "webgl" },
    });
  }

  attach(map: Map): void {
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
  }

  setPresentation(presentation: LayerPresentation): void {
    this.layer.setVisible(presentation.visible);
    this.layer.setOpacity(presentation.opacity);
    this.layer.setMinZoom(annotationLayerMinZoom(this.store, presentation));
    const compiled = compileWebGlStyle(presentation, this.store);
    this.layer.updateStyleVariables(compiled.variables);
    if (compiled.structureKey !== this.structureKey) {
      this.layer.setStyle(compiled.rules);
      this.structureKey = compiled.structureKey;
    }
  }

  async pick(
    _pixel: Pixel,
    slideCoordinate?: readonly [number, number],
    resolution?: number,
  ): Promise<PickResult | null> {
    return pickFeatureOnServer(
      this.store,
      slideCoordinate,
      resolution,
      this.requests,
    );
  }

  dispose(): void {
    if (this.resolutionKey) unByKey(this.resolutionKey);
    this.resolutionKey = undefined;
    this.requests.abortAll();
    // WebGLVectorTileLayer must be disposed explicitly or its context survives.
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
