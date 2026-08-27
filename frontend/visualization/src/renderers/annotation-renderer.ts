import type Map from "ol/Map.js";
import type { Pixel } from "ol/pixel.js";
import type Projection from "ol/proj/Projection.js";
import type TileGrid from "ol/tilegrid/TileGrid.js";

import type { FeatureId, SlideManifest, StoreManifest } from "../api/types";
import type { LayerPresentation } from "../domain/style-spec";

export type RendererKind = "canvas" | "webgl";
export type RendererPreference = RendererKind | "auto";

export interface PickResult {
  fid: FeatureId;
  properties: Record<string, unknown>;
}

export interface RendererCapabilities {
  kind: RendererKind;
  localPick: boolean;
  dynamicStyle: boolean;
  gpu: boolean;
}

export interface AnnotationRendererContext {
  slide: SlideManifest;
  store: StoreManifest;
  projection: Projection;
  tileGrid: TileGrid;
  presentation: LayerPresentation;
}

export interface AnnotationLayerHandle {
  readonly capabilities: RendererCapabilities;
  attach(map: Map): void;
  detach(map: Map): void;
  setOrder(order: number): void;
  setPresentation(presentation: LayerPresentation): void;
  pick(
    pixel: Pixel,
    slideCoordinate?: readonly [number, number],
    resolution?: number,
  ): Promise<PickResult | null>;
  dispose(): void;
}

let cachedWebGlAvailability: boolean | undefined;

export function webGlAvailable(): boolean {
  if (cachedWebGlAvailability !== undefined) return cachedWebGlAvailability;
  if (typeof document === "undefined") return false;
  const canvas = document.createElement("canvas");
  const context =
    canvas.getContext("webgl2", { failIfMajorPerformanceCaveat: true }) ??
    canvas.getContext("webgl", { failIfMajorPerformanceCaveat: true });
  cachedWebGlAvailability = Boolean(context);
  context?.getExtension("WEBGL_lose_context")?.loseContext();
  return cachedWebGlAvailability;
}

export function resolvedRendererKind(
  preference: RendererPreference,
): RendererKind {
  if (preference === "canvas") return "canvas";
  if (preference === "webgl") return webGlAvailable() ? "webgl" : "canvas";
  return webGlAvailable() ? "webgl" : "canvas";
}
