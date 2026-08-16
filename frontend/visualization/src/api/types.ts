export type ResourceId = string;
export type Revision = string;
export type FeatureId = number | string;
export type ResourceKind = "slide" | "annotation" | "raster-overlay";

export interface ResourceSummary {
  id: ResourceId;
  name: string;
  kind?: ResourceKind;
  revision?: Revision;
}

export interface BootstrapManifest {
  apiVersion: string;
  title: string;
  slides: ResourceSummary[];
  overlays: ResourceSummary[];
  session: SessionManifest;
  capabilities: Record<string, unknown>;
  /** Compatibility alias for the pre-session prototype contract. */
  stores: ResourceSummary[];
  defaultSlideId?: ResourceId;
  defaultStoreIds: ResourceId[];
}

export interface SlideManifest {
  id: ResourceId;
  name: string;
  revision?: Revision;
  width: number;
  height: number;
  mpp: readonly [number, number] | null;
  objectivePower?: number;
  tileSize: number;
  maxZoom: number;
  resolutions: number[];
  mapExtent: readonly [number, number, number, number];
  rasterTileUrlTemplate: string;
  associatedOverlays: ResourceSummary[];
}

export interface RasterLayerManifest {
  id: ResourceId;
  name: string;
  revision?: Revision;
  layerName: string;
  width: number;
  height: number;
  rasterTileUrlTemplate: string;
}

export type PropertyKind = "categorical" | "numeric" | "text" | "boolean";
export type PropertyValue = string | number | boolean | null;

export interface StorePropertySchema {
  name: string;
  kind: PropertyKind;
  values?: PropertyValue[];
  min?: number;
  max?: number;
}

export type RepresentationKind =
  | "aggregate"
  | "centroid"
  | "polygon"
  | "auto";

export type ConcreteRepresentationKind = Exclude<RepresentationKind, "auto">;

export interface ZoomRepresentationRange {
  minZoom: number;
  maxZoom: number;
  representation: ConcreteRepresentationKind;
}

export interface RepresentationPolicy {
  scope: string;
  ranges: ZoomRepresentationRange[];
}

export interface RepresentationManifest {
  kind: RepresentationKind;
  minZoom: number;
  maxZoom: number;
  urlTemplate: string;
  maxFeatures?: number;
  maxVertices?: number;
  /** Authoritative server-side policy metadata; requests still use `auto`. */
  policy?: RepresentationPolicy;
}

export interface StoreManifest {
  id: ResourceId;
  name: string;
  revision: Revision;
  count: number;
  bounds: readonly [number, number, number, number];
  overlaps: boolean;
  geometryTypes: Record<string, number>;
  lodStatus?: "not-built" | "building" | "ready" | "failed" | string;
  properties: StorePropertySchema[];
  representations: RepresentationManifest[];
  tileUrlTemplates: Partial<Record<RepresentationKind, string>>;
  pickUrl?: string;
  featureUrlTemplate: string;
}

export interface SessionManifest {
  id: string;
  slide: SlideManifest | null;
  annotationStores: StoreManifest[];
  rasterLayers: RasterLayerManifest[];
}

export type AddedOverlay =
  | { kind: "annotation"; store: StoreManifest }
  | { kind: "raster"; layer: RasterLayerManifest };

export interface FeatureDetail {
  fid: FeatureId;
  uuid?: string;
  geometry: Record<string, unknown> | null;
  properties: Record<string, unknown>;
}

export interface LoadedStore {
  manifest: StoreManifest;
  presentation: import("../domain/style-spec").LayerPresentation;
}

export interface RasterPresentation {
  visible: boolean;
  opacity: number;
}

export interface LoadedRaster {
  manifest: RasterLayerManifest;
  presentation: RasterPresentation;
}
