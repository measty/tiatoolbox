import type { RasterPresentation } from "../api/types";
import type {
  CategoryStyle,
  ColorBy,
  LayerPresentation,
  RangeFilter,
} from "./style-spec";

export const VIEWER_CONFIG_SCHEMA = "tiatoolbox.viewer";
export const VIEWER_CONFIG_VERSION = 1 as const;
export const MAX_VIEWER_CONFIG_BYTES = 1_000_000;
const MAX_STYLE_NUMBER = 1e30;

export type ConfigOverlayKind = "annotation" | "raster";
export type ConfigRenderer = "auto" | "canvas" | "webgl";

export type ViewerConfigLayer =
  | {
      kind: "annotation";
      resourceId: string;
      presentation: LayerPresentation;
    }
  | {
      kind: "raster";
      resourceId: string;
      presentation: RasterPresentation;
    };

export interface ViewerConfigV1 {
  schema: typeof VIEWER_CONFIG_SCHEMA;
  version: typeof VIEWER_CONFIG_VERSION;
  slideResourceId: string;
  overlayResourceIds: string[];
  renderer: ConfigRenderer;
  compare: {
    enabled: boolean;
    linked: boolean;
  };
  views: {
    primary: ViewerConfigLayer[];
    secondary: ViewerConfigLayer[];
  };
}

export interface ViewerConfigCatalog {
  slideResourceIds: readonly string[];
  overlayKinds: Readonly<Record<string, ConfigOverlayKind>>;
}

type ConfigState = Omit<ViewerConfigV1, "schema" | "version">;
type JsonObject = Record<string, unknown>;

export function createViewerConfig(state: ConfigState): ViewerConfigV1 {
  const overlayResourceIds = unique(state.overlayResourceIds);
  const allowed = new Set(overlayResourceIds);
  return {
    schema: VIEWER_CONFIG_SCHEMA,
    version: VIEWER_CONFIG_VERSION,
    slideResourceId: state.slideResourceId,
    overlayResourceIds,
    renderer: state.renderer,
    compare: { ...state.compare },
    views: {
      primary: state.views.primary
        .filter((layer) => allowed.has(layer.resourceId))
        .map(cloneLayer),
      secondary: state.views.secondary
        .filter((layer) => allowed.has(layer.resourceId))
        .map(cloneLayer),
    },
  };
}

export function serialiseViewerConfig(config: ViewerConfigV1): string {
  return JSON.stringify(config, null, 2);
}

export function parseViewerConfig(
  text: string,
  catalog: ViewerConfigCatalog,
): ViewerConfigV1 {
  if (new Blob([text]).size > MAX_VIEWER_CONFIG_BYTES) {
    throw new TypeError("Viewer configuration exceeds the 1 MB limit.");
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new TypeError("Viewer configuration is not valid JSON.");
  }
  return normaliseViewerConfig(value, catalog);
}

export function normaliseViewerConfig(
  value: unknown,
  catalog: ViewerConfigCatalog,
): ViewerConfigV1 {
  const raw = record(value, "configuration");
  if (raw.schema !== VIEWER_CONFIG_SCHEMA) {
    throw new TypeError(`Configuration schema must be ${VIEWER_CONFIG_SCHEMA}.`);
  }
  if (raw.version !== VIEWER_CONFIG_VERSION) {
    throw new TypeError(`Unsupported viewer configuration version: ${String(raw.version)}.`);
  }

  const slideResourceId = resourceId(raw.slideResourceId, "slideResourceId");
  if (!catalog.slideResourceIds.includes(slideResourceId)) {
    throw new TypeError("The configured slide resource is not in this catalog.");
  }
  if (!Array.isArray(raw.overlayResourceIds)) {
    throw new TypeError("overlayResourceIds must be an array.");
  }
  if (raw.overlayResourceIds.length > 256) {
    throw new TypeError("A viewer configuration can contain at most 256 overlays.");
  }
  const overlayResourceIds = unique(
    raw.overlayResourceIds.map((entry, index) => {
      const id = resourceId(entry, `overlayResourceIds[${index}]`);
      if (!catalog.overlayKinds[id]) {
        throw new TypeError(`Unknown overlay resource ID: ${id}.`);
      }
      return id;
    }),
  );
  const allowed = new Set(overlayResourceIds);
  const compare = optionalRecord(raw.compare);
  const views = record(raw.views, "views");
  return {
    schema: VIEWER_CONFIG_SCHEMA,
    version: VIEWER_CONFIG_VERSION,
    slideResourceId,
    overlayResourceIds,
    renderer:
      raw.renderer === "canvas" || raw.renderer === "webgl"
        ? raw.renderer
        : "auto",
    compare: {
      enabled: boolean(raw.compare && compare.enabled, false),
      linked: boolean(raw.compare && compare.linked, true),
    },
    views: {
      primary: normaliseView(views.primary, "views.primary", catalog, allowed),
      secondary: normaliseView(
        views.secondary,
        "views.secondary",
        catalog,
        allowed,
      ),
    },
  };
}

function normaliseView(
  value: unknown,
  label: string,
  catalog: ViewerConfigCatalog,
  allowed: Set<string>,
): ViewerConfigLayer[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array.`);
  if (value.length > 256) throw new TypeError(`${label} has too many layers.`);
  const seen = new Set<string>();
  const output: ViewerConfigLayer[] = [];
  value.forEach((entry, index) => {
    const raw = record(entry, `${label}[${index}]`);
    const id = resourceId(raw.resourceId, `${label}[${index}].resourceId`);
    if (!allowed.has(id)) {
      throw new TypeError(`${label} references an overlay that is not loaded: ${id}.`);
    }
    if (seen.has(id)) return;
    seen.add(id);
    const expectedKind = catalog.overlayKinds[id];
    if (raw.kind !== expectedKind) {
      throw new TypeError(`${label} has the wrong layer kind for ${id}.`);
    }
    output.push(
      expectedKind === "annotation"
        ? {
            kind: "annotation",
            resourceId: id,
            presentation: normaliseAnnotationPresentation(raw.presentation),
          }
        : {
            kind: "raster",
            resourceId: id,
            presentation: normaliseRasterPresentation(raw.presentation),
          },
    );
  });
  return output;
}

function normaliseAnnotationPresentation(value: unknown): LayerPresentation {
  const raw = optionalRecord(value);
  const colorBy = normaliseColorBy(raw.colorBy);
  const rangeFilter = normaliseRangeFilter(raw.rangeFilter);
  return {
    visible: boolean(raw.visible, true),
    opacity: boundedNumber(raw.opacity, 1, 0, 1),
    overviewMode: raw.overviewMode === "hidden" ? "hidden" : "aggregate",
    fillOpacity: boundedNumber(raw.fillOpacity, 0.55, 0, 1),
    strokeColor: color(raw.strokeColor, "#111827"),
    strokeWidth: boundedNumber(raw.strokeWidth, 0.75, 0, 16),
    pointRadius: boundedNumber(raw.pointRadius, 3, 0, 64),
    colorBy,
    ...(rangeFilter ? { rangeFilter } : {}),
  };
}

function normaliseRasterPresentation(value: unknown): RasterPresentation {
  const raw = optionalRecord(value);
  return {
    visible: boolean(raw.visible, true),
    opacity: boundedNumber(raw.opacity, 0.65, 0, 1),
  };
}

function normaliseColorBy(value: unknown): ColorBy {
  const raw = optionalRecord(value);
  if (raw.mode === "direct") {
    return {
      mode: "direct",
      property: "color",
      fallback: color(raw.fallback, "#9ca3af"),
    };
  }
  if (raw.mode === "categorical") {
    const categories = Array.isArray(raw.categories)
      ? raw.categories.slice(0, 256).flatMap(normaliseCategory)
      : [];
    return {
      mode: "categorical",
      property: propertyName(raw.property, "type"),
      categories,
    };
  }
  if (raw.mode === "numeric") {
    const numeric = optionalRecord(raw.numeric);
    const domain = finitePair(numeric.domain, [0, 1]);
    const colors = Array.isArray(numeric.colors) ? numeric.colors : [];
    return {
      mode: "numeric",
      property: propertyName(raw.property, "prob"),
      numeric: {
        domain: orderedPair(domain),
        colors: [color(colors[0], "#2563eb"), color(colors[1], "#f59e0b")],
      },
    };
  }
  return { mode: "constant", color: color(raw.color, "#e11d48") };
}

function normaliseCategory(value: unknown): CategoryStyle[] {
  const raw = optionalRecord(value);
  const categoryValue = raw.value;
  if (
    categoryValue !== null &&
    typeof categoryValue !== "string" &&
    typeof categoryValue !== "number" &&
    typeof categoryValue !== "boolean"
  ) {
    return [];
  }
  if (
    typeof categoryValue === "number" &&
    (!Number.isFinite(categoryValue) || Math.abs(categoryValue) > MAX_STYLE_NUMBER)
  ) {
    return [];
  }
  if (typeof categoryValue === "string" && categoryValue.length > 1024) {
    return [];
  }
  return [
    {
      value: categoryValue,
      color: color(raw.color, "#9ca3af"),
      visible: boolean(raw.visible, true),
    },
  ];
}

function normaliseRangeFilter(value: unknown): RangeFilter | undefined {
  if (value === undefined || value === null) return undefined;
  const raw = optionalRecord(value);
  const min = finiteNumber(raw.min);
  const max = finiteNumber(raw.max);
  if (min === undefined || max === undefined) return undefined;
  const [orderedMin, orderedMax] = orderedPair([min, max]);
  return {
    property: propertyName(raw.property, "prob"),
    min: orderedMin,
    max: orderedMax,
  };
}

function finitePair(
  value: unknown,
  fallback: readonly [number, number],
): [number, number] {
  if (!Array.isArray(value)) return [...fallback];
  const first = finiteNumber(value[0]);
  const second = finiteNumber(value[1]);
  return first === undefined || second === undefined
    ? [...fallback]
    : [first, second];
}

function orderedPair(value: readonly [number, number]): [number, number] {
  return value[0] <= value[1] ? [...value] : [value[1], value[0]];
}

function boundedNumber(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
): number {
  const parsed = finiteNumber(value);
  return parsed === undefined ? fallback : Math.max(min, Math.min(max, parsed));
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" &&
    Number.isFinite(value) &&
    Math.abs(value) <= MAX_STYLE_NUMBER
    ? value
    : undefined;
}

function boolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function color(value: unknown, fallback: string): string {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)
    ? value.toLowerCase()
    : fallback;
}

function propertyName(value: unknown, fallback: string): string {
  return typeof value === "string" && /^[^\u0000-\u001f]{1,128}$/.test(value)
    ? value
    : fallback;
}

function resourceId(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 512) {
    throw new TypeError(`${label} must be a non-empty resource ID.`);
  }
  return value;
}

function record(value: unknown, label: string): JsonObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
  return value as JsonObject;
}

function optionalRecord(value: unknown): JsonObject {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function cloneLayer(layer: ViewerConfigLayer): ViewerConfigLayer {
  return JSON.parse(JSON.stringify(layer)) as ViewerConfigLayer;
}
