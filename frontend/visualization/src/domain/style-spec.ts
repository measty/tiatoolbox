import type { PropertyValue, StoreManifest } from "../api/types";

export interface CategoryStyle {
  value: PropertyValue;
  color: string;
  visible: boolean;
}

export interface NumericStyle {
  domain: readonly [number, number];
  colors: readonly [string, string];
}

export type ColorBy =
  | { mode: "constant"; color: string }
  | { mode: "direct"; property: "color"; fallback: string }
  | { mode: "categorical"; property: string; categories: CategoryStyle[] }
  | { mode: "numeric"; property: string; numeric: NumericStyle };

export const DIRECT_COLOR_OPTION = "__feature_color__";

export function supportsDirectColor(store: StoreManifest): boolean {
  return store.properties.some((property) => property.name === "color");
}

export interface RangeFilter {
  property: string;
  min: number;
  max: number;
}

/** How an annotation layer behaves in the precomputed overview LOD range. */
export type OverviewMode = "aggregate" | "hidden";

export interface LayerPresentation {
  visible: boolean;
  opacity: number;
  overviewMode: OverviewMode;
  fillOpacity: number;
  strokeColor: string;
  strokeWidth: number;
  pointRadius: number;
  colorBy: ColorBy;
  rangeFilter?: RangeFilter;
}

const DEFAULT_COLORS = [
  "#e11d48",
  "#2563eb",
  "#16a34a",
  "#ca8a04",
  "#9333ea",
  "#0891b2",
  "#ea580c",
  "#4f46e5",
  "#65a30d",
  "#db2777",
];

export function defaultPresentation(store: StoreManifest): LayerPresentation {
  const categorical = store.properties.find(
    (property) => property.kind === "categorical" && property.values?.length,
  );
  const numeric = store.properties.find(
    (property) =>
      property.kind === "numeric" &&
      property.min !== undefined &&
      property.max !== undefined,
  );
  let colorBy: ColorBy = { mode: "constant", color: "#e11d48" };
  let rangeFilter: RangeFilter | undefined;
  if (categorical?.values) {
    colorBy = {
      mode: "categorical",
      property: categorical.name,
      categories: categorical.values.map((value, index) => ({
        value,
        color: DEFAULT_COLORS[index % DEFAULT_COLORS.length] ?? "#e11d48",
        visible: true,
      })),
    };
  } else if (
    numeric &&
    numeric.min !== undefined &&
    numeric.max !== undefined
  ) {
    colorBy = {
      mode: "numeric",
      property: numeric.name,
      numeric: {
        domain: [numeric.min, numeric.max],
        colors: ["#2563eb", "#f59e0b"],
      },
    };
    rangeFilter = {
      property: numeric.name,
      min: numeric.min,
      max: numeric.max,
    };
  }
  return {
    visible: true,
    opacity: 1,
    overviewMode: "aggregate",
    fillOpacity: 0.55,
    strokeColor: "#111827",
    strokeWidth: 0.75,
    pointRadius: 3,
    colorBy,
    ...(rangeFilter ? { rangeFilter } : {}),
  };
}

/**
 * Return the OpenLayers minimum zoom for this presentation.
 *
 * OpenLayers treats `minZoom` as an exclusive bound, while vector sources use
 * the lower source-tile zoom between integer view zooms.  Placing the bound
 * immediately below the first detail level therefore hides every aggregate
 * tile and admits exact detail tiles from that integer zoom onward.  This
 * deliberately never moves detail rendering into the overview range, where
 * doing so could require an unbounded source-store query.
 */
export function annotationLayerMinZoom(
  store: StoreManifest,
  presentation: LayerPresentation,
): number {
  if (presentation.overviewMode !== "hidden") return -Infinity;
  const overviewMaxZoom = aggregateOverviewMaxZoom(store);
  return overviewMaxZoom === undefined
    ? -Infinity
    : overviewMaxZoom + 1 - 1e-9;
}

/** Return a usable persisted-overview boundary advertised by the manifest. */
export function aggregateOverviewMaxZoom(
  store: StoreManifest,
): number | undefined {
  const aggregate = store.representations.find(
    (representation) => representation.kind === "aggregate",
  );
  const maxZoom = aggregate?.maxZoom;
  return maxZoom !== undefined &&
    Number.isSafeInteger(maxZoom) &&
    maxZoom >= 0 &&
    maxZoom < Number.MAX_SAFE_INTEGER
    ? maxZoom
    : undefined;
}

export function presentationForProperty(
  store: StoreManifest,
  current: LayerPresentation,
  propertyName: string,
): LayerPresentation {
  if (propertyName === "__constant__") {
    return {
      ...current,
      colorBy: { mode: "constant", color: "#e11d48" },
      rangeFilter: undefined,
    };
  }
  if (propertyName === DIRECT_COLOR_OPTION) {
    if (!supportsDirectColor(store)) return current;
    return {
      ...current,
      colorBy: {
        mode: "direct",
        property: "color",
        fallback: "#9ca3af",
      },
      rangeFilter: undefined,
    };
  }
  const schema = store.properties.find((property) => property.name === propertyName);
  if (!schema) return current;
  if (schema.kind === "categorical" && schema.values) {
    return {
      ...current,
      colorBy: {
        mode: "categorical",
        property: schema.name,
        categories: schema.values.map((value, index) => ({
          value,
          color: DEFAULT_COLORS[index % DEFAULT_COLORS.length] ?? "#e11d48",
          visible: true,
        })),
      },
      rangeFilter: undefined,
    };
  }
  if (
    schema.kind === "numeric" &&
    schema.min !== undefined &&
    schema.max !== undefined
  ) {
    return {
      ...current,
      colorBy: {
        mode: "numeric",
        property: schema.name,
        numeric: {
          domain: [schema.min, schema.max],
          colors: ["#2563eb", "#f59e0b"],
        },
      },
      rangeFilter: {
        property: schema.name,
        min: schema.min,
        max: schema.max,
      },
    };
  }
  return current;
}

/** Replace a store-specific style that the loaded manifest cannot satisfy. */
export function presentationForStore(
  store: StoreManifest,
  presentation: LayerPresentation,
): LayerPresentation {
  return presentation.colorBy.mode === "direct" && !supportsDirectColor(store)
    ? defaultPresentation(store)
    : presentation;
}

export function matchesPresentation(
  properties: Record<string, unknown>,
  presentation: LayerPresentation,
): boolean {
  if (presentation.colorBy.mode === "categorical") {
    const value = properties[presentation.colorBy.property];
    const category = presentation.colorBy.categories.find(
      (entry) => String(entry.value) === String(value),
    );
    if (category && !category.visible) return false;
  }
  if (presentation.rangeFilter) {
    const value = Number(properties[presentation.rangeFilter.property]);
    if (
      !Number.isFinite(value) ||
      value < presentation.rangeFilter.min ||
      value > presentation.rangeFilter.max
    ) {
      return false;
    }
  }
  return true;
}

/** Properties that must be embedded in MVTs for this presentation. */
export function requiredTileProperties(
  presentation: LayerPresentation,
): string[] {
  const properties = new Set<string>();
  if (presentation.colorBy.mode !== "constant") {
    properties.add(presentation.colorBy.property);
  }
  if (presentation.rangeFilter) properties.add(presentation.rangeFilter.property);
  return [...properties].sort();
}

/**
 * Upgrade a presentation created from provisional store metadata.
 *
 * LOD construction initially exposes no property summaries, so the default is
 * a constant colour. Once summaries arrive, replace only that untouched
 * colour/range choice with the richer default. Independent appearance edits
 * survive, and an explicitly changed colour/property is never overwritten.
 */
export function refreshProvisionalPresentation(
  current: LayerPresentation,
  previousStore: StoreManifest,
  nextStore: StoreManifest,
): LayerPresentation {
  const previousDefault = defaultPresentation(previousStore);
  if (!sameDataStyle(current, previousDefault)) return current;
  const nextDefault = defaultPresentation(nextStore);
  if (sameDataStyle(current, nextDefault)) return current;
  return {
    ...current,
    colorBy: nextDefault.colorBy,
    rangeFilter: nextDefault.rangeFilter,
  };
}

function sameDataStyle(
  left: LayerPresentation,
  right: LayerPresentation,
): boolean {
  return (
    JSON.stringify(left.colorBy) === JSON.stringify(right.colorBy) &&
    JSON.stringify(left.rangeFilter) === JSON.stringify(right.rangeFilter)
  );
}
