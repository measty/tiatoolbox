import type {
  AddedOverlay,
  BootstrapManifest,
  FeatureDetail,
  PropertyKind,
  PropertyValue,
  RepresentationKind,
  RepresentationManifest,
  RasterLayerManifest,
  ResourceKind,
  ResourceSummary,
  SessionManifest,
  SlideManifest,
  StoreManifest,
  StorePropertySchema,
} from "./types";

type JsonObject = Record<string, unknown>;

function object(value: unknown, label: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be a JSON object.`);
  }
  return value as JsonObject;
}

function first(record: JsonObject, ...keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined) return record[key];
  }
  return undefined;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  const result = typeof value === "string" ? Number(value) : value;
  if (typeof result !== "number" || !Number.isFinite(result)) {
    throw new TypeError(`${label} must be a finite number.`);
  }
  return result;
}

function optionalFiniteNumber(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined;
  const result = typeof value === "string" ? Number(value) : value;
  return typeof result === "number" && Number.isFinite(result)
    ? result
    : undefined;
}

function resource(value: unknown, label: string): ResourceSummary {
  if (typeof value === "string") return { id: value, name: value };
  const raw = object(value, label);
  const id = text(first(raw, "id", "resource_id"), `${label}.id`);
  const nameValue = first(raw, "name", "label", "title");
  const revisionValue = first(raw, "revision", "rev");
  const kindValue = first(raw, "kind", "resource_kind");
  return {
    id,
    name: typeof nameValue === "string" && nameValue.length > 0 ? nameValue : id,
    ...(kindValue === "slide" ||
    kindValue === "annotation" ||
    kindValue === "raster-overlay"
      ? { kind: kindValue as ResourceKind }
      : {}),
    ...(revisionValue === undefined
      ? {}
      : { revision: String(revisionValue) }),
  };
}

export function normaliseBootstrap(value: unknown): BootstrapManifest {
  const raw = object(value, "bootstrap");
  const catalogValue = first(raw, "catalog");
  const catalog =
    catalogValue === undefined ? raw : object(catalogValue, "bootstrap.catalog");
  const slidesValue = first(catalog, "slides", "slide_resources");
  const overlaysValue = first(
    catalog,
    "overlays",
    "stores",
    "annotation_stores",
  );
  if (!Array.isArray(slidesValue) || !Array.isArray(overlaysValue)) {
    throw new TypeError(
      "bootstrap.catalog.slides and bootstrap.catalog.overlays must be arrays.",
    );
  }
  const slides = slidesValue.map((entry, index) =>
    resource(entry, `bootstrap.slides[${index}]`),
  );
  const overlays = overlaysValue.map((entry, index) =>
    resource(entry, `bootstrap.catalog.overlays[${index}]`),
  );
  const defaultStores = first(raw, "defaultStoreIds", "default_store_ids");
  const sessionValue = first(raw, "session");
  const session =
    sessionValue === undefined
      ? emptySession()
      : normaliseSession(sessionValue);
  return {
    apiVersion: String(first(raw, "apiVersion", "api_version") ?? "v1"),
    title: String(first(raw, "title", "name") ?? "TIAToolbox Viewer"),
    slides,
    overlays,
    session,
    capabilities:
      first(raw, "capabilities") &&
      typeof first(raw, "capabilities") === "object" &&
      !Array.isArray(first(raw, "capabilities"))
        ? (first(raw, "capabilities") as JsonObject)
        : {},
    stores: overlays.filter((entry) => entry.kind !== "raster-overlay"),
    defaultSlideId:
      (first(raw, "defaultSlideId", "default_slide_id") as string | undefined) ??
      session.slide?.id ??
      slides[0]?.id,
    defaultStoreIds: Array.isArray(defaultStores)
      ? defaultStores.map(String)
      : [],
  };
}

function dimensions(
  raw: JsonObject,
  fallback: JsonObject | undefined = undefined,
): readonly [number, number] {
  const value = first(raw, "dimensions", "slide_dimensions", "size");
  if (Array.isArray(value) && value.length >= 2) {
    return [
      finiteNumber(value[0], "slide.dimensions[0]"),
      finiteNumber(value[1], "slide.dimensions[1]"),
    ];
  }
  return [
    finiteNumber(first(raw, "width") ?? fallback?.width, "slide.width"),
    finiteNumber(first(raw, "height") ?? fallback?.height, "slide.height"),
  ];
}

function mpp(raw: JsonObject): readonly [number, number] | null {
  const value = first(raw, "mpp", "microns_per_pixel");
  if (value === undefined || value === null) return null;
  if (Array.isArray(value) && value.length >= 2) {
    return [finiteNumber(value[0], "slide.mpp[0]"), finiteNumber(value[1], "slide.mpp[1]")];
  }
  const scalar = finiteNumber(value, "slide.mpp");
  return [scalar, scalar];
}

export function normaliseSlide(value: unknown): SlideManifest {
  const raw = object(value, "slide");
  const matrixValue = first(raw, "tileMatrix", "tile_matrix");
  const matrix =
    matrixValue === undefined
      ? undefined
      : object(matrixValue, "slide.tileMatrix");
  const [width, height] = dimensions(raw, matrix);
  const id = text(first(raw, "id", "resource_id"), "slide.id");
  const tileSize = finiteNumber(
    first(matrix ?? {}, "tileSize", "tile_size") ??
      first(raw, "tileSize", "tile_size") ??
      256,
    "slide.tile_size",
  );
  const computedMaxZoom = Math.max(
    0,
    Math.ceil(Math.log2(Math.max(width, height) / tileSize)),
  );
  const raster = first(
    raw,
    "rasterTileUrlTemplate",
    "raster_tile_url_template",
    "tileUrlTemplate",
    "tile_url_template",
    "zoomify_url_template",
    "tileUrl",
    "tile_url",
  );
  const maxZoomValue =
    first(matrix ?? {}, "maxZoom", "max_zoom") ??
    first(raw, "maxZoom", "max_zoom") ??
    computedMaxZoom;
  const maxZoom = finiteNumber(maxZoomValue, "slide.max_zoom");
  const resolutionsValue =
    first(matrix ?? {}, "resolutions") ?? first(raw, "resolutions");
  const resolutions = Array.isArray(resolutionsValue)
    ? resolutionsValue.map((item, index) =>
        finiteNumber(item, `slide.resolutions[${index}]`),
      )
    : Array.from(
        { length: maxZoom + 1 },
        (_, zoom) => 2 ** (maxZoom - zoom),
      );
  const mapExtentValue =
    first(matrix ?? {}, "mapExtent", "map_extent") ??
    first(raw, "mapExtent", "map_extent");
  const mapExtent =
    Array.isArray(mapExtentValue) && mapExtentValue.length >= 4
      ? (mapExtentValue.slice(0, 4).map((entry, index) =>
          finiteNumber(entry, `slide.mapExtent[${index}]`),
        ) as [number, number, number, number])
      : ([0, -height, width, 0] as const);
  const associatedValue = first(
    raw,
    "associatedOverlays",
    "associated_overlays",
  );
  return {
    id,
    name: String(first(raw, "name", "label", "title") ?? id),
    ...(first(raw, "revision", "rev") === undefined
      ? {}
      : { revision: String(first(raw, "revision", "rev")) }),
    width,
    height,
    mpp: mpp(raw),
    ...(optionalFiniteNumber(
      first(raw, "objectivePower", "objective_power"),
    ) === undefined
      ? {}
      : {
          objectivePower: optionalFiniteNumber(
            first(raw, "objectivePower", "objective_power"),
          ),
        }),
    tileSize,
    maxZoom,
    resolutions,
    mapExtent,
    rasterTileUrlTemplate: text(raster, "slide.raster_tile_url_template"),
    associatedOverlays: Array.isArray(associatedValue)
      ? associatedValue.map((entry, index) =>
          resource(entry, `slide.associatedOverlays[${index}]`),
        )
      : [],
  };
}

export function normaliseRasterLayer(value: unknown): RasterLayerManifest {
  const raw = object(value, "raster layer");
  const [width, height] = dimensions(raw);
  const id = text(first(raw, "id", "resource_id"), "raster layer.id");
  return {
    id,
    name: String(first(raw, "name", "label", "title") ?? id),
    ...(first(raw, "revision", "rev") === undefined
      ? {}
      : { revision: String(first(raw, "revision", "rev")) }),
    layerName: String(first(raw, "layerName", "layer_name") ?? id),
    width,
    height,
    rasterTileUrlTemplate: text(
      first(
        raw,
        "tileUrl",
        "tile_url",
        "rasterTileUrlTemplate",
        "raster_tile_url_template",
      ),
      "raster layer.tileUrl",
    ),
  };
}

function propertyKind(value: unknown, schema: JsonObject): PropertyKind {
  const raw = String(value ?? "").toLowerCase();
  if (["number", "numeric", "float", "integer", "continuous"].includes(raw)) {
    return "numeric";
  }
  if (["bool", "boolean"].includes(raw)) return "boolean";
  if (["category", "categorical", "enum"].includes(raw)) return "categorical";
  if (Array.isArray(first(schema, "values", "categories"))) return "categorical";
  return "text";
}

function normaliseProperty(name: string, value: unknown): StorePropertySchema {
  const raw = object(value, `store.properties.${name}`);
  const categories = first(raw, "values", "categories");
  const values = Array.isArray(categories)
    ? categories.map((entry) => {
        if (entry && typeof entry === "object" && !Array.isArray(entry)) {
          return (entry as JsonObject).value as PropertyValue;
        }
        return entry as PropertyValue;
      })
    : undefined;
  const numericValue = first(raw, "numeric");
  const numeric =
    numericValue && typeof numericValue === "object" && !Array.isArray(numericValue)
      ? (numericValue as JsonObject)
      : {};
  const min = optionalFiniteNumber(
    first(raw, "min", "minimum") ?? first(numeric, "min", "minimum"),
  );
  const max = optionalFiniteNumber(
    first(raw, "max", "maximum") ?? first(numeric, "max", "maximum"),
  );
  return {
    name: String(first(raw, "name") ?? name),
    kind: propertyKind(first(raw, "kind", "type", "data_type"), raw),
    ...(values ? { values } : {}),
    ...(min === undefined ? {} : { min }),
    ...(max === undefined ? {} : { max }),
  };
}

function properties(raw: JsonObject): StorePropertySchema[] {
  const value = first(
    raw,
    "properties",
    "property_schemas",
    "property_schema",
    "schema",
  );
  if (Array.isArray(value)) {
    return value.map((item, index) => {
      const entry = object(item, `store.properties[${index}]`);
      return normaliseProperty(
        text(first(entry, "name", "id"), `store.properties[${index}].name`),
        entry,
      );
    });
  }
  if (value && typeof value === "object") {
    return Object.entries(value as JsonObject).map(([name, schema]) =>
      normaliseProperty(name, schema),
    );
  }
  return [];
}

const REPRESENTATIONS: RepresentationKind[] = [
  "aggregate",
  "centroid",
  "polygon",
  "auto",
];

function representation(
  value: unknown,
  index: number,
): RepresentationManifest {
  const raw = object(value, `store.representations[${index}]`);
  const kind = String(first(raw, "kind", "name")) as RepresentationKind;
  if (!REPRESENTATIONS.includes(kind)) {
    throw new TypeError(`Unsupported representation kind: ${kind}`);
  }
  return {
    kind,
    minZoom: finiteNumber(first(raw, "minZoom", "min_zoom") ?? 0, "min_zoom"),
    maxZoom: finiteNumber(
      first(raw, "maxZoom", "max_zoom") ?? Number.MAX_SAFE_INTEGER,
      "max_zoom",
    ),
    urlTemplate: text(
      first(raw, "urlTemplate", "url_template", "tile_url_template"),
      "representation.url_template",
    ),
    ...(optionalFiniteNumber(first(raw, "maxFeatures", "max_features")) === undefined
      ? {}
      : { maxFeatures: optionalFiniteNumber(first(raw, "maxFeatures", "max_features")) }),
    ...(optionalFiniteNumber(first(raw, "maxVertices", "max_vertices")) === undefined
      ? {}
      : { maxVertices: optionalFiniteNumber(first(raw, "maxVertices", "max_vertices")) }),
  };
}

function urlTemplates(raw: JsonObject): Partial<Record<RepresentationKind, string>> {
  const result: Partial<Record<RepresentationKind, string>> = {};
  const urlsValue = first(raw, "urls");
  const urls =
    urlsValue && typeof urlsValue === "object" && !Array.isArray(urlsValue)
      ? (urlsValue as JsonObject)
      : undefined;
  const autoUrl = urls && first(urls, "tiles", "tile");
  if (typeof autoUrl === "string") result.auto = autoUrl;
  const value =
    first(raw, "tileUrlTemplates", "tile_url_templates", "tiles") ??
    (urls && first(urls, "representations"));
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const kind of REPRESENTATIONS) {
      const candidate = (value as JsonObject)[kind];
      if (typeof candidate === "string") result[kind] = candidate;
      else if (candidate && typeof candidate === "object") {
        const nested = first(candidate as JsonObject, "url", "url_template");
        if (typeof nested === "string") result[kind] = nested;
      }
    }
  }
  return result;
}

export function normaliseStore(value: unknown): StoreManifest {
  const raw = object(value, "store");
  const id = text(first(raw, "id", "resource_id"), "store.id");
  const revision = text(first(raw, "revision", "rev"), "store.revision");
  const boundsValue = first(raw, "bounds", "extent");
  if (!Array.isArray(boundsValue) || boundsValue.length < 4) {
    throw new TypeError("store.bounds must contain four numbers.");
  }
  const templates = urlTemplates(raw);
  const repsValue = first(raw, "representations", "lod_representations");
  const representations = Array.isArray(repsValue)
    ? repsValue.map(representation)
    : repsValue && typeof repsValue === "object"
      ? Object.entries(repsValue as JsonObject).flatMap(([kind, metadata], index) => {
          if (!REPRESENTATIONS.includes(kind as RepresentationKind)) return [];
          const rawMetadata = object(
            metadata,
            `store.representations.${kind}`,
          );
          const urlTemplate = templates[kind as RepresentationKind];
          if (!urlTemplate) return [];
          return [
            representation(
              { ...rawMetadata, kind, urlTemplate },
              index,
            ),
          ];
        })
      : [];
  for (const rep of representations) templates[rep.kind] = rep.urlTemplate;
  const fallbackTemplate =
    templates.auto ??
    `/api/v1/stores/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revision)}/tiles/auto/{z}/{x}/{y}.mvt`;
  templates.auto = fallbackTemplate;
  return {
    id,
    name: String(first(raw, "name", "label", "title") ?? id),
    revision,
    count: finiteNumber(
      first(raw, "featureCount", "feature_count", "count", "annotation_count") ??
        0,
      "store.count",
    ),
    bounds: [
      finiteNumber(boundsValue[0] ?? 0, "store.bounds[0]"),
      finiteNumber(boundsValue[1] ?? 0, "store.bounds[1]"),
      finiteNumber(boundsValue[2] ?? 0, "store.bounds[2]"),
      finiteNumber(boundsValue[3] ?? 0, "store.bounds[3]"),
    ],
    overlaps: Boolean(first(raw, "overlaps", "has_overlaps") ?? true),
    geometryTypes: normaliseGeometryTypes(
      first(raw, "geometryTypes", "geometry_types"),
    ),
    ...(typeof first(raw, "lodStatus", "lod_status") === "string"
      ? { lodStatus: String(first(raw, "lodStatus", "lod_status")) }
      : {}),
    properties: properties(raw),
    representations,
    tileUrlTemplates: templates,
    ...(pickUrl(raw) ? { pickUrl: pickUrl(raw) } : {}),
    featureUrlTemplate: String(
      first(raw, "featureUrlTemplate", "feature_url_template") ??
        featureUrl(raw) ??
        `/api/v1/stores/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revision)}/features/{fid}`,
    ),
  };
}

function featureUrl(raw: JsonObject): string | undefined {
  const value = first(raw, "urls");
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const candidate = first(value as JsonObject, "feature", "features");
  return typeof candidate === "string" ? candidate : undefined;
}

function pickUrl(raw: JsonObject): string | undefined {
  const value = first(raw, "urls");
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const candidate = first(value as JsonObject, "pick", "featurePick");
  return typeof candidate === "string" ? candidate : undefined;
}

function normaliseGeometryTypes(value: unknown): Record<string, number> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value as JsonObject).flatMap(([name, count]) => {
      const numeric = optionalFiniteNumber(count);
      return numeric === undefined ? [] : [[name, numeric]];
    }),
  );
}

export function normaliseSession(value: unknown): SessionManifest {
  const raw = object(value, "session");
  const slideValue = first(raw, "slide");
  const storesValue = first(raw, "annotationStores", "annotation_stores", "stores");
  const rastersValue = first(raw, "rasterLayers", "raster_layers");
  return {
    id: String(first(raw, "id", "sessionId", "session_id") ?? ""),
    slide:
      slideValue === null || slideValue === undefined
        ? null
        : normaliseSlide(slideValue),
    annotationStores: Array.isArray(storesValue)
      ? storesValue.map(normaliseStore)
      : [],
    rasterLayers: Array.isArray(rastersValue)
      ? rastersValue.map(normaliseRasterLayer)
      : [],
  };
}

function emptySession(): SessionManifest {
  return { id: "", slide: null, annotationStores: [], rasterLayers: [] };
}

export function normaliseAddedOverlay(value: unknown): AddedOverlay {
  const raw = object(value, "overlay response");
  const kind = first(raw, "kind");
  if (kind === "annotation") {
    return { kind, store: normaliseStore(first(raw, "store")) };
  }
  if (kind === "raster") {
    return { kind, layer: normaliseRasterLayer(first(raw, "layer")) };
  }
  throw new TypeError(`Unsupported loaded overlay kind: ${String(kind)}`);
}

export function normaliseFeatureDetail(value: unknown): FeatureDetail {
  const raw = object(value, "feature");
  const fidValue = first(raw, "fid", "id", "feature_id");
  if (typeof fidValue !== "number" && typeof fidValue !== "string") {
    throw new TypeError("feature.fid must be a number or string.");
  }
  let geometry = first(raw, "geometry", "geom") ?? null;
  if (typeof geometry === "string") {
    try {
      geometry = JSON.parse(geometry) as Record<string, unknown>;
    } catch {
      throw new TypeError("feature.geometry must be GeoJSON.");
    }
  }
  const propertiesValue = first(raw, "properties", "props") ?? {};
  const featureProperties = object(propertiesValue, "feature.properties");
  const toolboxValue = first(featureProperties, "_tiatoolbox");
  const toolbox =
    toolboxValue && typeof toolboxValue === "object" && !Array.isArray(toolboxValue)
      ? (toolboxValue as JsonObject)
      : undefined;
  const canonicalId = toolbox && first(toolbox, "canonicalId", "canonical_id");
  return {
    fid: fidValue,
    ...(first(raw, "uuid", "key") === undefined && canonicalId === undefined
      ? {}
      : { uuid: String(first(raw, "uuid", "key") ?? canonicalId) }),
    geometry:
      geometry !== null && typeof geometry === "object"
        ? (geometry as Record<string, unknown>)
        : null,
    properties: featureProperties,
  };
}
