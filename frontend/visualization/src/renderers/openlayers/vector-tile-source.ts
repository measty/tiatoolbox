import MVT from "ol/format/MVT.js";
import type { Extent } from "ol/extent.js";
import type Projection from "ol/proj/Projection.js";
import type RenderFeature from "ol/render/Feature.js";
import TileGrid from "ol/tilegrid/TileGrid.js";
import TileState from "ol/TileState.js";
import type VectorTile from "ol/VectorTile.js";
import VectorTileSource, {
  type Options as VectorTileSourceOptions,
} from "ol/source/VectorTile.js";

import type { AnnotationRendererContext } from "../annotation-renderer";
import {
  requiredTileProperties,
  zoomRepresentationPolicyRanges,
} from "../../domain/style-spec";
import { TileRequestManager } from "./tile-request-manager";

export interface ManagedVectorTileSource {
  source: VectorTileSource<RenderFeature>;
  requests: TileRequestManager;
  /** Return a source whose tile grid is confined to one semantic LOD band. */
  sourceForResolution(
    resolution: number | undefined,
  ): VectorTileSource<RenderFeature>;
  disposeSources(): void;
}

interface ManagedTileLoaderOptions {
  fetcher?: typeof fetch;
  retryDelays?: readonly number[];
}

const TILE_RETRY_DELAYS = [100, 300] as const;

/**
 * OpenLayers normally expands a vector source's render grid down to z0 and up
 * to its global default max zoom. Viewer sources and views share the same
 * slide projection, so retaining the supplied grid is both correct and what
 * prevents alternate-zoom rendering from crossing a semantic LOD boundary.
 */
class BandLimitedVectorTileSource extends VectorTileSource<RenderFeature> {
  constructor(
    private readonly bandGrid: TileGrid,
    options: VectorTileSourceOptions<RenderFeature>,
  ) {
    super(options);
  }

  override getTileGridForProjection(_projection: Projection): TileGrid {
    return this.bandGrid;
  }
}

export function createManagedVectorTileSource(
  context: AnnotationRendererContext,
): ManagedVectorTileSource {
  const requests = new TileRequestManager();
  const format = new MVT();
  const template =
    context.store.tileUrlTemplates.auto ??
    context.store.representations.find((item) => item.kind === "auto")
      ?.urlTemplate;
  if (!template) throw new Error(`Store ${context.store.id} has no vector tile URL.`);
  const separator = template.includes("?") ? "&" : "?";
  const fields = requiredTileProperties(context.presentation).join(",");
  const url = `${template}${separator}fields=${encodeURIComponent(fields)}`;

  const policyRanges = zoomRepresentationPolicyRanges(context.store);
  const completePolicy =
    policyRanges.length > 0 &&
    policyRanges.at(-1)?.maxZoom === context.tileGrid.getMaxZoom();
  const sourceBands = completePolicy
    ? policyRanges.map((range) => ({
        minZoom: range.minZoom,
        maxZoom: range.maxZoom,
        source: createSource(bandTileGrid(
          context.tileGrid,
          range.minZoom,
          range.maxZoom,
        )),
      }))
    : [{
        minZoom: context.tileGrid.getMinZoom(),
        maxZoom: context.tileGrid.getMaxZoom(),
        source: createSource(context.tileGrid),
      }];
  const source = sourceBands[0]!.source;
  return {
    source,
    requests,
    sourceForResolution(resolution) {
      if (resolution === undefined || sourceBands.length === 1) return source;
      const zoom = context.tileGrid.getZForResolution(
        resolution,
        source.zDirection,
      );
      return sourceBands.find(
        (band) => zoom >= band.minZoom && zoom <= band.maxZoom,
      )?.source ?? source;
    },
    disposeSources() {
      for (const band of sourceBands) band.source.dispose();
    },
  };

  function createSource(tileGrid: TileGrid): VectorTileSource<RenderFeature> {
    return new BandLimitedVectorTileSource(tileGrid, {
      format,
      overlaps: context.store.overlaps,
      projection: context.projection,
      tileGrid,
      url,
      wrapX: false,
      transition: 0,
      tileLoadFunction: (tile, tileUrl) => {
        const vectorTile = tile as VectorTile<RenderFeature>;
        vectorTile.setLoader(async (
          extent: Extent,
          _resolution: number,
          projection: Projection,
        ) => {
          return loadManagedVectorTile({
            vectorTile,
            tileUrl,
            extent,
            projection,
            format,
            requests,
          });
        });
      },
    });
  }
}

/** Clone a tile grid while making other representation zooms unreachable. */
function bandTileGrid(base: TileGrid, minZoom: number, maxZoom: number): TileGrid {
  const resolutions = base.getResolutions().slice(0, maxZoom + 1);
  const origins = resolutions.map((_, zoom) => [...base.getOrigin(zoom)]);
  const tileSizes = resolutions.map((_, zoom) => {
    const size = base.getTileSize(zoom);
    return typeof size === "number" ? size : [...size];
  });
  const extent = base.getExtent();
  return new TileGrid({
    ...(extent ? { extent: [...extent] } : {}),
    minZoom,
    origins,
    resolutions,
    tileSizes,
  });
}

export async function loadManagedVectorTile({
  vectorTile,
  tileUrl,
  extent,
  projection,
  format,
  requests,
  fetcher = fetch,
  retryDelays = TILE_RETRY_DELAYS,
}: {
  vectorTile: Pick<VectorTile<RenderFeature>, "setFeatures" | "setState">;
  tileUrl: string;
  extent: Extent;
  projection: Projection;
  format: Pick<MVT<RenderFeature>, "readFeatures">;
  requests: TileRequestManager;
} & ManagedTileLoaderOptions): Promise<RenderFeature[]> {
  const signal = requests.start(tileUrl, {
    owner: vectorTile,
    // OpenLayers has no cancelled tile state. ERROR is the only terminal state
    // which releases its tile queue without caching cancelled work as data.
    onAbandon: () => vectorTile.setState(TileState.ERROR),
  });
  try {
    for (let attempt = 0; ; attempt += 1) {
      try {
        const response = await fetcher.call(globalThis, tileUrl, {
          signal,
          headers: { Accept: "application/vnd.mapbox-vector-tile" },
        });
        if (!requests.isCurrent(tileUrl, signal)) return [];
        if (!response.ok || response.status === 204) {
          throw new TileLoadError(
            `Tile request failed (${response.status}).`,
            response.status === 204 || retryableStatus(response.status),
          );
        }
        const buffer = await response.arrayBuffer();
        if (!requests.isCurrent(tileUrl, signal)) return [];
        if (buffer.byteLength === 0) {
          throw new TileLoadError("Tile response was empty.", true);
        }
        const parsed = format.readFeatures(buffer, {
          extent,
          featureProjection: projection,
        });
        if (!requests.isCurrent(tileUrl, signal)) return [];
        vectorTile.setFeatures(parsed);
        return parsed;
      } catch (error) {
        if (!requests.isCurrent(tileUrl, signal) || signal.aborted) return [];
        const retryDelay = retryDelays[attempt];
        if (retryDelay === undefined || !retryableError(error)) throw error;
        await waitForRetry(retryDelay, signal);
      }
    }
  } catch (error) {
    if (requests.isCurrent(tileUrl, signal) && !signal.aborted) {
      vectorTile.setState(TileState.ERROR);
      console.error("Unable to load annotation vector tile", error);
    }
    return [];
  } finally {
    requests.finish(tileUrl, signal);
  }
}

class TileLoadError extends Error {
  constructor(
    message: string,
    readonly retryable: boolean,
  ) {
    super(message);
  }
}

function retryableStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function retryableError(error: unknown): boolean {
  return !(error instanceof TileLoadError) || error.retryable;
}

function waitForRetry(delay: number, signal: AbortSignal): Promise<void> {
  if (delay <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, delay);
    const abort = () => {
      globalThis.clearTimeout(timeout);
      reject(signal.reason);
    };
    signal.addEventListener("abort", abort, { once: true });
  });
}
