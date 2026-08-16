import { describe, expect, it, vi } from "vitest";

import TileState from "ol/TileState.js";
import Projection from "ol/proj/Projection.js";
import type RenderFeature from "ol/render/Feature.js";
import TileGrid from "ol/tilegrid/TileGrid.js";

import type { StoreManifest } from "../src/api/types";
import { defaultPresentation } from "../src/domain/style-spec";
import { TileRequestManager } from "../src/renderers/openlayers/tile-request-manager";
import {
  createManagedVectorTileSource,
  loadManagedVectorTile,
} from "../src/renderers/openlayers/vector-tile-source";

const projection = new Projection({ code: "test-slide", units: "pixels" });
const extent: [number, number, number, number] = [0, 0, 256, 256];

describe("managed vector tile loading", () => {
  it("isolates alternate-zoom caches at server-advertised LOD boundaries", () => {
    const resolutions = Array.from({ length: 10 }, (_, zoom) => 2 ** (9 - zoom));
    const tileGrid = new TileGrid({
      extent: [0, -1024, 1024, 0],
      origin: [0, 0],
      resolutions,
      tileSize: 256,
    });
    const store: StoreManifest = {
      id: "cells",
      name: "Cells",
      revision: "policy-1",
      count: 627_761,
      bounds: [0, 0, 1024, 1024],
      overlaps: false,
      geometryTypes: { Polygon: 627_761 },
      properties: [{ name: "type", kind: "categorical", values: [0, 1] }],
      representations: [{
        kind: "auto",
        minZoom: 0,
        maxZoom: 9,
        urlTemplate: "/auto/{z}/{x}/{y}.mvt",
        policy: {
          scope: "store-zoom",
          ranges: [
            { minZoom: 0, maxZoom: 3, representation: "aggregate" },
            { minZoom: 4, maxZoom: 6, representation: "centroid" },
            { minZoom: 7, maxZoom: 9, representation: "polygon" },
          ],
        },
      }],
      tileUrlTemplates: {
        auto: "/auto/{z}/{x}/{y}.mvt",
        aggregate: "/aggregate/{z}/{x}/{y}.mvt",
      },
      featureUrlTemplate: "/features/{fid}",
    };
    const managed = createManagedVectorTileSource({
      slide: {
        id: "slide",
        name: "Slide",
        width: 1024,
        height: 1024,
        mpp: null,
        tileSize: 256,
        maxZoom: 9,
        resolutions,
        mapExtent: [0, -1024, 1024, 0],
        rasterTileUrlTemplate: "/slide/{z}/{x}/{y}",
        associatedOverlays: [],
      },
      store,
      projection,
      tileGrid,
      presentation: defaultPresentation(store),
    });

    const aggregateSource = managed.sourceForResolution(resolutions[3]);
    expect(managed.sourceForResolution(resolutions[2])).toBe(aggregateSource);
    expect(aggregateSource.getTileGrid()?.getMinZoom()).toBe(0);
    expect(aggregateSource.getTileGrid()?.getMaxZoom()).toBe(3);

    const centroidSource = managed.sourceForResolution(resolutions[4]);
    expect(centroidSource).not.toBe(aggregateSource);
    expect(managed.sourceForResolution(resolutions[6])).toBe(centroidSource);
    expect(centroidSource.getTileGrid()?.getMinZoom()).toBe(4);
    expect(centroidSource.getTileGrid()?.getMaxZoom()).toBe(6);
    expect(
      centroidSource.getTileCoordForTileUrlFunction([3, 0, 0], projection),
    ).toBeNull();
    expect(
      centroidSource.getTile(3, 0, 0, 1, projection)?.getState(),
    ).toBe(TileState.EMPTY);
    expect(
      centroidSource.getTileCoordForTileUrlFunction([4, 0, 0], projection),
    ).toEqual([4, 0, 0]);

    const polygonSource = managed.sourceForResolution(resolutions[7]);
    expect(polygonSource).not.toBe(centroidSource);
    expect(polygonSource.getTileGrid()?.getMinZoom()).toBe(7);
    expect(polygonSource.getTileGrid()?.getMaxZoom()).toBe(9);
    expect(
      polygonSource.getTileCoordForTileUrlFunction([6, 0, 0], projection),
    ).toBeNull();
    expect(
      polygonSource.getTile(6, 0, 0, 1, projection)?.getState(),
    ).toBe(TileState.EMPTY);
    expect(polygonSource.getUrls()).toEqual([
      "/auto/{z}/{x}/{y}.mvt?fields=type",
    ]);
  });

  it("terminalizes a superseded tile without writing stale feature data", async () => {
    const requests = new TileRequestManager();
    let resolveResponse: ((response: Response) => void) | undefined;
    const fetcher = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveResponse = resolve;
        }),
    );
    const vectorTile = {
      setFeatures: vi.fn(),
      setState: vi.fn(),
    };
    const format = { readFeatures: vi.fn(() => []) };

    const loading = loadManagedVectorTile({
      vectorTile,
      tileUrl: "/tile/1",
      extent,
      projection,
      format,
      requests,
      fetcher: fetcher as typeof fetch,
      retryDelays: [],
    });
    await vi.waitFor(() => expect(resolveResponse).toBeTypeOf("function"));
    const replacement = requests.start("/tile/1");
    resolveResponse!(new Response(new Uint8Array([1])));
    await loading;

    expect(vectorTile.setFeatures).not.toHaveBeenCalled();
    expect(vectorTile.setState).toHaveBeenCalledOnce();
    expect(vectorTile.setState).toHaveBeenCalledWith(TileState.ERROR);
    expect(format.readFeatures).not.toHaveBeenCalled();
    expect(requests.isCurrent("/tile/1", replacement)).toBe(true);
    requests.abortAll();
  });

  it("does not let an old generation terminalize a reused tile object", async () => {
    const requests = new TileRequestManager();
    let resolveOldResponse: ((response: Response) => void) | undefined;
    const oldFetcher = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveOldResponse = resolve;
        }),
    );
    const oldFeature = { generation: "old" } as unknown as RenderFeature;
    const replacementFeature = {
      generation: "replacement",
    } as unknown as RenderFeature;
    const vectorTile = {
      setFeatures: vi.fn(),
      setState: vi.fn(),
    };
    const oldFormat = { readFeatures: vi.fn(() => [oldFeature]) };
    const replacementFormat = {
      readFeatures: vi.fn(() => [replacementFeature]),
    };

    const oldLoading = loadManagedVectorTile({
      vectorTile,
      tileUrl: "/tile/reused",
      extent,
      projection,
      format: oldFormat,
      requests,
      fetcher: oldFetcher as typeof fetch,
      retryDelays: [],
    });
    await vi.waitFor(() => expect(resolveOldResponse).toBeTypeOf("function"));
    const replacementLoading = loadManagedVectorTile({
      vectorTile,
      tileUrl: "/tile/reused",
      extent,
      projection,
      format: replacementFormat,
      requests,
      fetcher: vi.fn(async () => new Response(new Uint8Array([2]))),
      retryDelays: [],
    });
    await replacementLoading;
    resolveOldResponse!(new Response(new Uint8Array([1])));
    await oldLoading;

    expect(vectorTile.setState).not.toHaveBeenCalledWith(TileState.ERROR);
    expect(vectorTile.setFeatures).toHaveBeenCalledOnce();
    expect(vectorTile.setFeatures).toHaveBeenCalledWith([replacementFeature]);
    expect(oldFormat.readFeatures).not.toHaveBeenCalled();
  });

  it("terminalizes pending tiles immediately when a renderer is disposed", async () => {
    const requests = new TileRequestManager();
    const vectorTile = {
      setFeatures: vi.fn(),
      setState: vi.fn(),
    };
    const format = { readFeatures: vi.fn(() => []) };
    const fetcher = vi.fn(
      (_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => reject(init.signal?.reason),
            { once: true },
          );
        }),
    );

    const loading = loadManagedVectorTile({
      vectorTile,
      tileUrl: "/tile/disposed",
      extent,
      projection,
      format,
      requests,
      fetcher: fetcher as typeof fetch,
      retryDelays: [],
    });
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledOnce());
    requests.abortAll();

    expect(vectorTile.setState).toHaveBeenCalledOnce();
    expect(vectorTile.setState).toHaveBeenCalledWith(TileState.ERROR);
    await loading;
    expect(vectorTile.setFeatures).not.toHaveBeenCalled();
    expect(requests.size).toBe(0);
  });

  it("retries a transient response before completing an empty tile", async () => {
    const requests = new TileRequestManager();
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(new Response(new Uint8Array([1]), { status: 200 }));
    const vectorTile = {
      setFeatures: vi.fn(),
      setState: vi.fn(),
    };
    const format = { readFeatures: vi.fn(() => []) };

    const features = await loadManagedVectorTile({
      vectorTile,
      tileUrl: "/tile/2",
      extent,
      projection,
      format,
      requests,
      fetcher,
      retryDelays: [0],
    });

    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(features).toEqual([]);
    expect(vectorTile.setFeatures).toHaveBeenCalledWith(features);
    expect(vectorTile.setState).not.toHaveBeenCalledWith(TileState.ERROR);
    expect(requests.size).toBe(0);
  });

  it("retries empty HTTP responses instead of caching a broken tile", async () => {
    const requests = new TileRequestManager();
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(new Response(new Uint8Array(), { status: 200 }))
      .mockResolvedValueOnce(new Response(new Uint8Array([1]), { status: 200 }));
    const vectorTile = {
      setFeatures: vi.fn(),
      setState: vi.fn(),
    };
    const format = { readFeatures: vi.fn(() => []) };

    const features = await loadManagedVectorTile({
      vectorTile,
      tileUrl: "/tile/empty-response",
      extent,
      projection,
      format,
      requests,
      fetcher,
      retryDelays: [0, 0],
    });

    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(format.readFeatures).toHaveBeenCalledTimes(1);
    expect(vectorTile.setFeatures).toHaveBeenCalledWith(features);
    expect(vectorTile.setState).not.toHaveBeenCalledWith(TileState.ERROR);
    expect(requests.size).toBe(0);
  });
});
