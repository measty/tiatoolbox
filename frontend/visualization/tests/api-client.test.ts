import { describe, expect, it } from "vitest";

import { ApiClient, expandUrlTemplate, isAbortError } from "../src/api/client";

describe("ApiClient", () => {
  it("invokes native-style fetch with the global receiver", async () => {
    const fetchImplementation = function (this: unknown) {
      expect(this).toBe(globalThis);
      return Promise.resolve(
        new Response(
          JSON.stringify({
            apiVersion: "1.0",
            catalog: { slides: [], overlays: [] },
            session: {
              id: "session",
              slide: null,
              annotationStores: [],
              rasterLayers: [],
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    } as typeof fetch;
    const api = new ApiClient(fetchImplementation);

    await expect(api.bootstrap()).resolves.toMatchObject({ apiVersion: "1.0" });
  });

  it("aborts an in-flight scoped request", async () => {
    const fetchImplementation = ((_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      })) as typeof fetch;
    const api = new ApiClient(fetchImplementation);
    const request = api.bootstrap();
    api.abort("bootstrap");
    await expect(request).rejects.toSatisfy(isAbortError);
  });

  it("escapes values inserted into server URL templates", () => {
    expect(expandUrlTemplate("/features/{featureId}", { featureId: "a/b" })).toBe(
      "/features/a%2Fb",
    );
  });

  it("selects a slide through the session mutation route", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetchImplementation = (async (url: string, init?: RequestInit) => {
      requests.push({ url, init });
      return new Response(
        JSON.stringify({
          id: "slide-a",
          name: "Slide A",
          dimensions: [1024, 512],
          mpp: [0.25, 0.25],
          tileMatrix: {
            tileSize: 256,
            maxZoom: 2,
            resolutions: [4, 2, 1],
            mapExtent: [0, -512, 1024, 0],
          },
          tileUrl: "/tiles/{TileGroup}/{z}-{x}-{y}.jpg",
          associatedOverlays: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof fetch;
    const api = new ApiClient(fetchImplementation);

    await api.selectSlide("opaque/slide");

    expect(requests[0]?.url).toBe("/api/v1/session/slide");
    expect(requests[0]?.init?.method).toBe("PUT");
    expect(requests[0]?.init?.credentials).toBe("same-origin");
    expect(requests[0]?.init?.body).toBe(
      JSON.stringify({ resourceId: "opaque/slide" }),
    );
  });

  it("handles a no-content overlay removal", async () => {
    const fetchImplementation = (async () =>
      new Response(null, { status: 204 })) as typeof fetch;
    const api = new ApiClient(fetchImplementation);
    await expect(api.removeOverlay("loaded/store")).resolves.toBeUndefined();
  });
});
