import { describe, expect, it } from "vitest";

import View from "ol/View.js";

import { ViewLinkController } from "../src/map/view-link-controller";

describe("ViewLinkController", () => {
  it("synchronises view state in canonical slide coordinates", () => {
    const controller = new ViewLinkController();
    const first = new View({ center: [10, -20], resolution: 4, rotation: 0 });
    const second = new View({ center: [0, 0], resolution: 1, rotation: 0 });
    const unregisterFirst = controller.register("first", first);
    const unregisterSecond = controller.register("second", second);

    first.setCenter([100, -200]);
    first.setResolution(2);
    first.setRotation(0.25);

    expect(second.getCenter()).toEqual([100, -200]);
    expect(second.getResolution()).toBe(2);
    expect(second.getRotation()).toBe(0.25);

    unregisterFirst();
    unregisterSecond();
    controller.dispose();
  });

  it("leaves views independent while unlinked", () => {
    const controller = new ViewLinkController();
    const first = new View({ center: [0, 0], resolution: 1 });
    const second = new View({ center: [0, 0], resolution: 1 });
    controller.register("first", first);
    controller.register("second", second);
    controller.setEnabled(false);
    first.setCenter([50, -60]);
    expect(second.getCenter()).toEqual([0, 0]);
  });

  it("does not restore the previous slide viewport when asked to fit a new slide", () => {
    const controller = new ViewLinkController();
    const previous = new View({ center: [100, -200], resolution: 2 });
    const unregister = controller.register("primary", previous);
    unregister();

    const replacement = new View();
    controller.register("primary", replacement, undefined, {
      restoreLastState: false,
    });

    expect(replacement.getCenter()).toBeUndefined();
    expect(replacement.getResolution()).toBeUndefined();
  });
});
