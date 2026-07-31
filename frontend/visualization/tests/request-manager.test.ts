import { describe, expect, it } from "vitest";

import { TileRequestManager } from "../src/renderers/openlayers/tile-request-manager";

describe("TileRequestManager", () => {
  it("aborts superseded requests for the same tile", () => {
    const manager = new TileRequestManager();
    const first = manager.start("tile");
    const second = manager.start("tile");
    expect(first.aborted).toBe(true);
    expect(second.aborted).toBe(false);
    expect(manager.size).toBe(1);
  });

  it("leaves distinct tile requests to OpenLayers' bounded map queue", () => {
    const manager = new TileRequestManager();
    const first = manager.start("first");
    const second = manager.start("second");
    const third = manager.start("third");

    expect(first.aborted).toBe(false);
    expect(second.aborted).toBe(false);
    expect(third.aborted).toBe(false);
    expect(manager.size).toBe(3);
  });

  it("does not let a stale request finish its replacement", () => {
    const manager = new TileRequestManager();
    const original = manager.start("tile");
    const replacement = manager.start("tile");

    expect(original.aborted).toBe(true);
    expect(manager.isCurrent("tile", original)).toBe(false);
    expect(manager.isCurrent("tile", replacement)).toBe(true);
    manager.finish("tile", original);
    expect(replacement.aborted).toBe(false);
    expect(manager.size).toBe(1);
  });

  it("terminalizes an abandoned owner unless ownership was transferred", () => {
    const manager = new TileRequestManager();
    const firstOwner = {};
    const replacementOwner = {};
    let firstAbandoned = 0;
    let replacementAbandoned = 0;
    manager.start("tile", {
      owner: firstOwner,
      onAbandon: () => firstAbandoned += 1,
    });
    manager.start("tile", {
      owner: replacementOwner,
      onAbandon: () => replacementAbandoned += 1,
    });

    expect(firstAbandoned).toBe(1);
    expect(replacementAbandoned).toBe(0);

    manager.start("tile", {
      owner: replacementOwner,
      onAbandon: () => replacementAbandoned += 10,
    });
    expect(replacementAbandoned).toBe(0);
  });

  it("aborts every request when a renderer is disposed", () => {
    const manager = new TileRequestManager();
    let terminalized = 0;
    const first = manager.start("a", {
      owner: {},
      onAbandon: () => terminalized += 1,
    });
    const second = manager.start("b", {
      owner: {},
      onAbandon: () => terminalized += 1,
    });
    manager.abortAll();
    expect(first.aborted).toBe(true);
    expect(second.aborted).toBe(true);
    expect(terminalized).toBe(2);
    expect(manager.size).toBe(0);
  });
});
