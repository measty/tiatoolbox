import { describe, expect, it, vi } from "vitest";

import { reconcileKeyedLayers } from "../src/map/keyed-layer-reconciler";

interface Item {
  id: string;
  identity: string;
}

describe("keyed layer reconciliation", () => {
  it("preserves unchanged resources while updating order and replacing only changes", () => {
    const entries = new Map();
    const create = vi.fn((item: Item) => ({ createdFor: item.id }));
    const update = vi.fn();
    const dispose = vi.fn();
    const reconciler = {
      key: (item: Item) => item.id,
      identity: (item: Item) => item.identity,
      create,
      update,
      dispose,
    };

    reconcileKeyedLayers(entries, [
      { id: "a", identity: "a1" },
      { id: "b", identity: "b1" },
    ], reconciler);
    const originalA = entries.get("a")?.value;
    const originalB = entries.get("b")?.value;
    create.mockClear();
    update.mockClear();

    reconcileKeyedLayers(entries, [
      { id: "b", identity: "b1" },
      { id: "a", identity: "a2" },
      { id: "c", identity: "c1" },
    ], reconciler);

    expect([...entries.keys()]).toEqual(["b", "a", "c"]);
    expect(entries.get("b")?.value).toBe(originalB);
    expect(entries.get("a")?.value).not.toBe(originalA);
    expect(create).toHaveBeenCalledTimes(2);
    expect(dispose).toHaveBeenCalledOnce();
    expect(dispose).toHaveBeenCalledWith(originalA);
    expect(update.mock.calls.map(([, , order]) => order)).toEqual([0, 1, 2]);
  });

  it("disposes resources removed from the desired layer set", () => {
    const entries = new Map();
    const dispose = vi.fn();
    const reconciler = {
      key: (item: Item) => item.id,
      identity: (item: Item) => item.identity,
      create: (item: Item) => ({ createdFor: item.id }),
      update: vi.fn(),
      dispose,
    };
    reconcileKeyedLayers(entries, [{ id: "a", identity: "a1" }], reconciler);
    const value = entries.get("a")?.value;

    reconcileKeyedLayers(entries, [], reconciler);

    expect(entries.size).toBe(0);
    expect(dispose).toHaveBeenCalledWith(value);
  });
});
