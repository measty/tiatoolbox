export interface ManagedLayerEntry<T> {
  identity: string;
  value: T;
}

export interface KeyedLayerReconciler<TItem, TValue> {
  key(item: TItem): string;
  identity(item: TItem): string;
  create(item: TItem): TValue;
  update(value: TValue, item: TItem, order: number): void;
  dispose(value: TValue): void;
}

/**
 * Reconcile ordered layer inputs without disturbing resources whose source
 * identity has not changed.
 */
export function reconcileKeyedLayers<TItem, TValue>(
  entries: Map<string, ManagedLayerEntry<TValue>>,
  items: readonly TItem[],
  reconciler: KeyedLayerReconciler<TItem, TValue>,
): void {
  const desiredKeys = new Set(items.map((item) => reconciler.key(item)));
  for (const [key, entry] of entries) {
    if (desiredKeys.has(key)) continue;
    entries.delete(key);
    reconciler.dispose(entry.value);
  }

  items.forEach((item, order) => {
    const key = reconciler.key(item);
    const identity = reconciler.identity(item);
    let entry = entries.get(key);
    if (entry?.identity !== identity) {
      if (entry) {
        entries.delete(key);
        reconciler.dispose(entry.value);
      }
      entry = { identity, value: reconciler.create(item) };
      entries.set(key, entry);
    }
    reconciler.update(entry.value, item, order);
  });

  // Map iteration order is also the logical painter/pick order. Reinsert the
  // retained entries without recreating their underlying layer resources.
  const orderedEntries = items.flatMap((item) => {
    const key = reconciler.key(item);
    const entry = entries.get(key);
    return entry ? [[key, entry] as const] : [];
  });
  entries.clear();
  for (const [key, entry] of orderedEntries) entries.set(key, entry);
}
