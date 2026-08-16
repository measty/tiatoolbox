import type { StoreManifest } from "../api/types";
import type { LayerPresentation } from "../domain/style-spec";
import {
  DIRECT_COLOR_OPTION,
  aggregateOverviewMaxZoom,
  annotationRepresentationSummary,
  presentationForProperty,
  supportsDirectColor,
} from "../domain/style-spec";

interface LayerPanelProps {
  store: StoreManifest;
  presentation: LayerPresentation;
  onChange(presentation: LayerPresentation): void;
  onRemove(): void;
}

export function LayerPanel({
  store,
  presentation,
  onChange,
  onRemove,
}: LayerPanelProps) {
  const selectedProperty = presentation.colorBy.mode === "constant"
    ? "__constant__"
    : presentation.colorBy.mode === "direct"
      ? DIRECT_COLOR_OPTION
      : presentation.colorBy.property;
  const aggregateMaxZoom = aggregateOverviewMaxZoom(store);
  const representationSummary = annotationRepresentationSummary(store);
  return (
    <details className="layer-panel" open>
      <summary>
        <input
          type="checkbox"
          checked={presentation.visible}
          aria-label={`Show ${store.name}`}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) =>
            onChange({ ...presentation, visible: event.target.checked })
          }
        />
        <span title={`${store.count.toLocaleString()} annotations`}>{store.name}</span>
        <button
          className="icon-button"
          type="button"
          aria-label={`Remove ${store.name}`}
          onClick={(event) => {
            event.preventDefault();
            onRemove();
          }}
        >
          ×
        </button>
      </summary>

      <label>
        Opacity
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={presentation.opacity}
          onChange={(event) =>
            onChange({ ...presentation, opacity: Number(event.target.value) })
          }
        />
        <output>{presentation.opacity.toFixed(2)}</output>
      </label>

      {aggregateMaxZoom !== undefined && (
        <label
          title={
            representationSummary
              ? `${representationSummary} Hiding low zoom suppresses the aggregate range.`
              : `Aggregate dots cover zoom levels 0-${aggregateMaxZoom}. ` +
                `When hidden, annotations begin at zoom ${aggregateMaxZoom + 1}.`
          }
        >
          Low zoom
          <select
            value={presentation.overviewMode}
            onChange={(event) =>
              onChange({
                ...presentation,
                overviewMode:
                  event.target.value === "hidden" ? "hidden" : "aggregate",
              })
            }
          >
            <option value="aggregate">Aggregate dots</option>
            <option value="hidden">Hide annotations</option>
          </select>
        </label>
      )}

      <label>
        Colour by
        <select
          value={selectedProperty}
          onChange={(event) =>
            onChange(
              presentationForProperty(store, presentation, event.target.value),
            )
          }
        >
          <option value="__constant__">Constant</option>
          {supportsDirectColor(store) && (
            <option value={DIRECT_COLOR_OPTION}>Feature colour (color)</option>
          )}
          {store.properties
            .filter((property) =>
              ["categorical", "numeric"].includes(property.kind),
            )
            .map((property) => (
              <option key={property.name} value={property.name}>
                {property.name}
              </option>
            ))}
        </select>
      </label>

      {presentation.colorBy.mode === "constant" && (
        <label>
          Fill colour
          <input
            type="color"
            value={presentation.colorBy.color}
            onChange={(event) =>
              onChange({
                ...presentation,
                colorBy: { mode: "constant", color: event.target.value },
              })
            }
          />
        </label>
      )}

      {presentation.colorBy.mode === "categorical" && (
        <fieldset className="category-list">
          <legend>Categories</legend>
          {presentation.colorBy.categories.map((category, index) => (
            <div className="category-row" key={String(category.value)}>
              <input
                type="checkbox"
                checked={category.visible}
                aria-label={`Show ${String(category.value)}`}
                onChange={(event) => {
                  if (presentation.colorBy.mode !== "categorical") return;
                  const categories = [...presentation.colorBy.categories];
                  const current = categories[index];
                  if (!current) return;
                  categories[index] = {
                    ...current,
                    visible: event.target.checked,
                  };
                  onChange({
                    ...presentation,
                    colorBy: { ...presentation.colorBy, categories },
                  });
                }}
              />
              <span title={String(category.value)}>{String(category.value)}</span>
              <input
                type="color"
                value={category.color}
                aria-label={`Colour for ${String(category.value)}`}
                onChange={(event) => {
                  if (presentation.colorBy.mode !== "categorical") return;
                  const categories = [...presentation.colorBy.categories];
                  const current = categories[index];
                  if (!current) return;
                  categories[index] = { ...current, color: event.target.value };
                  onChange({
                    ...presentation,
                    colorBy: { ...presentation.colorBy, categories },
                  });
                }}
              />
            </div>
          ))}
        </fieldset>
      )}

      {presentation.colorBy.mode === "numeric" && (
        <div className="numeric-style">
          <div className="colour-pair">
            {presentation.colorBy.numeric.colors.map((color, index) => (
              <input
                key={index}
                type="color"
                value={color}
                aria-label={index === 0 ? "Low colour" : "High colour"}
                onChange={(event) => {
                  if (presentation.colorBy.mode !== "numeric") return;
                  const colors: [string, string] = [
                    ...presentation.colorBy.numeric.colors,
                  ];
                  colors[index] = event.target.value;
                  onChange({
                    ...presentation,
                    colorBy: {
                      ...presentation.colorBy,
                      numeric: { ...presentation.colorBy.numeric, colors },
                    },
                  });
                }}
              />
            ))}
          </div>
          {presentation.rangeFilter && (
            <div className="range-filter">
              <label>
                Minimum
                <input
                  type="number"
                  value={presentation.rangeFilter.min}
                  step="any"
                  onChange={(event) =>
                    onChange({
                      ...presentation,
                      rangeFilter: {
                        ...presentation.rangeFilter!,
                        min: Number(event.target.value),
                      },
                    })
                  }
                />
              </label>
              <label>
                Maximum
                <input
                  type="number"
                  value={presentation.rangeFilter.max}
                  step="any"
                  onChange={(event) =>
                    onChange({
                      ...presentation,
                      rangeFilter: {
                        ...presentation.rangeFilter!,
                        max: Number(event.target.value),
                      },
                    })
                  }
                />
              </label>
            </div>
          )}
        </div>
      )}

      <div className="compact-grid">
        <label>
          Fill opacity
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={presentation.fillOpacity}
            onChange={(event) =>
              onChange({
                ...presentation,
                fillOpacity: Number(event.target.value),
              })
            }
          />
        </label>
        <label>
          Outline
          <input
            type="number"
            min="0"
            max="8"
            step="0.25"
            value={presentation.strokeWidth}
            onChange={(event) =>
              onChange({
                ...presentation,
                strokeWidth: Number(event.target.value),
              })
            }
          />
        </label>
      </div>
    </details>
  );
}
