import type {
  RasterLayerManifest,
  RasterPresentation,
} from "../api/types";

interface RasterLayerPanelProps {
  layer: RasterLayerManifest;
  presentation: RasterPresentation;
  onChange(presentation: RasterPresentation): void;
  onRemove(): void;
}

export function RasterLayerPanel({
  layer,
  presentation,
  onChange,
  onRemove,
}: RasterLayerPanelProps) {
  return (
    <details className="layer-panel" open>
      <summary>
        <input
          type="checkbox"
          checked={presentation.visible}
          aria-label={`Show ${layer.name}`}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) =>
            onChange({ ...presentation, visible: event.target.checked })
          }
        />
        <span title="Raster overlay">{layer.name}</span>
        <button
          className="icon-button"
          type="button"
          aria-label={`Remove ${layer.name}`}
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
    </details>
  );
}
