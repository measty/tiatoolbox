import type { FeatureDetail } from "../api/types";

interface InspectorProps {
  detail: FeatureDetail | null;
  loading: boolean;
  error: string | null;
  onClose(): void;
}

export function Inspector({ detail, loading, error, onClose }: InspectorProps) {
  if (!detail && !loading && !error) return null;
  return (
    <aside className="inspector" aria-label="Annotation properties">
      <header>
        <strong>Annotation</strong>
        <button className="icon-button" type="button" onClick={onClose}>
          ×
        </button>
      </header>
      {loading && <p>Loading exact annotation…</p>}
      {error && <p className="error-message">{error}</p>}
      {detail && (
        <dl>
          <div>
            <dt>Feature ID</dt>
            <dd>{String(detail.fid)}</dd>
          </div>
          {detail.uuid && (
            <div>
              <dt>UUID</dt>
              <dd>{detail.uuid}</dd>
            </div>
          )}
          {Object.entries(detail.properties).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </aside>
  );
}

function formatValue(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
