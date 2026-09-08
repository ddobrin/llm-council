import { formatModelName, getAvatarColor, seatLetter } from '../utils/councilUtils';
import './DrawerRoster.css';

export default function DrawerRoster({
  councilModels = [],
  chairmanModel = '',
  modelEfforts = {},
  onUpdateModelEffort,
  availableEfforts = ['default', 'minimal', 'low', 'medium', 'high'],
}) {
  const effortOptions = availableEfforts.map((eff) => ({
    value: eff,
    label: eff === 'default' ? 'Default' : eff.charAt(0).toUpperCase() + eff.slice(1),
  }));

  return (
    <div className="drawer-roster">
      <div className="drawer-roster-head">
        <h4 className="drawer-roster-title">Council & Effort</h4>
      </div>

      {chairmanModel && (
        <>
          <div className="drawer-roster-label">Chair</div>
          <div className="drawer-roster-row">
            <span className="drawer-roster-glyph is-chair" title="Council Chair">
              CH
            </span>
            <span
              className="drawer-roster-dot"
              style={{ backgroundColor: getAvatarColor(chairmanModel) }}
            />
            <span className="drawer-roster-name" title={chairmanModel}>
              {formatModelName(chairmanModel)}
            </span>
            <select
              className="drawer-roster-effort-select"
              value={modelEfforts[chairmanModel] || 'default'}
              onChange={(e) => onUpdateModelEffort?.(chairmanModel, e.target.value)}
              title={`Reasoning effort for Chair (${formatModelName(chairmanModel)})`}
              aria-label={`Reasoning effort for Chair ${formatModelName(chairmanModel)}`}
            >
              {effortOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </>
      )}

      {councilModels.length > 0 && (
        <>
          <div className="drawer-roster-label">Members</div>
          {councilModels.map((modelId, index) => {
            const letter = seatLetter(index);
            return (
              <div key={modelId || index} className="drawer-roster-row">
                <span
                  className="drawer-roster-glyph is-seat"
                  title={`Seat ${letter}`}
                >
                  {letter}
                </span>
                <span
                  className="drawer-roster-dot"
                  style={{ backgroundColor: getAvatarColor(modelId, index) }}
                />
                <span className="drawer-roster-name" title={modelId}>
                  {formatModelName(modelId)}
                </span>
                <select
                  className="drawer-roster-effort-select"
                  value={modelEfforts[modelId] || 'default'}
                  onChange={(e) => onUpdateModelEffort?.(modelId, e.target.value)}
                  title={`Reasoning effort for Seat ${letter} (${formatModelName(modelId)})`}
                  aria-label={`Reasoning effort for Seat ${letter} ${formatModelName(modelId)}`}
                >
                  {effortOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
