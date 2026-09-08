import { formatModelName, getAvatarColor, seatLetter } from '../utils/councilUtils';
import './DrawerRoster.css';

export default function DrawerRoster({ councilModels = [], chairmanModel = '' }) {
  return (
    <div className="drawer-roster">
      <div className="drawer-roster-head">
        <h4 className="drawer-roster-title">Council</h4>
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
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
