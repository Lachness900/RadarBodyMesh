import { POSES } from "../constants";

export function ConfidenceBars({ probabilities }) {
  return (
    <section className="panel confidence-panel">
      <div className="panel-label">Confidence</div>
      <div className="bars">
        {POSES.map((pose) => {
          const value = probabilities?.[pose.key] || 0;
          return (
            <div className="bar-row" key={pose.key}>
              <div className="bar-heading">
                <span>{pose.label}</span>
                <span>{Math.round(value * 100)}%</span>
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{ width: `${value * 100}%`, background: pose.color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
