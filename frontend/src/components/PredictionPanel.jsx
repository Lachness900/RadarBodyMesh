import { POSES } from "../constants";

function formatPose(label) {
  return POSES.find((pose) => pose.key === label)?.label || label;
}

export function PredictionPanel({ prediction }) {
  const confidence = Math.round((prediction?.confidence || 0) * 100);
  return (
    <section className="panel prediction-panel">
      <div className="panel-label">Current Pose</div>
      <div className="pose-name">{formatPose(prediction?.label || "unknown")}</div>
      <div className="confidence-ring" style={{ "--value": `${confidence}%` }}>
        <span>{confidence}%</span>
      </div>
    </section>
  );
}
