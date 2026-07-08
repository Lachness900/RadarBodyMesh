import { POSES } from "../constants";

function getPoseMeta(label) {
  return POSES.find((pose) => pose.key === label) || {
    color: "#0f8b8d",
    key: label,
    label,
  };
}

export function PredictionPanel({ prediction }) {
  const confidence = Math.round((prediction?.confidence || 0) * 100);
  const pose = getPoseMeta(prediction?.label || "unknown");
  const poseLabel = pose.label;
  const isLongPose = poseLabel.length > 9;

  return (
    <section className="panel prediction-panel">
      <div className="panel-label">Current Pose</div>
      <div className="prediction-content">
        <div className={isLongPose ? "pose-name long" : "pose-name"} title={poseLabel}>
          {poseLabel}
        </div>
        <div
          className="confidence-ring"
          style={{ "--pose-color": pose.color, "--value": `${confidence}%` }}
        >
          <span>{confidence}%</span>
        </div>
      </div>
    </section>
  );
}
