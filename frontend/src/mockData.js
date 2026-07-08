import { POSES } from "./constants";

/**
 * Local fallback message used when the backend WebSocket is unavailable.
 * It mirrors the real backend payload shape so the dashboard can be reviewed
 * before replay/parser work is restored.
 */
export function makeMockMessage(tick) {
  const phase = tick / 16;
  const raw = POSES.map((_, index) =>
    Math.max(0.05, 0.42 + 0.24 * Math.sin(phase + index * 1.7)),
  );
  const total = raw.reduce((sum, value) => sum + value, 0);
  const probabilities = Object.fromEntries(
    POSES.map((pose, index) => [pose.key, raw[index] / total]),
  );
  const best = POSES.reduce((current, pose) =>
    probabilities[pose.key] > probabilities[current.key] ? pose : current,
  );

  const points = Array.from({ length: 90 }, (_, index) => {
    const angle = (index / 90) * Math.PI * 2;
    const radius = 0.28 + 0.08 * Math.sin(angle * 3 + phase);
    return {
      x: 2.8 + radius * Math.cos(angle),
      y: 0.2 + radius * Math.sin(angle),
      z: 0.7 + 0.5 * Math.sin(angle * 2 + phase),
      intensity: 12 + 8 * Math.abs(Math.sin(angle + phase)),
      velocity: 0,
    };
  });
  const visualizerPoints = points.map((point) => ({ ...point, x: 0 }));

  return {
    timestamp_ms: tick * 100,
    source: "mock",
    prediction: {
      label: best.key,
      confidence: probabilities[best.key],
      probabilities,
    },
    points,
    point_sets: {
      projected_radar: visualizerPoints,
      filtered_radar: points,
      raw_radar: points,
    },
    metrics: {
      fps: 10,
      latency_ms: 8 + Math.abs(Math.sin(phase)) * 6,
    },
  };
}
