function fileLabel(path) {
  if (!path) return "none";
  return path.split("/").pop() || path;
}

export function StatusStrip({ message, pointCount, selection, status }) {
  const source = message?.source || "mock";
  const metrics = message?.metrics || {};
  const fps = Number(metrics.fps) || 0;
  const inferenceLatency = Number(metrics.latency_ms) || 0;
  const fpsLabel = source === "mock" ? "mock fps" : "radar fps";
  const connectionLabel =
    status === "mock" ? (source === "mock" ? "local mock" : "connected") : status;
  const items = [
    connectionLabel,
    source,
    ...(source === "replay" && selection?.source === "replay"
      ? [fileLabel(selection?.replayFile)]
      : []),
    `${fps.toFixed(1)} ${fpsLabel}`,
    ...(source === "replay" ? [`${inferenceLatency.toFixed(1)} ms inference`] : []),
    `${pointCount} displayed points`,
  ];

  return (
    <section className="status-strip">
      <span className={`status-dot ${connectionLabel === "connected" ? "connected" : status}`} />
      {items.map((item) => (
        <span className="status-item" key={item}>
          {item}
        </span>
      ))}
    </section>
  );
}
