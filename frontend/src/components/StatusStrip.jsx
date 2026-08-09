function fileLabel(path) {
  if (!path) return "none";
  return path.split("/").pop() || path;
}

export function StatusStrip({ message, pointCount, selection, status }) {
  const source = selection?.source || message?.source || "mock";
  const hasCurrentData = message?.source === source && ["connected", "mock"].includes(status);
  const metrics = hasCurrentData ? message?.metrics || {} : {};
  const fps = Number(metrics.fps) || 0;
  const inferenceLatency = Number(metrics.latency_ms) || 0;
  const fpsLabel = source === "mock" ? "mock fps" : "radar fps";
  const connectionLabel =
    {
      mock: "local mock",
      connecting: "connecting",
      waiting: "waiting for data",
      connected: "connected",
      unavailable: "unavailable",
      disconnected: "disconnected",
    }[status] || status;
  const items = [
    connectionLabel,
    source,
    ...(source === "replay" && selection?.source === "replay"
      ? [fileLabel(selection?.replayFile)]
      : []),
    ...(hasCurrentData ? [`${fps.toFixed(1)} ${fpsLabel}`] : []),
    ...(hasCurrentData && source !== "mock"
      ? [`${inferenceLatency.toFixed(1)} ms inference`]
      : []),
    ...(hasCurrentData ? [`${pointCount} displayed points`] : []),
  ];

  return (
    <section className="status-strip">
      <span className={`status-dot ${status}`} />
      {items.map((item) => (
        <span className="status-item" key={item}>
          {item}
        </span>
      ))}
    </section>
  );
}
