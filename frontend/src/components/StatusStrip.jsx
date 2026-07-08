function fileLabel(path) {
  if (!path) return "none";
  return path.split("/").pop() || path;
}

export function StatusStrip({ message, pointCount, selection, status }) {
  const source = message?.source || "mock";
  const metrics = message?.metrics || {};
  const fpsLabel = source === "mock" ? "mock fps" : "replay fps";
  const connectionLabel =
    status === "mock" ? (source === "mock" ? "local mock" : "connected") : status;
  const items = [
    connectionLabel,
    source,
    ...(selection?.source === "replay" ? [fileLabel(selection?.replayFile)] : []),
    `${(metrics.fps || 0).toFixed(1)} ${fpsLabel}`,
    `${pointCount} points`,
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
