export function StatusStrip({ message, status }) {
  const source = message?.source || "mock";
  const metrics = message?.metrics || {};
  return (
    <section className="status-strip">
      <div>
        <span className={`status-dot ${status}`} />
        <span>{status}</span>
      </div>
      <div>source {source}</div>
      <div>fps {(metrics.fps || 0).toFixed(1)}</div>
      <div>latency {(metrics.latency_ms || 0).toFixed(1)} ms</div>
      <div>points {(message?.points || []).length}</div>
    </section>
  );
}
