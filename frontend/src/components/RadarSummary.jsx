function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "-";
}

export function RadarSummary({ points }) {
  const safePoints = points || [];
  const samples = safePoints.slice(0, 4);

  return (
    <section className="panel radar-summary-panel">
      <div className="panel-label">Radar Frame</div>
      <div className="point-count">
        {safePoints.length}
        <span>raw xyz points</span>
      </div>
      <div className="sample-table">
        <div>x</div>
        <div>y</div>
        <div>z</div>
        <div>intensity</div>
        {samples.map((point, index) => (
          <div className="sample-row" key={`${point.x}-${point.y}-${point.z}-${index}`}>
            <span>{formatNumber(point.x)}</span>
            <span>{formatNumber(point.y)}</span>
            <span>{formatNumber(point.z)}</span>
            <span>{formatNumber(point.intensity)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
