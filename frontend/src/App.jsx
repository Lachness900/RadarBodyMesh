import { useMemo } from "react";
import { ConfidenceBars } from "./components/ConfidenceBars";
import { PredictionPanel } from "./components/PredictionPanel";
import { RadarPointCloud } from "./components/RadarPointCloud";
import { StatusStrip } from "./components/StatusStrip";
import { usePredictionStream } from "./hooks/usePredictionStream";

export default function App() {
  const { message, status } = usePredictionStream();
  const updatedAt = useMemo(() => {
    const timestamp = message?.timestamp_ms || 0;
    return `${(timestamp / 1000).toFixed(1)}s`;
  }, [message]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">mmWave Pose Matching</div>
          <h1>mmYoga</h1>
        </div>
        <div className="timestamp">{updatedAt}</div>
      </header>

      <div className="dashboard-grid">
        <PredictionPanel prediction={message.prediction} />
        <ConfidenceBars probabilities={message.prediction?.probabilities} />
        <RadarPointCloud points={message.points} />
      </div>

      <StatusStrip message={message} status={status} />
    </main>
  );
}
