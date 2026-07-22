import { useEffect, useMemo, useState } from "react";
import { ConfidenceBars } from "./components/ConfidenceBars";
import { InputControls } from "./components/InputControls";
import { PredictionPanel } from "./components/PredictionPanel";
import { RadarPointCloud } from "./components/RadarPointCloud";
import { StatusStrip } from "./components/StatusStrip";
import { API_URL } from "./constants";
import { usePredictionStream } from "./hooks/usePredictionStream";

export default function App() {
  const [sourceOptions, setSourceOptions] = useState({
    default_source: "mock",
    replay_files: [],
  });
  const [sourceSelection, setSourceSelection] = useState({
    source: "mock",
    replayFile: "",
  });
  const { message, status } = usePredictionStream(sourceSelection);
  const [pointMode, setPointMode] = useState("projected_radar");
  const updatedAt = useMemo(() => {
    const timestamp = message?.timestamp_ms || 0;
    return `${(timestamp / 1000).toFixed(1)}s`;
  }, [message]);
  const timestampLabel = message?.source === "replay" ? "Replay time" : "Mock time";
  const pointView = useMemo(() => {
    const pointSets = message.point_sets || {};
    const selected = pointSets[pointMode] || message.points || [];
    if (selected.length > 0) {
      return { mode: pointMode, points: selected };
    }
    return {
      mode: pointSets.raw_radar?.length ? "raw_radar" : pointMode,
      points: pointSets.raw_radar || message.points || [],
    };
  }, [message, pointMode]);

  useEffect(() => {
    let isActive = true;
    fetch(`${API_URL}/api/sources`)
      .then((response) => response.json())
      .then((data) => {
        if (!isActive) return;
        const replayFile =
          data.replay_files?.find((file) => file.selected)?.path ||
          data.replay_files?.[0]?.path ||
          "";
        setSourceOptions(data);
        setSourceSelection((current) => {
          if (current.source !== "mock" || current.replayFile) return current;
          return {
            source: data.default_source || "mock",
            replayFile,
          };
        });
      })
      .catch(() => {
        if (!isActive) return;
        setSourceSelection({ source: "mock", replayFile: "" });
      });
    return () => {
      isActive = false;
    };
  }, []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">mmWave Pose Matching</div>
          <h1>mmYoga</h1>
        </div>
        <div className="timestamp">
          <span>{timestampLabel}</span>
          <strong>{updatedAt}</strong>
        </div>
      </header>

      <div className="dashboard-grid">
        <PredictionPanel prediction={message.prediction} />
        <ConfidenceBars probabilities={message.prediction?.probabilities} />
        <RadarPointCloud
          mode={pointView.mode}
          onModeChange={setPointMode}
          points={pointView.points}
          pointSets={message.point_sets}
          viewKey={`${sourceSelection.source}:${sourceSelection.replayFile}:${pointView.mode}`}
        />
      </div>

      <InputControls
        options={sourceOptions}
        selection={sourceSelection}
        onChange={setSourceSelection}
        onOptionsChange={setSourceOptions}
      />

      <StatusStrip
        message={message}
        pointCount={pointView.points.length}
        selection={sourceSelection}
        status={status}
      />
    </main>
  );
}
