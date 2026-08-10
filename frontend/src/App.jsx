import { useCallback, useEffect, useMemo, useState } from "react";
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
    default_model: "",
    models: [],
    replay_files: [],
    loaded: false,
  });
  const [sourceSelection, setSourceSelection] = useState({
    source: "mock",
    replayFile: "",
    model: "",
    replayModel: "",
    liveModel: "",
  });
  const syncLiveModel = useCallback((model) => {
    if (!model) return;
    setSourceSelection((current) => {
      if (
        current.liveModel === model &&
        (current.source !== "live" || current.model === model)
      ) {
        return current;
      }
      return {
        ...current,
        liveModel: model,
        model: current.source === "live" ? model : current.model,
      };
    });
  }, []);
  const { error: streamError, message, status } =
    usePredictionStream(sourceSelection);
  const [pointMode, setPointMode] = useState("projected_radar");
  const updatedAt = useMemo(() => {
    const timestamp = message?.timestamp_ms || 0;
    return `${(timestamp / 1000).toFixed(1)}s`;
  }, [message]);
  const timestampLabel =
    message?.source === "replay"
      ? "Replay time"
      : message?.source === "live"
        ? "Live time"
        : "Mock time";
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
        const model = data.default_model || data.models?.[0]?.id || "";
        const activeLiveModel =
          data.sources?.find((source) => source.id === "live")?.status?.model_id ||
          model;
        setSourceOptions({ ...data, loaded: true });
        setSourceSelection((current) => {
          const isInitialSelection = current.source === "mock" && !current.replayFile;
          const source = isInitialSelection
            ? data.default_source || "mock"
            : current.source;
          const replayModel = current.replayModel || model;
          const liveModel = current.liveModel || activeLiveModel;
          return {
            ...current,
            source,
            replayFile: isInitialSelection ? replayFile : current.replayFile,
            replayModel,
            liveModel,
            model:
              source === "live"
                ? liveModel
                : source === "replay"
                  ? replayModel
                  : current.model || model,
          };
        });
      })
      .catch(() => {
        if (!isActive) return;
        setSourceOptions((current) => ({
          ...current,
          loaded: true,
          error: "Could not load input-source status from the backend.",
        }));
        setSourceSelection((current) => ({
          ...current,
          source: "mock",
          replayFile: "",
        }));
      });
    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    if (sourceSelection.source !== "live") return undefined;
    let isActive = true;

    const refreshLiveModel = () => {
      fetch(`${API_URL}/api/sources`, { cache: "no-store" })
        .then((response) => {
          if (!response.ok) throw new Error("Could not refresh Live Radar status.");
          return response.json();
        })
        .then((data) => {
          if (!isActive) return;
          setSourceOptions((current) => ({
            ...current,
            ...data,
            loaded: true,
          }));
          const model = data.sources?.find(
            (source) => source.id === "live",
          )?.status?.model_id;
          syncLiveModel(model);
        })
        .catch(() => {
          // The WebSocket owns connection errors; this lightweight poll only
          // keeps the shared model selector synchronized between dashboards.
        });
    };

    refreshLiveModel();
    const timer = window.setInterval(refreshLiveModel, 1500);
    return () => {
      isActive = false;
      window.clearInterval(timer);
    };
  }, [sourceSelection.source, syncLiveModel]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">mmWave Pose Matching</div>
          <h1>mmPose</h1>
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
        streamError={streamError}
        streamStatus={status}
        onChange={setSourceSelection}
        onOptionsChange={setSourceOptions}
      />

      <StatusStrip
        message={message}
        models={sourceOptions.models}
        pointCount={pointView.points.length}
        selection={sourceSelection}
        status={status}
      />
    </main>
  );
}
