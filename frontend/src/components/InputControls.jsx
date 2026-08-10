import { useState } from "react";
import { API_URL } from "../constants";

export function InputControls({
  onChange,
  onOptionsChange,
  options,
  selection,
  streamError,
  streamStatus,
}) {
  const [modelUpdating, setModelUpdating] = useState(false);
  const replayFiles = options.replay_files || [];
  const models = options.models || [];
  const hasReplayFiles = replayFiles.length > 0;
  const hasModels = models.length > 0;
  const liveOption = options.sources?.find((option) => option.id === "live");
  const liveStatusMessage = (() => {
    if (selection.source !== "live") return "";
    if (options.modelError) return options.modelError;
    if (!liveOption) {
      if (!options.loaded) return "Checking Live Radar availability...";
      return (
        streamError ||
        options.error ||
        "Live Radar is not advertised by the running backend."
      );
    }
    if (streamStatus === "connecting" || streamStatus === "waiting") {
      return `Waiting for UDP radar frames on ${liveOption.status?.endpoint || "the configured endpoint"}...`;
    }
    if (streamStatus === "unavailable") {
      return streamError || "Live Radar connection failed.";
    }
    return "";
  })();

  const setInput = async (source) => {
    let refreshedOptions = options;
    if (source === "live") {
      try {
        const response = await fetch(`${API_URL}/api/sources`, {
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Could not read Live Radar status.");
        const data = await response.json();
        refreshedOptions = { ...data, loaded: true };
        onOptionsChange(refreshedOptions);
      } catch {
        // The WebSocket still provides the final availability/error state.
      }
    }

    const refreshedModels = refreshedOptions.models || models;
    const defaultModel =
      refreshedOptions.default_model || refreshedModels[0]?.id || "";
    const activeLiveModel =
      refreshedOptions.sources?.find((option) => option.id === "live")?.status
        ?.model_id || "";
    onChange((current) => ({
      ...current,
      source,
      replayFile: current.replayFile || replayFiles[0]?.path || "",
      replayModel: current.replayModel || defaultModel,
      liveModel: activeLiveModel || current.liveModel || defaultModel,
      model:
        source === "live"
          ? activeLiveModel || current.liveModel || defaultModel
          : source === "replay"
            ? current.replayModel || defaultModel
            : current.model || defaultModel,
    }));
  };

  const setReplayFile = (replayFile) => {
    onChange((current) => ({
      ...current,
      source: "replay",
      replayFile,
      model: current.replayModel || options.default_model || models[0]?.id || "",
    }));
  };

  const setModel = async (model) => {
    if (selection.source !== "live") {
      onChange((current) => ({ ...current, model, replayModel: model }));
      return;
    }

    setModelUpdating(true);
    onOptionsChange((current) => ({ ...current, modelError: "" }));
    try {
      const params = new URLSearchParams({ model });
      const response = await fetch(`${API_URL}/api/live-model?${params.toString()}`, {
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail =
          typeof payload.detail === "string"
            ? payload.detail
            : "The backend could not switch the Live Radar model.";
        throw new Error(detail);
      }
      onOptionsChange((current) => ({
        ...current,
        modelError: "",
        sources: (current.sources || []).map((option) =>
          option.id === "live"
            ? { ...option, status: payload.live || option.status }
            : option,
        ),
      }));
      onChange((current) => ({
        ...current,
        model: payload.model_id,
        liveModel: payload.model_id,
      }));
    } catch (error) {
      onOptionsChange((current) => ({
        ...current,
        modelError:
          error instanceof Error
            ? error.message
            : "The backend could not switch the Live Radar model.",
      }));
    } finally {
      setModelUpdating(false);
    }
  };

  const uploadReplayFile = async (file) => {
    if (!file) return;
    const params = new URLSearchParams({ filename: file.name });
    const response = await fetch(`${API_URL}/api/replay-files?${params.toString()}`, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: file,
    });
    if (!response.ok) return;
    const uploaded = await response.json();
    onOptionsChange((current) => {
      const existing = current.replay_files || [];
      const replayFilesByPath = new Map(existing.map((item) => [item.path, item]));
      replayFilesByPath.set(uploaded.path, uploaded);
      return {
        ...current,
        replay_files: Array.from(replayFilesByPath.values()).map((item) => ({
          ...item,
          selected: item.path === uploaded.path,
        })),
      };
    });
    onChange((current) => ({
      ...current,
      source: "replay",
      replayFile: uploaded.path,
    }));
  };

  return (
    <section className="input-controls" aria-label="Input controls">
      <span className="input-label">Input</span>
      <div className="input-mode-group" role="group" aria-label="Input source">
        <button
          className={selection.source === "mock" ? "active" : ""}
          onClick={() => void setInput("mock")}
          type="button"
        >
          Mock
        </button>
        <button
          className={selection.source === "replay" ? "active" : ""}
          onClick={() => void setInput("replay")}
          type="button"
        >
          Replay
        </button>
        <button
          className={selection.source === "live" ? "active" : ""}
          onClick={() => void setInput("live")}
          title="Receive the UDP stream from the ROS2 radar bridge"
          type="button"
        >
          Live Radar
        </button>
      </div>

      {selection.source !== "mock" && (
        <label
          className="model-control"
          title={
            selection.source === "live"
              ? "Shared by all connected Live Radar dashboards"
              : "Used only by this Replay dashboard"
          }
        >
          <span>Model</span>
          <select
            disabled={!hasModels || modelUpdating}
            value={selection.model}
            onChange={(event) => void setModel(event.target.value)}
          >
            {hasModels ? (
              models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))
            ) : (
              <option value="">No model available</option>
            )}
          </select>
        </label>
      )}

      {liveStatusMessage && (
        <span
          className="input-status-message"
          data-state={streamStatus}
          role="status"
        >
          {liveStatusMessage}
        </span>
      )}

      {selection.source === "replay" && (
        <>
          <label className="replay-file-control">
            <span>Replay file</span>
            <select
              disabled={!hasReplayFiles}
              value={selection.replayFile}
              onChange={(event) => setReplayFile(event.target.value)}
            >
              {hasReplayFiles ? (
                replayFiles.map((file) => (
                  <option key={file.path} value={file.path}>
                    {file.label}
                  </option>
                ))
              ) : (
                <option value="">No replay file loaded</option>
              )}
            </select>
          </label>

          <label className="upload-button">
            <span>Open .dat File</span>
            <input
              accept=".dat"
              type="file"
              onChange={(event) => {
                uploadReplayFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </label>
        </>
      )}
    </section>
  );
}
