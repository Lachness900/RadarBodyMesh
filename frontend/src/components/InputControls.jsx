import { API_URL } from "../constants";

export function InputControls({
  onChange,
  onOptionsChange,
  options,
  selection,
  streamError,
  streamStatus,
}) {
  const replayFiles = options.replay_files || [];
  const models = options.models || [];
  const hasReplayFiles = replayFiles.length > 0;
  const hasModels = models.length > 0;
  const liveOption = options.sources?.find((option) => option.id === "live");
  const liveStatusMessage = (() => {
    if (selection.source !== "live") return "";
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

  const setInput = (source) => {
    onChange((current) => ({
      ...current,
      source,
      replayFile: current.replayFile || replayFiles[0]?.path || "",
      model: current.model || options.default_model || models[0]?.id || "",
    }));
  };

  const setReplayFile = (replayFile) => {
    onChange((current) => ({ ...current, source: "replay", replayFile }));
  };

  const setModel = (model) => {
    onChange((current) => ({ ...current, model }));
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
          onClick={() => setInput("mock")}
          type="button"
        >
          Mock
        </button>
        <button
          className={selection.source === "replay" ? "active" : ""}
          onClick={() => setInput("replay")}
          type="button"
        >
          Replay
        </button>
        <button
          className={selection.source === "live" ? "active" : ""}
          onClick={() => setInput("live")}
          title="Receive the UDP stream from the ROS2 radar bridge"
          type="button"
        >
          Live Radar
        </button>
      </div>

      {selection.source !== "mock" && (
        <label className="model-control">
          <span>Model</span>
          <select
            disabled={!hasModels}
            value={selection.model}
            onChange={(event) => setModel(event.target.value)}
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
