import { API_URL } from "../constants";

export function InputControls({ onChange, onOptionsChange, options, selection }) {
  const replayFiles = options.replay_files || [];
  const hasReplayFiles = replayFiles.length > 0;

  const setInput = (source) => {
    if (source === "live") return;
    onChange((current) => ({
      source,
      replayFile: source === "replay" ? current.replayFile || replayFiles[0]?.path || "" : "",
    }));
  };

  const setReplayFile = (replayFile) => {
    onChange((current) => ({ ...current, source: "replay", replayFile }));
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
    onChange({ source: "replay", replayFile: uploaded.path });
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
        <button disabled title="Live radar is not connected yet" type="button">
          Live soon
        </button>
      </div>

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
