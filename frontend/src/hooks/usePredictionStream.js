import { useEffect, useState } from "react";
import { buildPredictionWsUrl } from "../constants";
import { makeMockMessage, makeWaitingMessage } from "../mockData";

/**
 * Owns the WebSocket lifecycle for the dashboard.
 * Local mock fallback is allowed only when Mock is selected. Replay and Live
 * keep their own waiting/error states so one source cannot impersonate another.
 */
export function usePredictionStream(sourceSelection) {
  const [message, setMessage] = useState(() => makeMockMessage(0));
  const [status, setStatus] = useState("mock");
  const [error, setError] = useState("");

  useEffect(() => {
    let websocket;
    let mockTimer;
    let flushTimer;
    let tick = 1;
    let isActive = true;
    let terminalError = "";
    let latestMessage = null;
    const selectedSource = sourceSelection.source || "mock";
    const selectedModel = sourceSelection.model || "";
    setError("");

    if (selectedSource !== "mock") {
      setMessage(makeWaitingMessage(selectedSource, selectedModel));
      setStatus("connecting");
    }

    // Coalesce high-rate WebSocket frames so React re-renders at most ~30 fps,
    // mirroring the offline visualizer's capped render rate. The latest frame
    // is always kept, so the dashboard never shows stale data.
    const FLUSH_INTERVAL_MS = 33;
    const scheduleFlush = () => {
      if (flushTimer) return;
      flushTimer = window.setInterval(() => {
        if (!isActive) return;
        if (latestMessage) {
          setMessage(latestMessage);
          latestMessage = null;
        } else {
          window.clearInterval(flushTimer);
          flushTimer = undefined;
        }
      }, FLUSH_INTERVAL_MS);
    };
    const flushNow = () => {
      if (latestMessage) {
        setMessage(latestMessage);
        latestMessage = null;
      }
      if (flushTimer) {
        window.clearInterval(flushTimer);
        flushTimer = undefined;
      }
    };

    const startMock = () => {
      if (!isActive || mockTimer || selectedSource !== "mock") return;
      setStatus("mock");
      mockTimer = window.setInterval(() => {
        setMessage(makeMockMessage(tick));
        tick += 1;
      }, 100);
    };

    try {
      websocket = new WebSocket(buildPredictionWsUrl(sourceSelection));
      websocket.onopen = () => {
        if (!isActive) return;
        setError("");
        setStatus(selectedSource === "mock" ? "connected" : "waiting");
        if (mockTimer) {
          window.clearInterval(mockTimer);
          mockTimer = undefined;
        }
      };
      websocket.onmessage = (event) => {
        if (!isActive) return;
        let nextMessage;
        try {
          nextMessage = JSON.parse(event.data);
        } catch {
          terminalError = "The backend returned an invalid data message.";
          setStatus(selectedSource === "live" ? "unavailable" : "disconnected");
          setError(terminalError);
          websocket.close();
          return;
        }
        if (nextMessage.source !== selectedSource) {
          terminalError = `Expected ${selectedSource} data, but received ${nextMessage.source || "an unknown source"}.`;
          setMessage(makeWaitingMessage(selectedSource, selectedModel));
          setStatus(selectedSource === "live" ? "unavailable" : "disconnected");
          setError(terminalError);
          websocket.close();
          return;
        }
        if (selectedSource !== "mock" && nextMessage.model_id !== selectedModel) {
          terminalError = `Expected model ${selectedModel}, but received ${nextMessage.model_id || "an unknown model"}.`;
          setMessage(makeWaitingMessage(selectedSource, selectedModel));
          setStatus(selectedSource === "live" ? "unavailable" : "disconnected");
          setError(terminalError);
          websocket.close();
          return;
        }
        latestMessage = nextMessage;
        scheduleFlush();
        setStatus("connected");
        setError("");
      };
      websocket.onerror = () => {
        terminalError ||= "Could not connect to the backend data stream.";
        websocket.close();
      };
      websocket.onclose = (event) => {
        if (!isActive) return;
        flushNow();
        if (selectedSource === "mock") {
          startMock();
        } else {
          setStatus(selectedSource === "live" ? "unavailable" : "disconnected");
          setError(terminalError || event.reason || "The backend closed the data stream.");
        }
      };
    } catch (streamError) {
      if (selectedSource === "mock") {
        startMock();
      } else {
        setStatus(selectedSource === "live" ? "unavailable" : "disconnected");
        setError(streamError instanceof Error ? streamError.message : "Could not connect.");
      }
    }

    return () => {
      isActive = false;
      if (flushTimer) window.clearInterval(flushTimer);
      if (websocket) websocket.close();
      if (mockTimer) window.clearInterval(mockTimer);
    };
  }, [sourceSelection.model, sourceSelection.replayFile, sourceSelection.source]);

  return { error, message, status };
}
