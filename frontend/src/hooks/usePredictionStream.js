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
    let tick = 1;
    let isActive = true;
    let terminalError = "";
    const selectedSource = sourceSelection.source || "mock";
    setError("");

    if (selectedSource !== "mock") {
      setMessage(makeWaitingMessage(selectedSource));
      setStatus("connecting");
    }

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
          setMessage(makeWaitingMessage(selectedSource));
          setStatus(selectedSource === "live" ? "unavailable" : "disconnected");
          setError(terminalError);
          websocket.close();
          return;
        }
        setMessage(nextMessage);
        setStatus("connected");
        setError("");
      };
      websocket.onerror = () => {
        terminalError ||= "Could not connect to the backend data stream.";
        websocket.close();
      };
      websocket.onclose = (event) => {
        if (!isActive) return;
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
      if (websocket) websocket.close();
      if (mockTimer) window.clearInterval(mockTimer);
    };
  }, [sourceSelection.replayFile, sourceSelection.source]);

  return { error, message, status };
}
