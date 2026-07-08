import { useEffect, useState } from "react";
import { buildPredictionWsUrl } from "../constants";
import { makeMockMessage } from "../mockData";

/**
 * Owns the WebSocket lifecycle for the dashboard.
 * Backend data wins when ws://localhost:8000 is available; otherwise the hook
 * switches to the local mock stream so the UI remains usable.
 */
export function usePredictionStream(sourceSelection) {
  const [message, setMessage] = useState(() => makeMockMessage(0));
  const [status, setStatus] = useState("mock");

  useEffect(() => {
    let websocket;
    let mockTimer;
    let tick = 1;
    let isActive = true;

    const startMock = () => {
      if (!isActive || mockTimer) return;
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
        setStatus("connected");
        if (mockTimer) {
          window.clearInterval(mockTimer);
          mockTimer = undefined;
        }
      };
      websocket.onmessage = (event) => {
        if (!isActive) return;
        setMessage(JSON.parse(event.data));
      };
      websocket.onerror = () => {
        websocket.close();
      };
      websocket.onclose = () => {
        if (!isActive) return;
        setStatus("disconnected");
        startMock();
      };
    } catch {
      startMock();
    }

    return () => {
      isActive = false;
      if (websocket) websocket.close();
      if (mockTimer) window.clearInterval(mockTimer);
    };
  }, [sourceSelection.replayFile, sourceSelection.source]);

  return { message, status };
}
