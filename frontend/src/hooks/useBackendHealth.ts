import { useCallback, useEffect, useState } from "react";

import { getBackendHealth } from "../services/api";
import type { BackendConnectionState } from "../types/health";

export function useBackendHealth() {
  const [state, setState] = useState<BackendConnectionState>("checking");
  const [message, setMessage] = useState("Checking backend connection…");
  const [attempt, setAttempt] = useState(0);

  const retry = useCallback(() => {
    setState("checking");
    setMessage("Checking backend connection…");
    setAttempt((current) => current + 1);
  }, []);

  useEffect(() => {
    let isCurrent = true;

    getBackendHealth()
      .then(() => {
        if (isCurrent) {
          setState("connected");
          setMessage("Backend is responding.");
        }
      })
      .catch((error: Error) => {
        if (isCurrent) {
          setState("unavailable");
          setMessage(error.message);
        }
      });

    return () => {
      isCurrent = false;
    };
  }, [attempt]);

  return { state, message, retry };
}
