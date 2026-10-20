import { useCallback, useEffect, useRef, useState } from "react";

import {
  getDatasetAudit,
  getDatasetCases,
  getDatasetScan,
  getPreprocessingStatus,
  scanDataset,
} from "../services/datasetApi";
import type {
  DatasetAudit,
  DatasetCase,
  DatasetScanStatus,
  PreprocessingStatus,
} from "../types/dataset";

export function useDatasetStatus() {
  const [scan, setScan] = useState<DatasetScanStatus | null>(null);
  const [cases, setCases] = useState<DatasetCase[]>([]);
  const [preprocessing, setPreprocessing] =
    useState<PreprocessingStatus | null>(null);
  const [audit, setAudit] = useState<DatasetAudit | null>(null);
  const [auditError, setAuditError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isScanning, setIsScanning] = useState(false);
  const [isLoadingAudit, setIsLoadingAudit] = useState(false);
  const auditWasRequested = useRef(false);
  const auditRequestId = useRef(0);
  const auditRequest = useRef<Promise<DatasetAudit> | null>(null);
  const auditCache = useRef<DatasetAudit | null>(null);

  const loadAudit = useCallback(async (force = false) => {
    auditWasRequested.current = true;
    if (!force && (auditRequest.current || auditCache.current)) return;

    const requestId = ++auditRequestId.current;
    const request = getDatasetAudit();
    auditRequest.current = request;
    setIsLoadingAudit(true);
    setAuditError(null);
    try {
      const result = await request;
      if (requestId === auditRequestId.current) {
        auditCache.current = result;
        setAudit(result);
      }
    } catch (caught) {
      if (requestId === auditRequestId.current) {
        setAuditError(
          caught instanceof Error ? caught.message : "The audit report is unavailable.",
        );
      }
    } finally {
      if (requestId === auditRequestId.current) {
        auditRequest.current = null;
        setIsLoadingAudit(false);
      }
    }
  }, []);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [nextScan, nextCases, nextPreprocessing] =
        await Promise.all([
          getDatasetScan(),
          getDatasetCases(),
          getPreprocessingStatus(),
        ]);
      setScan(nextScan);
      setCases(nextCases.cases);
      setPreprocessing(nextPreprocessing);
      auditCache.current = null;
      setAudit(null);
      setAuditError(null);
      if (auditWasRequested.current) {
        void loadAudit(true);
      }
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Dataset status is unavailable.",
      );
    } finally {
      setIsLoading(false);
    }
  }, [loadAudit]);

  const runScan = useCallback(async () => {
    setIsScanning(true);
    setError(null);
    try {
      await scanDataset();
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Dataset scan failed.");
    } finally {
      setIsScanning(false);
    }
  }, [refresh]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    scan,
    cases,
    preprocessing,
    audit,
    auditError,
    error,
    isLoading,
    isScanning,
    isLoadingAudit,
    refresh,
    runScan,
    loadAudit,
  };
}
