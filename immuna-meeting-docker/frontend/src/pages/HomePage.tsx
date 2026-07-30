import { startTransition, useEffect, useRef, useState } from "react";
import EditorWorkspace from "../components/home/EditorWorkspace";
import OutputSection from "../components/home/OutputSection";
import BackendStatus from "../components/home/BackendStatus";
import FeaturesGrid from "../components/home/FeaturesGrid";
import {
  createRequestId,
  getProcessProgress,
  getProcessResult,
  startProcessImage,
} from "../lib/api";
import type {
  GeneratedImage,
  ProcessProgressResponse,
  ProcessRequest,
  ProcessResponse,
} from "../types/api";

export default function HomePage() {
  const [outputs, setOutputs] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProcessProgressResponse | null>(null);
  const [metadata, setMetadata] = useState<
    Pick<
      ProcessResponse,
      | "device"
      | "modelSource"
      | "immunizationProfile"
      | "workingResolution"
      | "outputFormat"
      | "losslessOutput"
    > | null
  >(null);
  const pollTimeoutRef = useRef<number | null>(null);
  const activeRequestIdRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (pollTimeoutRef.current !== null) {
        window.clearTimeout(pollTimeoutRef.current);
      }
    };
  }, []);

  const stopPolling = () => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
    activeRequestIdRef.current = null;
  };

  const applyCompletedResponse = (response: ProcessResponse) => {
    startTransition(() => {
      setOutputs(response.outputs);
      setMetadata({
        device: response.device,
        modelSource: response.modelSource,
        immunizationProfile: response.immunizationProfile,
        workingResolution: response.workingResolution,
        outputFormat: response.outputFormat,
        losslessOutput: response.losslessOutput,
      });
      setError(null);
      setStatusText(response.statusText);
      setProgress((current) => ({
        requestId: response.requestId,
        status: "completed",
        stage: "completed",
        percent: 100,
        message: "PhotoGuard request completed successfully.",
        iteration: current?.iteration ?? null,
        totalIterations: current?.totalIterations ?? null,
        metrics: current?.metrics ?? {},
        statusText: response.statusText,
      }));
    });
  };

  const pollProgress = async (requestId: string) => {
    if (activeRequestIdRef.current !== requestId) {
      return;
    }

    try {
      const nextProgress = await getProcessProgress(requestId);
      if (activeRequestIdRef.current !== requestId) {
        return;
      }
      if (nextProgress) {
        setProgress(nextProgress);
        setStatusText(nextProgress.statusText);
        if (nextProgress.status === "completed") {
          const completedResponse = await getProcessResult(requestId).catch(() => null);
          if (activeRequestIdRef.current !== requestId) {
            return;
          }
          if (completedResponse) {
            stopPolling();
            applyCompletedResponse(completedResponse);
            setIsLoading(false);
            return;
          }
        } else if (nextProgress.status === "failed") {
          stopPolling();
          setOutputs([]);
          setMetadata(null);
          setError(nextProgress.statusText || nextProgress.message || "Unable to process the image.");
          setStatusText(nextProgress.statusText);
          setIsLoading(false);
          pollTimeoutRef.current = null;
          return;
        }
      }
    } catch {
      // Keep polling while the request is active.
    }

    if (activeRequestIdRef.current === requestId) {
      pollTimeoutRef.current = window.setTimeout(() => {
        void pollProgress(requestId);
      }, 700);
    }
  };

  const handleProcess = async (request: ProcessRequest) => {
    const requestId = createRequestId();
    setIsLoading(true);
    setError(null);
    setStatusText(null);
    stopPolling();
    activeRequestIdRef.current = requestId;
    setProgress({
      requestId,
      status: "running",
      stage: "starting",
      percent: 0,
      message: "Submitting request to the API.",
      iteration: null,
      totalIterations: null,
      metrics: {},
      statusText: "[PROCESSING] Request submitted. Waiting for the background job to start...",
    });
    void pollProgress(requestId);

    try {
      const response = await startProcessImage({ ...request, requestId });
      setStatusText(response.statusText);
      setProgress((current) =>
        current && current.requestId === requestId
          ? {
              ...current,
              message: "Background job accepted. Polling for progress updates.",
              statusText: response.statusText,
            }
          : current,
      );
    } catch (caughtError) {
      const latestProgress = await getProcessProgress(requestId).catch(() => null);
      stopPolling();
      setOutputs([]);
      setMetadata(null);
      const message =
        caughtError instanceof Error ? caughtError.message : "Unable to process the image.";
      setError(message);
      setStatusText(latestProgress?.statusText ?? message);
      setProgress(
        latestProgress ?? {
          requestId,
          status: "failed",
          stage: "failed",
          percent: 100,
          message,
          iteration: null,
          totalIterations: null,
          metrics: {},
          statusText: message,
        },
      );
    } finally {
      if (activeRequestIdRef.current !== requestId) {
        setIsLoading(false);
      }
    }
  };

  return (
    <main className="w-full">
      <section className="px-6 pt-20 pb-12 text-center bg-transparent">
        <h1 className="text-3xl md:text-4xl font-semibold mb-3" style={{ color: "var(--foreground)" }}>
          Protecting Image Authenticity Against AI Manipulation
        </h1>
        <p className="max-w-2xl mx-auto text-sm md:text-base leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
          A privacy-first cybersecurity research platform that embeds cryptographic watermarks into
          images. Detect tampering, verify authenticity, and establish provenance—without storing your data.
        </p>
      </section>

      <div className="max-w-7xl mx-auto px-4 md:px-8 py-8 space-y-6">
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] gap-6" style={{ alignItems: "start" }}>
          <EditorWorkspace onProcess={handleProcess} isLoading={isLoading} error={error} />
          <OutputSection
            outputs={outputs}
            isLoading={isLoading}
            error={error}
            progress={progress}
            statusText={statusText}
            device={metadata?.device ?? null}
            modelSource={metadata?.modelSource ?? null}
            immunizationProfile={metadata?.immunizationProfile ?? null}
            workingResolution={metadata?.workingResolution ?? null}
            outputFormat={metadata?.outputFormat ?? null}
            losslessOutput={metadata?.losslessOutput ?? null}
          />
        </div>
        <BackendStatus suspend={isLoading} />
        <FeaturesGrid />
      </div>
    </main>
  );
}
