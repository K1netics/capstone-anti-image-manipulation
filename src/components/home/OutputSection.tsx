import { Image as ImageIcon, LoaderCircle } from "lucide-react";
import Card from "../ui/Card";
import type {
  GeneratedImage,
  ImmunizationProfile,
  OutputFormat,
  ProcessProgressResponse,
  WorkingResolution,
} from "../../types/api";

interface OutputSectionProps {
  outputs: GeneratedImage[];
  isLoading: boolean;
  error: string | null;
  progress: ProcessProgressResponse | null;
  statusText: string | null;
  device: string | null;
  modelSource: string | null;
  immunizationProfile: ImmunizationProfile | null;
  workingResolution: WorkingResolution | null;
  outputFormat: OutputFormat | null;
  losslessOutput: boolean | null;
}

function profileLabel(profile: ImmunizationProfile | null) {
  if (profile === "nano_banana_experimental") {
    return "Nano Banana experimental";
  }
  if (profile === "stable_diffusion") {
    return "Stable Diffusion";
  }
  return null;
}

function formatStage(stage: string | null) {
  if (!stage) {
    return null;
  }
  return stage.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

export default function OutputSection({
  outputs,
  isLoading,
  error,
  progress,
  statusText,
  device,
  modelSource,
  immunizationProfile,
  workingResolution,
  outputFormat,
  losslessOutput,
}: OutputSectionProps) {
  const metricEntries = progress
    ? Object.entries(progress.metrics).sort(([left], [right]) => left.localeCompare(right))
    : [];
  const feedbackText = isLoading
    ? progress?.statusText ?? "[PROCESSING] Request submitted. Waiting for the API response..."
    : statusText ?? progress?.statusText ?? "[INFO] Run the pipeline to see the feedback log here.";

  return (
    <Card className="min-h-[720px] gap-5">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-2xl font-semibold" style={{ color: "var(--foreground)" }}>
            Generated outputs
          </h2>
          <p className="text-sm mt-1" style={{ color: "var(--muted-foreground)" }}>
            Results from the separate API-backed workflow in this integrated workspace.
          </p>
        </div>
        {(device || modelSource) && (
          <div
            className="rounded-2xl px-4 py-3 text-xs max-w-full"
            style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
          >
            {device ? <p>Device: {device}</p> : null}
            {modelSource ? <p className="truncate max-w-[20rem]">Model: {modelSource}</p> : null}
            {workingResolution ? <p>Working: {workingResolution}</p> : null}
            {outputFormat ? (
              <p>
                Output: {outputFormat.toUpperCase()}
                {outputFormat === "webp" ? ` (${losslessOutput ? "lossless" : "lossy"})` : ""}
              </p>
            ) : null}
            {profileLabel(immunizationProfile) ? <p>Profile: {profileLabel(immunizationProfile)}</p> : null}
          </div>
        )}
      </div>

      <div className="space-y-5">
        {isLoading ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-4 rounded-2xl p-10" style={{ backgroundColor: "var(--muted)" }}>
            <LoaderCircle className="w-10 h-10 animate-spin" style={{ color: "var(--accent)" }} />
            <p className="text-sm" style={{ color: "var(--muted-foreground)" }}>
              Running the PhotoGuard pipeline...
            </p>
          </div>
        ) : outputs.length > 0 ? (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            {outputs.map((output) => (
              <div
                key={output.label}
                className="rounded-2xl overflow-hidden"
                style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
              >
                <div className="px-4 py-3 border-b" style={{ borderColor: "var(--border)" }}>
                  <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                    {output.label}
                  </h3>
                </div>
                <div className="p-4 space-y-3">
                  <img
                    src={output.dataUrl}
                    alt={output.label}
                    className="w-full rounded-xl object-contain"
                  />
                  <a
                    href={output.dataUrl}
                    download={`${output.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.${outputFormat ?? "png"}`}
                    className="text-sm underline underline-offset-4"
                    style={{ color: "var(--accent)" }}
                  >
                    Download image
                  </a>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex-1 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-10 text-center" style={{ borderColor: "var(--border)", backgroundColor: "var(--muted)" }}>
            <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4" style={{ backgroundColor: "var(--secondary)" }}>
              <ImageIcon className="w-7 h-7" style={{ color: "var(--accent)" }} />
            </div>
            <p className="text-sm leading-relaxed max-w-md" style={{ color: "var(--muted-foreground)" }}>
              Upload an image, paint the protected region, and run the new API workflow to see the
              immunized and edited outputs here.
            </p>
            {error ? (
              <p
                className="text-sm mt-4"
                style={{ color: "var(--destructive)", whiteSpace: "pre-line" }}
              >
                Last error: {error}
              </p>
            ) : null}
          </div>
        )}

        {progress ? (
          <div
            className="rounded-2xl px-4 py-4"
            style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
          >
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div>
                <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                  Request progress
                </h3>
                <p className="text-xs mt-1" style={{ color: "var(--muted-foreground)" }}>
                  Live polling from the API during immunization and edit execution.
                </p>
              </div>
              <p className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                {progress.percent.toFixed(1)}%
              </p>
            </div>

            <div
              className="mt-3 h-2 rounded-full overflow-hidden"
              style={{ backgroundColor: "var(--secondary)" }}
            >
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${progress.percent}%`, backgroundColor: "var(--accent)" }}
              />
            </div>

            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs" style={{ color: "var(--foreground)" }}>
              <p>Status: {formatStage(progress.status) ?? progress.status}</p>
              <p>Stage: {formatStage(progress.stage) ?? progress.stage}</p>
              {progress.iteration && progress.totalIterations ? (
                <p>
                  Iteration: {progress.iteration}/{progress.totalIterations}
                </p>
              ) : null}
              {progress.message ? <p>{progress.message}</p> : null}
            </div>

            {metricEntries.length > 0 ? (
              <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
                {metricEntries.map(([name, value]) => (
                  <div
                    key={name}
                    className="rounded-xl px-3 py-2"
                    style={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                  >
                    <p className="text-[11px] uppercase tracking-[0.16em]" style={{ color: "var(--muted-foreground)" }}>
                      {name}
                    </p>
                    <p className="text-sm mt-1" style={{ color: "var(--foreground)" }}>
                      {value.toFixed(4)}
                    </p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <div
          className="rounded-2xl px-4 py-4"
          style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
        >
          <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
            Processing feedback
          </h3>
          <p className="text-xs mt-1" style={{ color: "var(--muted-foreground)" }}>
            User-facing validation and confirmation messages returned by the integrated API.
          </p>
          <pre
            className="mt-3 text-xs leading-6 whitespace-pre-wrap font-mono"
            style={{ color: error ? "var(--destructive)" : "var(--foreground)" }}
          >
            {feedbackText}
          </pre>
        </div>
      </div>
    </Card>
  );
}
