import { Image as ImageIcon, LoaderCircle } from "lucide-react";
import Card from "../ui/Card";
import type { GeneratedImage } from "../../types/api";

interface OutputSectionProps {
  outputs: GeneratedImage[];
  isLoading: boolean;
  error: string | null;
  device: string | null;
  modelSource: string | null;
}

export default function OutputSection({
  outputs,
  isLoading,
  error,
  device,
  modelSource,
}: OutputSectionProps) {
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
          </div>
        )}
      </div>

      {isLoading ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-4 rounded-2xl" style={{ backgroundColor: "var(--muted)" }}>
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
                  download={`${output.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.webp`}
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
            <p className="text-sm mt-4" style={{ color: "var(--destructive)" }}>
              Last error: {error}
            </p>
          ) : null}
        </div>
      )}
    </Card>
  );
}
