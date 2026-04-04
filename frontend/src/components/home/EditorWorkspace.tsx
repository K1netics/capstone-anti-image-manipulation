import { useEffect, useRef, useState } from "react";
import { Sparkles, ShieldCheck } from "lucide-react";
import Card from "../ui/Card";
import Button from "../ui/Button";
import UploadZone from "../ui/UploadZone";
import MaskEditor, { type MaskEditorHandle } from "./MaskEditor";
import { createEditorImage, type EditorImageDimensions } from "../../lib/image";
import type { ProcessRequest } from "../../types/api";

interface EditorWorkspaceProps {
  onProcess: (request: ProcessRequest) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

export default function EditorWorkspace({ onProcess, isLoading, error }: EditorWorkspaceProps) {
  const maskEditorRef = useRef<MaskEditorHandle>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [preparedImage, setPreparedImage] = useState<File | null>(null);
  const [editorDimensions, setEditorDimensions] = useState<EditorImageDimensions | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("1234");
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [numInferenceSteps, setNumInferenceSteps] = useState(100);
  const [immunize, setImmunize] = useState(false);
  const [brushSize, setBrushSize] = useState(36);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = async (file: File) => {
    setLocalError(null);

    try {
      const prepared = await createEditorImage(file);
      setPreviewUrl((current) => {
        if (current) {
          URL.revokeObjectURL(current);
        }
        return prepared.previewUrl;
      });
      setPreparedImage(prepared.preparedFile);
      setEditorDimensions(prepared.dimensions);
    } catch (caughtError) {
      setPreparedImage(null);
      setEditorDimensions(null);
      setPreviewUrl((current) => {
        if (current) {
          URL.revokeObjectURL(current);
        }
        return null;
      });
      setLocalError(
        caughtError instanceof Error ? caughtError.message : "Unable to prepare the image.",
      );
    }
  };

  const handleSubmit = async () => {
    if (!preparedImage) {
      setLocalError("Upload an image before running the pipeline.");
      return;
    }

    const maskFile = await maskEditorRef.current?.exportMaskFile();
    if (!maskFile) {
      setLocalError("The mask editor is unavailable right now.");
      return;
    }

    setLocalError(null);
    await onProcess({
      image: preparedImage,
      mask: maskFile,
      prompt,
      seed,
      guidanceScale,
      numInferenceSteps,
      immunize,
    });
  };

  const handleClearMask = () => {
    maskEditorRef.current?.clearMask();
  };

  const combinedError = error ?? localError;

  return (
    <Card className="gap-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p
            className="text-xs uppercase tracking-[0.24em] mb-2"
            style={{ color: "var(--accent)" }}
          >
            Editor
          </p>
          <h2 className="text-2xl font-semibold mb-2" style={{ color: "var(--foreground)" }}>
            Build a protected image, then test the edit
          </h2>
          <p className="text-sm leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
            Upload an image, paint the sensitive regions, and send the image plus mask to the
            separate API running inside this dockerized workspace. The prepared editor keeps the
            whole photo frame instead of cropping everything to a square.
          </p>
        </div>
        <div
          className="rounded-2xl px-4 py-3 flex items-center gap-3"
          style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
        >
          <ShieldCheck className="w-5 h-5" style={{ color: "var(--accent)" }} />
          <span className="text-sm" style={{ color: "var(--foreground)" }}>
            Original backend untouched
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[300px_minmax(0,1fr)] gap-6">
        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-semibold mb-3" style={{ color: "var(--foreground)" }}>
              Source image
            </h3>
            <UploadZone onFileSelect={handleFileSelect} preview={previewUrl} />
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-semibold block mb-2" style={{ color: "var(--foreground)" }}>
                Prompt
              </label>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={4}
                placeholder="Describe the edit you want to test against the protected image"
                className="w-full rounded-2xl px-4 py-3 resize-y"
                style={{
                  backgroundColor: "var(--card)",
                  color: "var(--foreground)",
                  border: "1px solid var(--border)",
                }}
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">Seed</span>
                <input
                  value={seed}
                  onChange={(event) => setSeed(event.target.value)}
                  className="w-full rounded-2xl px-4 py-3"
                  style={{
                    backgroundColor: "var(--card)",
                    color: "var(--foreground)",
                    border: "1px solid var(--border)",
                  }}
                />
              </label>
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">Brush size</span>
                <input
                  type="range"
                  min={8}
                  max={96}
                  step={2}
                  value={brushSize}
                  onChange={(event) => setBrushSize(Number(event.target.value))}
                  className="w-full"
                />
                <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>
                  {brushSize}px
                </span>
              </label>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">
                  Guidance scale: {guidanceScale.toFixed(1)}
                </span>
                <input
                  type="range"
                  min={0.1}
                  max={25}
                  step={0.1}
                  value={guidanceScale}
                  onChange={(event) => setGuidanceScale(Number(event.target.value))}
                  className="w-full"
                />
              </label>
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">
                  Inference steps: {numInferenceSteps}
                </span>
                <input
                  type="range"
                  min={10}
                  max={250}
                  step={5}
                  value={numInferenceSteps}
                  onChange={(event) => setNumInferenceSteps(Number(event.target.value))}
                  className="w-full"
                />
              </label>
            </div>

            <label
              className="flex items-center gap-3 rounded-2xl px-4 py-3"
              style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
            >
              <input
                type="checkbox"
                checked={immunize}
                onChange={(event) => setImmunize(event.target.checked)}
              />
              <div>
                <span className="block text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                  Immunize before editing
                </span>
                <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>
                  Returns both the immunized image and the edited result.
                </span>
              </div>
            </label>
          </div>
        </div>

        <div className="space-y-5">
          <MaskEditor
            ref={maskEditorRef}
            imageUrl={previewUrl}
            imageWidth={editorDimensions?.width ?? null}
            imageHeight={editorDimensions?.height ?? null}
            brushSize={brushSize}
          />

          {combinedError ? (
            <div
              className="rounded-2xl px-4 py-3 text-sm"
              style={{
                backgroundColor: "rgba(212,24,61,0.08)",
                color: "var(--destructive)",
                border: "1px solid rgba(212,24,61,0.18)",
              }}
            >
              {combinedError}
            </div>
          ) : null}

          <div className="flex flex-col sm:flex-row gap-3">
            <Button fullWidth onClick={handleSubmit} disabled={isLoading || !preparedImage}>
              <span className="inline-flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                {isLoading ? "Processing image..." : "Run PhotoGuard API"}
              </span>
            </Button>
            <Button fullWidth variant="ghost" onClick={handleClearMask} disabled={!preparedImage || isLoading}>
              Clear mask
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
