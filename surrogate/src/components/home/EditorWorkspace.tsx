import { useEffect, useRef, useState } from "react";
import { Sparkles, ShieldCheck } from "lucide-react";
import Card from "../ui/Card";
import Button from "../ui/Button";
import UploadZone from "../ui/UploadZone";
import MaskEditor, { type MaskEditorHandle } from "./MaskEditor";
import {
  inspectEditorImage,
  type EditorImageDimensions,
} from "../../lib/image";
import {
  IMMUNIZATION_PROFILE_GROUPS,
  profileLabel,
  profileSummary,
} from "../../lib/immunizationProfiles";
import type {
  DefenseCanvas,
  ImmunizationProfile,
  OutputFormat,
  ProcessRequest,
  WorkingResolution,
} from "../../types/api";

interface EditorWorkspaceProps {
  onProcess: (request: ProcessRequest) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

export default function EditorWorkspace({ onProcess, isLoading, error }: EditorWorkspaceProps) {
  const maskEditorRef = useRef<MaskEditorHandle>(null);
  const previewUrlRef = useRef<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [editorDimensions, setEditorDimensions] = useState<EditorImageDimensions | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("1234");
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [numInferenceSteps, setNumInferenceSteps] = useState(100);
  const [immunize, setImmunize] = useState(false);
  const [immunizationProfile, setImmunizationProfile] = useState<ImmunizationProfile>("surrogate_deevid");
  const [workingResolution, setWorkingResolution] = useState<WorkingResolution>("1024");
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("png");
  const [losslessOutput, setLosslessOutput] = useState(true);
  const [brushSize, setBrushSize] = useState(36);
  const [defenseCanvas, setDefenseCanvas] = useState<DefenseCanvas>("profile_default");
  const [forceFullStrength, setForceFullStrength] = useState(true);
  const [immunizationIters, setImmunizationIters] = useState("");
  const [eotSamples, setEotSamples] = useState("");
  const [maxPromptVariants, setMaxPromptVariants] = useState("");
  const [denoiserStrength, setDenoiserStrength] = useState("");
  const [referenceConfusionStrength, setReferenceConfusionStrength] = useState("");
  const [identityDriftStrength, setIdentityDriftStrength] = useState("");
  const [semanticBoundaryStrength, setSemanticBoundaryStrength] = useState("");
  const [watermarkStrength, setWatermarkStrength] = useState("");
  const [tripwireGlobalStrength, setTripwireGlobalStrength] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  const replacePreviewUrl = (nextPreviewUrl: string | null) => {
    if (previewUrlRef.current && previewUrlRef.current !== nextPreviewUrl) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    previewUrlRef.current = nextPreviewUrl;
    setPreviewUrl(nextPreviewUrl);
  };

  const handleFileSelect = async (file: File) => {
    setLocalError(null);

    try {
      const inspected = await inspectEditorImage(file);
      replacePreviewUrl(inspected.previewUrl);
      setSelectedImage(file);
      setEditorDimensions(inspected.dimensions);
    } catch (caughtError) {
      setSelectedImage(null);
      setEditorDimensions(null);
      replacePreviewUrl(null);
      setLocalError(
        caughtError instanceof Error ? caughtError.message : "Unable to prepare the image.",
      );
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
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
      image: selectedImage,
      mask: maskFile,
      prompt,
      seed,
      guidanceScale,
      numInferenceSteps,
      immunize,
      immunizationProfile,
      workingResolution,
      outputFormat,
      losslessOutput,
      backendSettings: {
        defenseCanvas,
        forceFullStrength,
        immunizationIters,
        eotSamples,
        maxPromptVariants,
        denoiserStrength,
        referenceConfusionStrength,
        identityDriftStrength,
        semanticBoundaryStrength,
        watermarkStrength,
        tripwireGlobalStrength,
      },
    });
  };

  const handleClearMask = () => {
    maskEditorRef.current?.clearMask();
  };

  const combinedError = localError ?? error;
  const selectedProfileLabel = profileLabel(immunizationProfile);
  const selectedProfileSummary = profileSummary(immunizationProfile);

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
            separate API running inside this integrated workspace. The prepared editor keeps the
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
            <UploadZone
              onFileSelect={handleFileSelect}
              onError={setLocalError}
              preview={previewUrl}
            />
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
                  First generates a protected image, then runs the edit against that protected version.
                </span>
              </div>
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">Working resolution</span>
                <select
                  value={workingResolution}
                  onChange={(event) => setWorkingResolution(event.target.value as WorkingResolution)}
                  className="w-full rounded-2xl px-4 py-3"
                  style={{
                    backgroundColor: "var(--card)",
                    color: "var(--foreground)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <option value="512">512 px (legacy fast)</option>
                  <option value="1024">1024 px (recommended)</option>
                  <option value="original">Original size (heaviest)</option>
                </select>
              </label>
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">Output format</span>
                <select
                  value={outputFormat}
                  onChange={(event) => setOutputFormat(event.target.value as OutputFormat)}
                  className="w-full rounded-2xl px-4 py-3"
                  style={{
                    backgroundColor: "var(--card)",
                    color: "var(--foreground)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <option value="png">PNG</option>
                  <option value="webp">WebP</option>
                </select>
              </label>
            </div>

            <div
              className="rounded-2xl px-4 py-4 space-y-4"
              style={{
                backgroundColor: "var(--muted)",
                border: "1px solid var(--border)",
                opacity: immunize ? 1 : 0.7,
              }}
            >
              <div>
                <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                  Backend settings
                </h3>
                <p className="text-xs mt-1 leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
                  Optional per-run backend overrides for DeeVid testing. Leave fields blank to keep
                  the selected profile defaults.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Defense canvas</span>
                  <select
                    value={defenseCanvas}
                    onChange={(event) => setDefenseCanvas(event.target.value as DefenseCanvas)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{
                      backgroundColor: "var(--card)",
                      color: "var(--foreground)",
                      border: "1px solid var(--border)",
                    }}
                  >
                    <option value="profile_default">Profile default</option>
                    <option value="512">512 px</option>
                    <option value="640">640 px</option>
                    <option value="768">768 px</option>
                    <option value="1024">1024 px</option>
                    <option value="working">Use full working canvas</option>
                  </select>
                </label>

                <label
                  className="flex items-center gap-3 rounded-2xl px-4 py-3"
                  style={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                >
                  <input
                    type="checkbox"
                    checked={forceFullStrength}
                    onChange={(event) => setForceFullStrength(event.target.checked)}
                    disabled={!immunize}
                  />
                  <div>
                    <span className="block text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                      Force full strength
                    </span>
                    <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>
                      Fails closed instead of allowing the lower-memory fallback profile.
                    </span>
                  </div>
                </label>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">PGD iterations</span>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    inputMode="numeric"
                    placeholder="default"
                    value={immunizationIters}
                    onChange={(event) => setImmunizationIters(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">EOT samples</span>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    inputMode="numeric"
                    placeholder="default"
                    value={eotSamples}
                    onChange={(event) => setEotSamples(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Prompt variants</span>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    inputMode="numeric"
                    placeholder="default"
                    value={maxPromptVariants}
                    onChange={(event) => setMaxPromptVariants(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Denoiser strength</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    placeholder="default"
                    value={denoiserStrength}
                    onChange={(event) => setDenoiserStrength(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Reference confusion</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    placeholder="default"
                    value={referenceConfusionStrength}
                    onChange={(event) => setReferenceConfusionStrength(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Identity drift</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    placeholder="default"
                    value={identityDriftStrength}
                    onChange={(event) => setIdentityDriftStrength(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Semantic boundary</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    placeholder="default"
                    value={semanticBoundaryStrength}
                    onChange={(event) => setSemanticBoundaryStrength(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Watermark</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    placeholder="default"
                    value={watermarkStrength}
                    onChange={(event) => setWatermarkStrength(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
                <label className="text-sm" style={{ color: "var(--foreground)" }}>
                  <span className="block font-semibold mb-2">Global anchor</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    placeholder="default"
                    value={tripwireGlobalStrength}
                    onChange={(event) => setTripwireGlobalStrength(event.target.value)}
                    disabled={!immunize}
                    className="w-full rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--card)", color: "var(--foreground)", border: "1px solid var(--border)" }}
                  />
                </label>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="text-sm" style={{ color: "var(--foreground)" }}>
                <span className="block font-semibold mb-2">Immunization profile</span>
                <select
                  value={immunizationProfile}
                  onChange={(event) => setImmunizationProfile(event.target.value as ImmunizationProfile)}
                  disabled={!immunize}
                  className="w-full rounded-2xl px-4 py-3"
                  style={{
                    backgroundColor: "var(--card)",
                    color: "var(--foreground)",
                    border: "1px solid var(--border)",
                    opacity: immunize ? 1 : 0.65,
                  }}
                >
                  {IMMUNIZATION_PROFILE_GROUPS.map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {group.options.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <span className="block text-xs mt-2" style={{ color: "var(--muted-foreground)" }}>
                  Pick one defense profile for the immunization step. `Surrogate hybrid` keeps both
                  SD and NB2-style surrogates active for the broadest current coverage, `Nano Banana 2 layered`
                  is the cleaner NB2-focused option, and the strict profiles may fail closed on tight GPUs
                  instead of silently weakening.
                </span>
                {immunize ? (
                  <div
                    className="mt-3 rounded-2xl px-4 py-3"
                    style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
                  >
                    <p className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                      {selectedProfileLabel}
                    </p>
                    <p className="text-xs mt-1 leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
                      {selectedProfileSummary}
                    </p>
                  </div>
                ) : null}
              </label>

              <label
                className="flex items-center gap-3 rounded-2xl px-4 py-3"
                style={{ backgroundColor: "var(--muted)", border: "1px solid var(--border)" }}
              >
                <input
                  type="checkbox"
                  checked={losslessOutput}
                  onChange={(event) => setLosslessOutput(event.target.checked)}
                  disabled={outputFormat !== "webp"}
                />
                <div>
                  <span className="block text-sm font-semibold" style={{ color: "var(--foreground)" }}>
                    Lossless output
                  </span>
                  <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>
                    Applies to WebP. PNG stays lossless automatically.
                  </span>
                </div>
              </label>
            </div>
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
                whiteSpace: "pre-line",
              }}
            >
              {combinedError}
            </div>
          ) : null}

          <div className="flex flex-col sm:flex-row gap-3">
            <Button fullWidth onClick={handleSubmit} disabled={isLoading || !selectedImage}>
              <span className="inline-flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                {isLoading ? "Processing image..." : "Run PhotoGuard API"}
              </span>
            </Button>
            <Button fullWidth variant="ghost" onClick={handleClearMask} disabled={!selectedImage || isLoading}>
              Clear mask
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
