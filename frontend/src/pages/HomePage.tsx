import { startTransition, useState } from "react";
import EditorWorkspace from "../components/home/EditorWorkspace";
import OutputSection from "../components/home/OutputSection";
import BackendStatus from "../components/home/BackendStatus";
import FeaturesGrid from "../components/home/FeaturesGrid";
import { processImage } from "../lib/api";
import type { GeneratedImage, ProcessRequest, ProcessResponse } from "../types/api";

export default function HomePage() {
  const [outputs, setOutputs] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Pick<ProcessResponse, "device" | "modelSource"> | null>(null);

  const handleProcess = async (request: ProcessRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await processImage(request);
      startTransition(() => {
        setOutputs(response.outputs);
        setMetadata({ device: response.device, modelSource: response.modelSource });
      });
    } catch (caughtError) {
      setOutputs([]);
      setMetadata(null);
      setError(caughtError instanceof Error ? caughtError.message : "Unable to process the image.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="w-full">
      <section className="px-6 pt-20 pb-12 text-center" style={{ backgroundColor: "var(--background)" }}>
        <p className="text-xs md:text-sm uppercase tracking-[0.28em] mb-4" style={{ color: "var(--accent)" }}>
          Dockerized workspace
        </p>
        <h1 className="text-3xl md:text-5xl font-semibold mb-4" style={{ color: "var(--foreground)" }}>
          PhotoGuard without Gradio in the loop
        </h1>
        <p className="max-w-3xl mx-auto text-sm md:text-base leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
          This packaged workspace keeps the original demo untouched and runs the React-to-API flow behind a production-friendly proxy.
        </p>
      </section>

      <div className="max-w-7xl mx-auto px-4 md:px-8 pb-10 space-y-6">
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] gap-6" style={{ alignItems: "start" }}>
          <EditorWorkspace onProcess={handleProcess} isLoading={isLoading} error={error} />
          <OutputSection
            outputs={outputs}
            isLoading={isLoading}
            error={error}
            device={metadata?.device ?? null}
            modelSource={metadata?.modelSource ?? null}
          />
        </div>
        <BackendStatus />
        <FeaturesGrid />
      </div>
    </main>
  );
}
