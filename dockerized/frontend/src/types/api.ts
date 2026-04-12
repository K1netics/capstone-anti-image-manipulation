export interface GeneratedImage {
  label: string;
  dataUrl: string;
}

export type ImmunizationProfile = "stable_diffusion" | "nano_banana_experimental";
export type WorkingResolution = "512" | "1024" | "original";
export type OutputFormat = "png" | "webp";

export interface ProcessRequest {
  requestId?: string;
  image: File;
  mask: File;
  prompt: string;
  seed: string;
  guidanceScale: number;
  numInferenceSteps: number;
  immunize: boolean;
  immunizationProfile: ImmunizationProfile;
  workingResolution: WorkingResolution;
  outputFormat: OutputFormat;
  losslessOutput: boolean;
}

export interface ProcessResponse {
  requestId: string;
  outputs: GeneratedImage[];
  device: string;
  modelSource: string;
  processingMode: "edit" | "immunize";
  statusText: string;
  immunizationProfile: ImmunizationProfile;
  workingResolution: WorkingResolution;
  outputFormat: OutputFormat;
  losslessOutput: boolean;
}

export interface ProcessProgressResponse {
  requestId: string;
  status: "running" | "completed" | "failed";
  stage: string;
  percent: number;
  message: string | null;
  iteration: number | null;
  totalIterations: number | null;
  metrics: Record<string, number>;
  statusText: string;
}

export interface HealthResponse {
  status: "ok" | "offline";
  device: string;
  modelSource: string;
  pipelineReady: boolean;
}
