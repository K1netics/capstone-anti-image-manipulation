export interface GeneratedImage {
  label: string;
  dataUrl: string;
}

export type ImmunizationProfile =
  | "stable_diffusion"
  | "full_regeneration_scaffold"
  | "instruction_editing_scaffold"
  | "controlnet_scaffold"
  | "style_transfer_scaffold"
  | "artist_cloak"
  | "text_aware_scaffold"
  | "adversarial_hardened_scaffold"
  | "nano_banana_experimental"
  | "nano_banana_2"
  | "nano_banana_2_hard_block"
  | "nano_banana_2_distortion";
export type WorkingResolution = "512" | "1024" | "original";
export type OutputFormat = "png" | "webp";
export type DefenseCanvas =
  | "profile_default"
  | "512"
  | "640"
  | "768"
  | "1024"
  | "working";

export interface BackendSettings {
  defenseCanvas: DefenseCanvas;
  forceFullStrength: boolean;
  immunizationIters: string;
  eotSamples: string;
  maxPromptVariants: string;
  denoiserStrength: string;
  referenceConfusionStrength: string;
  identityDriftStrength: string;
  semanticBoundaryStrength: string;
  watermarkStrength: string;
  tripwireGlobalStrength: string;
}

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
  backendSettings?: Partial<BackendSettings>;
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

export interface ProcessStartResponse {
  requestId: string;
  status: "accepted";
  statusText: string;
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
