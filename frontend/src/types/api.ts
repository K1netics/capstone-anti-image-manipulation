export interface GeneratedImage {
  label: string;
  dataUrl: string;
}

export interface ProcessRequest {
  image: File;
  mask: File;
  prompt: string;
  seed: string;
  guidanceScale: number;
  numInferenceSteps: number;
  immunize: boolean;
}

export interface ProcessResponse {
  outputs: GeneratedImage[];
  device: string;
  modelSource: string;
  processingMode: "edit" | "immunize";
}

export interface HealthResponse {
  status: "ok" | "offline";
  device: string;
  modelSource: string;
  pipelineReady: boolean;
}
