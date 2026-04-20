import type {
  HealthResponse,
  ProcessProgressResponse,
  ProcessRequest,
  ProcessResponse,
  ProcessStartResponse,
} from "../types/api";

const DEFAULT_API_BASE_URL = "/api";

function stripTrailingSlash(value: string) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function getRuntimeApiBaseUrl() {
  if (typeof window === "undefined") {
    return null;
  }

  const params = new URLSearchParams(window.location.search);
  const backendVersion = params.get("backend")?.trim();
  if (backendVersion === "2.3.1") {
    return "/api/2.3.1";
  }
  if (backendVersion === "2.3.2") {
    return "/api/2.3.2";
  }
  return null;
}

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = configuredBaseUrl
  ? stripTrailingSlash(configuredBaseUrl)
  : getRuntimeApiBaseUrl() ?? DEFAULT_API_BASE_URL;

async function parseError(response: Response) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }
  } catch {
    // Ignore JSON parse errors and fall back to status text.
  }

  return response.statusText || "Request failed";
}

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, { signal });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json() as Promise<HealthResponse>;
}

export function createRequestId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `pg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export async function getProcessProgress(
  requestId: string,
  signal?: AbortSignal,
): Promise<ProcessProgressResponse | null> {
  const response = await fetch(`${API_BASE_URL}/progress/${encodeURIComponent(requestId)}`, { signal });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json() as Promise<ProcessProgressResponse>;
}

function buildProcessFormData(payload: ProcessRequest): FormData {
  const formData = new FormData();
  if (payload.requestId) {
    formData.append("request_id", payload.requestId);
  }
  formData.append("image", payload.image);
  formData.append("mask", payload.mask);
  formData.append("prompt", payload.prompt);
  formData.append("seed", payload.seed);
  formData.append("guidance_scale", String(payload.guidanceScale));
  formData.append("num_inference_steps", String(payload.numInferenceSteps));
  formData.append("immunize", String(payload.immunize));
  formData.append("immunization_profile", payload.immunizationProfile);
  formData.append("working_resolution", payload.workingResolution);
  formData.append("output_format", payload.outputFormat);
  formData.append("lossless_output", String(payload.losslessOutput));
  formData.append("defense_canvas", payload.backendSettings.defenseCanvas);
  formData.append("force_full_strength", String(payload.backendSettings.forceFullStrength));
  formData.append("immunization_iters", payload.backendSettings.immunizationIters);
  formData.append("eot_samples", payload.backendSettings.eotSamples);
  formData.append("max_prompt_variants", payload.backendSettings.maxPromptVariants);
  formData.append("denoiser_strength", payload.backendSettings.denoiserStrength);
  formData.append("reference_confusion_strength", payload.backendSettings.referenceConfusionStrength);
  formData.append("identity_drift_strength", payload.backendSettings.identityDriftStrength);
  formData.append("semantic_boundary_strength", payload.backendSettings.semanticBoundaryStrength);
  formData.append("watermark_strength", payload.backendSettings.watermarkStrength);
  formData.append("tripwire_global_strength", payload.backendSettings.tripwireGlobalStrength);
  return formData;
}

export async function getProcessResult(
  requestId: string,
  signal?: AbortSignal,
): Promise<ProcessResponse | null> {
  const response = await fetch(`${API_BASE_URL}/result/${encodeURIComponent(requestId)}`, { signal });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json() as Promise<ProcessResponse>;
}

export async function startProcessImage(payload: ProcessRequest): Promise<ProcessStartResponse> {
  const response = await fetch(`${API_BASE_URL}/process/start`, {
    method: "POST",
    body: buildProcessFormData(payload),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<ProcessStartResponse>;
}

export async function processImage(payload: ProcessRequest): Promise<ProcessResponse> {
  const response = await fetch(`${API_BASE_URL}/process`, {
    method: "POST",
    body: buildProcessFormData(payload),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<ProcessResponse>;
}
