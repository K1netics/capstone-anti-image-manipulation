declare global {
  interface Window {
    __IMMUNA_CONFIG__?: {
      API_UPSTREAM?: string;
    };
  }
}

import type {
  HealthResponse,
  ProcessProgressResponse,
  ProcessRequest,
  ProcessResponse,
} from "../types/api";

const DEFAULT_API_BASE_URL = "/api";

function stripTrailingSlash(value: string) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function getRuntimeApiBaseUrl() {
  if (typeof window === "undefined") {
    return null;
  }

  const configured = window.__IMMUNA_CONFIG__?.API_UPSTREAM?.trim();
  return configured ? stripTrailingSlash(configured) : null;
}

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = getRuntimeApiBaseUrl()
  ?? (configuredBaseUrl ? stripTrailingSlash(configuredBaseUrl) : DEFAULT_API_BASE_URL);

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

export async function processImage(payload: ProcessRequest): Promise<ProcessResponse> {
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

  const response = await fetch(`${API_BASE_URL}/process`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<ProcessResponse>;
}
