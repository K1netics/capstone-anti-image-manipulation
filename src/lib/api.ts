import type { HealthResponse, ProcessRequest, ProcessResponse } from "../types/api";

const DEFAULT_API_BASE_URL = "/api";

function stripTrailingSlash(value: string) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = configuredBaseUrl
  ? stripTrailingSlash(configuredBaseUrl)
  : DEFAULT_API_BASE_URL;

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

export async function processImage(payload: ProcessRequest): Promise<ProcessResponse> {
  const formData = new FormData();
  formData.append("image", payload.image);
  formData.append("mask", payload.mask);
  formData.append("prompt", payload.prompt);
  formData.append("seed", payload.seed);
  formData.append("guidance_scale", String(payload.guidanceScale));
  formData.append("num_inference_steps", String(payload.numInferenceSteps));
  formData.append("immunize", String(payload.immunize));

  const response = await fetch(`${API_BASE_URL}/process`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<ProcessResponse>;
}
