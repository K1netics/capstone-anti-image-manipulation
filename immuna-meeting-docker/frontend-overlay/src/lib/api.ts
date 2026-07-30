declare global {
  interface Window {
    __IMMUNA_CONFIG__?: {
      API_UPSTREAM?: string;
    };
  }
}

type HealthResponse = {
  status: string;
  device?: string;
  modelSource?: string;
  pipelineReady?: boolean;
};

type ProcessOutput = {
  label?: string;
  dataUrl?: string;
};

type ProcessResponse = {
  outputs?: ProcessOutput[];
};

const DEFAULT_API_UPSTREAM = "http://localhost:8000";

function stripTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

function getApiUpstream() {
  if (typeof window === "undefined") {
    return DEFAULT_API_UPSTREAM;
  }

  const configured = window.__IMMUNA_CONFIG__?.API_UPSTREAM?.trim();
  return configured ? stripTrailingSlash(configured) : DEFAULT_API_UPSTREAM;
}

function buildApiUrl(path: string) {
  return `${getApiUpstream()}${path.startsWith("/") ? path : `/${path}`}`;
}

async function parseError(response: Response) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }
  } catch {
    // Ignore parse failures and fall back to the status text.
  }

  return response.statusText || "Request failed";
}

function fileNameStem(file: File) {
  const index = file.name.lastIndexOf(".");
  return index > 0 ? file.name.slice(0, index) : file.name;
}

async function createFullMask(imageFile: File): Promise<File> {
  const imageBitmap = await createImageBitmap(imageFile);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Canvas context unavailable");
    }
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => {
        if (value) {
          resolve(value);
          return;
        }
        reject(new Error("Failed to create mask image"));
      }, "image/png");
    });

    return new File([blob], `${fileNameStem(imageFile)}-mask.png`, { type: "image/png" });
  } finally {
    imageBitmap.close();
  }
}

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(buildApiUrl("/health"), { signal });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json() as Promise<HealthResponse>;
}

export async function submitImmunizeRequest(image: File): Promise<string> {
  const mask = await createFullMask(image);
  const formData = new FormData();
  formData.append("image", image);
  formData.append("mask", mask);
  formData.append("prompt", "");
  formData.append("seed", "1234");
  formData.append("guidance_scale", "7.5");
  formData.append("num_inference_steps", "75");
  formData.append("immunize", "true");
  formData.append("immunization_profile", "stable_diffusion");
  formData.append("working_resolution", "1024");
  formData.append("output_format", "png");
  formData.append("lossless_output", "true");

  const response = await fetch(buildApiUrl("/process"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const payload = (await response.json()) as ProcessResponse;
  const outputs = payload.outputs;
  const dataUrl = Array.isArray(outputs) ? outputs[0]?.dataUrl : null;
  if (typeof dataUrl !== "string" || dataUrl.length === 0) {
    throw new Error("Backend returned no output image");
  }
  return dataUrl;
}
