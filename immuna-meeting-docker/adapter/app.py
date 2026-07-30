from __future__ import annotations

import base64
import io
import os
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

BACKEND_URL = os.getenv("PHOTOGUARD_API_URL", "http://backend:8000").rstrip("/")
DEFAULT_PROFILE = os.getenv("PHOTOGUARD_ADAPTER_PROFILE", "stable_diffusion")
DEFAULT_WORKING_RESOLUTION = os.getenv("PHOTOGUARD_ADAPTER_WORKING_RESOLUTION", "1024")
DEFAULT_OUTPUT_FORMAT = os.getenv("PHOTOGUARD_ADAPTER_OUTPUT_FORMAT", "png")
DEFAULT_GUIDANCE_SCALE = os.getenv("PHOTOGUARD_ADAPTER_GUIDANCE_SCALE", "7.5")
DEFAULT_NUM_INFERENCE_STEPS = os.getenv("PHOTOGUARD_ADAPTER_NUM_INFERENCE_STEPS", "100")
DEFAULT_SEED = os.getenv("PHOTOGUARD_ADAPTER_SEED", "1234")

app = FastAPI(title="Immuna Meeting Adapter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_full_protection_mask(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.new("L", image.size, color=255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return buffer.getvalue()


def _extract_first_output(payload: dict[str, Any]) -> tuple[bytes, str]:
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise HTTPException(status_code=502, detail="Backend returned no output images.")
    data_url = outputs[0].get("dataUrl") if isinstance(outputs[0], dict) else None
    if not isinstance(data_url, str) or "," not in data_url:
        raise HTTPException(status_code=502, detail="Backend returned an invalid image payload.")
    header, encoded = data_url.split(",", 1)
    content_type = "image/png"
    if header.startswith("data:") and ";" in header:
        content_type = header[5:].split(";", 1)[0]
    return base64.b64decode(encoded), content_type


@app.get("/health")
async def health() -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
        return JSONResponse(status_code=response.status_code, content=response.json())
    except Exception as exc:  # pragma: no cover - runtime path
        return JSONResponse(
            status_code=503,
            content={"status": "offline", "detail": f"Adapter could not reach backend: {exc}"},
        )


@app.post("/embed")
async def embed(image: UploadFile = File(...)) -> Response:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    mask_bytes = _make_full_protection_mask(image_bytes)

    files = {
        "image": (image.filename or "image.png", image_bytes, image.content_type or "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    data = {
        "prompt": "",
        "seed": DEFAULT_SEED,
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
        "immunize": "true",
        "immunization_profile": DEFAULT_PROFILE,
        "working_resolution": DEFAULT_WORKING_RESOLUTION,
        "output_format": DEFAULT_OUTPUT_FORMAT,
        "lossless_output": "true",
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(f"{BACKEND_URL}/process", data=data, files=files)
    except Exception as exc:  # pragma: no cover - runtime path
        raise HTTPException(status_code=502, detail=f"Adapter could not reach backend: {exc}") from exc

    if response.status_code >= 400:
        detail: str
        try:
            payload = response.json()
            detail = str(payload.get("detail") or payload)
        except Exception:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    output_bytes, content_type = _extract_first_output(response.json())
    return Response(content=output_bytes, media_type=content_type)
